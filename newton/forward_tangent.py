"""Full forward-mode tangent (Jvp) of the root scores w.r.t. theta.

``jvp_root_scores(static, theta, v, sv)`` returns ``d(Pi_root)/dtheta . v`` by threading tangents
through the whole forward solve, mirroring ``solve_e_pi`` + ``pi_wave_forward``:

  1. parameter tangent ``dparams = d(extract_parameters)/dtheta . v`` (forward-mode autodiff);
  2. E-step tangent fixed point (``e_tangent_fixed_point``);
  3. Pi-wave tangent: per wave (topological order), the cross-wave ``dts`` tangent then the
     self-loop tangent solved to convergence (the same true fixed point the adjoint differentiates).

This is the ``J`` of the Gauss-Newton operator ``M = J^T B J``; the matching ``J^T`` reuses the
existing backward (see ``newton/ggn.py``).
"""

from __future__ import annotations

import torch
from torch.func import jvp

from kbench.core.parameters.extract_parameters import (
    as_item_param, as_item_state, extract_parameters_uniform,
)
from kbench.core.kernels.dts_fused import compute_dts_forward
from kbench.core.kernels.dts_tangent import compute_dts_tangent
from kbench.core.kernels.e_step_tangent import e_tangent_fixed_point
from kbench.core.kernels.wave_tangent import (
    compute_wave_step_tangent, compute_wave_step_tangent_selfloop,
)


def param_jvp_uniform(static, theta, v):
    """Forward-mode tangent of extract_parameters_uniform along v (use_col_weights=False path)."""
    unnorm_row_max = static.state_helpers["unnorm_row_max"].to(device=theta.device, dtype=theta.dtype)

    def f(th):
        return extract_parameters_uniform(
            th, unnorm_row_max, statewise=static.statewise, itemwise=static.itemwise
        )

    primals, tangents = jvp(f, (theta,), (v,))
    # (log_pS, log_pD, log_pL, max_coupling)
    return tangents


def wave_step_constants(sv, S):
    """Base per-item wave-step constants (mirrors pi_wave_forward)."""
    item_rows = int(sv["E"].shape[0])
    e_item = as_item_state(sv["E"], S, item_rows)
    ebar_item = as_item_state(sv["Ebar"], S, item_rows)
    e_s1_item = as_item_state(sv["E_s1"], S, item_rows)
    e_s2_item = as_item_state(sv["E_s2"], S, item_rows)
    mc_item = as_item_state(sv["max_coupling"].squeeze(-1), S, item_rows)
    pd_item = as_item_state(sv["log_pD"], S, item_rows)
    ps_item = as_item_state(sv["log_pS"], S, item_rows)
    return {
        "dl": 1.0 + pd_item + e_item, "ebar": ebar_item, "e": e_item,
        "sl1": ps_item + e_s2_item, "sl2": ps_item + e_s1_item,
        "mc": mc_item, "leaf": ps_item,
        "pd_param": as_item_param(sv["log_pD"], item_rows, S),
        "ps_param": as_item_param(sv["log_pS"], item_rows, S),
    }


def _default_tol(dtype):
    return 1e-12 if dtype == torch.float64 else 1e-6


def _wave_tangent_constants(static, theta, v, sv, S, e_tol, raw_out=None):
    """E + parameter tangents assembled into the wave-step tangent constants."""
    dlog_pS, dlog_pD, dlog_pL, dmax_coupling = param_jvp_uniform(static, theta, v)
    sh = static.state_helpers
    dE, dE_s1, dE_s2, dEbar = e_tangent_fixed_point(
        sv["E"], dlog_pS, dlog_pD, dlog_pL, dmax_coupling,
        sv["log_pS"], sv["log_pD"], sv["log_pL"], sv["max_coupling"], sv["col_log_probs"],
        sh["node_parent"], sh["node_child1"], sh["node_child2"], int(sh["max_ancestor_depth"]),
        max_iter=int(static.solver_options.e_max_iter), tol=e_tol, use_col_weights=False,
    )
    if raw_out is not None:
        raw_out.update(dlog_pS=dlog_pS, dlog_pD=dlog_pD, dlog_pL=dlog_pL,
                       dmax_coupling=dmax_coupling, dE=dE, dE_s1=dE_s1, dE_s2=dE_s2, dEbar=dEbar)
    S_ = S
    item_rows = int(sv["E"].shape[0])
    de_item = as_item_state(dE, S_, item_rows)
    debar_item = as_item_state(dEbar, S_, item_rows)
    de_s1_item = as_item_state(dE_s1, S_, item_rows)
    de_s2_item = as_item_state(dE_s2, S_, item_rows)
    dpd_item = as_item_state(dlog_pD, S_, item_rows)
    dps_item = as_item_state(dlog_pS, S_, item_rows)
    dmc_item = as_item_state(dmax_coupling.squeeze(-1), S_, item_rows)
    return {
        "dDL": dpd_item + de_item, "dEbar": debar_item, "dE": de_item,
        "dSL1": dps_item + de_s2_item, "dSL2": dps_item + de_s1_item,
        "dMC": dmc_item, "dleaf": dps_item,
        "dpd_param": as_item_param(dlog_pD, item_rows, S_),
        "dps_param": as_item_param(dlog_pS, item_rows, S_),
    }


def jvp_root_scores(static, theta, v, sv, *, self_tol=None, self_max_iter=200, e_tol=None,
                    self_iters=None, return_full=False, keep_d_dts=True, fused_selfloop=True):
    """d(Pi_root)/dtheta . v  -> tensor [n_root_rows, S].

    ``self_iters`` (int): run the per-wave self-loop for a FIXED number of Jacobi steps with
    no per-iteration host sync — this matches the primal forward's ``pi_iters`` truncation
    (N Jacobi steps from a zero tangent == the N-term Neumann partial sum the primal uses) and
    streams the tangent sweep without CPU<->GPU stalls. ``self_iters=None`` (default) keeps the
    adaptive converge-to-``self_tol`` loop used by the fp64 verification gates.

    With ``return_full=True`` returns (root_tangents, full) where ``full`` carries everything the
    exact-HVP tangent-adjoint sweep needs: dPi/dPibar [C,S] buffers, per-wave d_dts (dict keyed by
    wave start), the tangent constants dict (dDL/dEbar/dE/dSL1/dSL2/dMC/dleaf/dpd_param/
    dps_param), and the raw parameter tangents (dlog_pS, dlog_pD, dlog_pL, dmax_coupling) plus
    the E tangents (dE*, dE_s1, dE_s2, dEbar).
    """
    sh, wl = static.state_helpers, static.wave_layout
    S = int(sh["S"])
    if self_tol is None:
        self_tol = _default_tol(theta.dtype)
    if e_tol is None:
        e_tol = _default_tol(theta.dtype)
    item_idx = static.rate_item_idx
    leaf_state_idx = wl["leaf_state_index"].to(torch.int32)
    c1, c2, parent = sh["node_child1"], sh["node_child2"], sh["node_parent"]
    mad = int(sh["max_ancestor_depth"])

    base = wave_step_constants(sv, S)
    raw = {} if return_full else None
    dcst = _wave_tangent_constants(static, theta, v, sv, S, e_tol, raw_out=raw)

    pi = sv["pi_wave"]
    pibar = sv["pibar_wave"]
    C = int(pi.shape[0])
    dpi = torch.zeros((C, S), device=pi.device, dtype=pi.dtype)
    dpibar = torch.zeros((C, S), device=pi.device, dtype=pi.dtype)
    d_dts_by_ws = {} if return_full else None

    def step(dPi_out, dts_r, d_dts, ws, W, has_leaf, store):
        compute_wave_step_tangent(
            pi, dpi, dPi_out, ws, W, S,
            base["mc"], dcst["dMC"], base["dl"], dcst["dDL"], base["ebar"], dcst["dEbar"],
            base["e"], dcst["dE"], base["sl1"], dcst["dSL1"], base["sl2"], dcst["dSL2"],
            sv["col_log_probs"], c1, c2, parent, mad, dts_r, d_dts,
            leaf_state_idx=leaf_state_idx, leaf_logp=base["leaf"], dleaf_logp=dcst["dleaf"],
            item_idx=item_idx, dPibar_out=(dpibar if store else None),
            has_leaf_term=has_leaf, input_ws=None, use_col_weights=False,
        )

    for meta in wl["wave_metas"]:
        ws, W = int(meta["start"]), int(meta["W"])
        has_splits = "sl" in meta
        has_leaf = not has_splits
        if has_splits:
            dts_r = compute_dts_forward(
                pi, pibar, meta["sl"], meta["sr"], c1, c2, W, meta["reduce_idx"],
                base["pd_param"], base["ps_param"], item_idx=item_idx,
                log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"), ge2_max_fanout=meta.get("ge2_max_fanout"),
                item_offset=ws,
            )
            d_dts = compute_dts_tangent(
                pi, pibar, dpi, dpibar, meta["sl"], meta["sr"], c1, c2, W, meta["reduce_idx"],
                base["pd_param"], base["ps_param"], dcst["dpd_param"], dcst["dps_param"], dts_r, item_idx,
                log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"), ge2_max_fanout=meta.get("ge2_max_fanout"),
                item_offset=ws,
            )
        else:
            dts_r = d_dts = None
        if return_full and d_dts is not None and keep_d_dts:
            d_dts_by_ws[ws] = d_dts

        if self_iters is not None and fused_selfloop:
            # fixed-count, sync-free Jacobi matching the primal forward's pi_iters truncation.
            # Fused into ONE launch: the n_it-step in-place self-loop runs register-resident
            # (primal weights/r/constants are loop-invariant -> loaded once), collapsing n_it
            # launches -> 1 and the invariant global traffic ~n_it x. Numerically identical to
            # looping `step` n_it times in-place (last step writes dpibar).
            compute_wave_step_tangent_selfloop(
                pi, dpi, ws, W, S, max(int(self_iters), 1),
                base["mc"], dcst["dMC"], base["dl"], dcst["dDL"], base["ebar"], dcst["dEbar"],
                base["e"], dcst["dE"], base["sl1"], dcst["dSL1"], base["sl2"], dcst["dSL2"],
                sv["col_log_probs"], c1, c2, parent, mad, dts_r, d_dts,
                leaf_state_idx=leaf_state_idx, leaf_logp=base["leaf"], dleaf_logp=dcst["dleaf"],
                item_idx=item_idx, dPibar_out=dpibar, has_leaf_term=has_leaf, use_col_weights=False,
            )
        elif self_iters is not None:
            # reference (unfused) fixed-count path: one launch per Jacobi step
            n_it = max(int(self_iters), 1)
            for _ in range(n_it - 1):
                step(dpi, dts_r, d_dts, ws, W, has_leaf, store=False)  # in-place Jacobi
            step(dpi, dts_r, d_dts, ws, W, has_leaf, store=True)  # last step writes dpibar
        else:
            prev = dpi.narrow(0, ws, W).clone()
            for _ in range(int(self_max_iter)):
                step(dpi, dts_r, d_dts, ws, W, has_leaf, store=False)  # in-place Jacobi on dpi[ws:ws+W]
                cur = dpi.narrow(0, ws, W)
                diff = float((cur - prev).abs().max())
                scale = float(cur.abs().max())
                if diff <= self_tol * max(1.0, scale):
                    break
                prev = cur.clone()
            step(dpi, dts_r, d_dts, ws, W, has_leaf, store=True)  # write converged dpibar[ws:ws+W]

    roots = dpi.index_select(0, wl["root_row_ids"])
    if return_full:
        return roots, dict(dPi=dpi, dPibar=dpibar, d_dts=d_dts_by_ws, dcst=dcst, **raw)
    return roots
