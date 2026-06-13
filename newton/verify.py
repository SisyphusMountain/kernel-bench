"""Verification routines for the tangent kernels and the GGN operator.

The forward solve is bit-exact deterministic and the converged states are O(1-50), so a central
finite difference of the *forward* (not of the noisy scalar loss) is a trustworthy oracle for the
analytical tangents. Checks run in fp64 by default (the kernels honor ``DTYPE`` from the input).
"""

from __future__ import annotations

import math

import torch

from newton.vg import load_problem
from kbench.runtime import move_to_device
from kbench.core.parameters.extract_parameters import as_item_param, as_item_state, extract_parameters_uniform
from kbench.core.inference.solver import solve_e_pi
from kbench.core.kernels.dts_fused import compute_dts_forward
from kbench.core.kernels.e_step import e_fixed_point_triton
from kbench.core.kernels.e_step_tangent import e_tangent_fixed_point
from kbench.core.kernels.wave_step import compute_wave_step
from kbench.core.kernels.wave_tangent import compute_wave_step_tangent
from kbench.core.kernels.dts_tangent import compute_dts_tangent


def _base_params_uniform(static, theta):
    """Replicate solve_e_pi's parameter extraction for the use_col_weights=False path."""
    S = int(static.state_helpers["S"])
    unnorm_row_max = static.state_helpers["unnorm_row_max"].to(device=theta.device, dtype=theta.dtype)
    log_pS, log_pD, log_pL, max_coupling = extract_parameters_uniform(
        theta, unnorm_row_max, statewise=static.statewise, itemwise=static.itemwise
    )
    col_log_probs = theta.new_full((S,), -math.log2(S))
    return log_pS, log_pD, log_pL, max_coupling, col_log_probs


def _solve_E(static, params, col_log_probs, e_init, e_max_iter, e_tol):
    sh = static.state_helpers
    S = int(sh["S"])
    item_rows = int(static.wave_layout["root_row_ids"].numel()) if static.itemwise else 1
    E0 = params[0].new_full((item_rows, S), float(e_init))
    E, E_s1, E_s2, Ebar = e_fixed_point_triton(
        E0, *params, col_log_probs,
        sh["node_parent"], sh["node_child1"], sh["node_child2"], int(sh["max_ancestor_depth"]),
        max_iter=int(e_max_iter), tol=float(e_tol), use_col_weights=False,
    )
    return E, E_s1, E_s2, Ebar


def check_e_tangent(label: str = "small", eps: float = 1e-6, seed: int = 0) -> bool:
    """FD-check dE*(dp) from e_tangent_fixed_point vs central diff of the converged E*."""
    cap, static, theta, col_weights = load_problem(label)
    so = static.solver_options
    theta = theta.double()
    sh = static.state_helpers

    log_pS, log_pD, log_pL, max_coupling, col_log_probs = _base_params_uniform(static, theta)
    params = (log_pS, log_pD, log_pL, max_coupling)

    E_star, _, _, _ = _solve_E(static, params, col_log_probs, so.e_init, so.e_max_iter, 1e-12)

    gen = torch.Generator(device=theta.device).manual_seed(seed)
    dparams = tuple(torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params)

    dE_star, _, _, _ = e_tangent_fixed_point(
        E_star, *dparams, *params, col_log_probs,
        sh["node_parent"], sh["node_child1"], sh["node_child2"], int(sh["max_ancestor_depth"]),
        max_iter=int(so.e_max_iter), tol=1e-13, use_col_weights=False,
    )

    pp = tuple(p + eps * dp for p, dp in zip(params, dparams))
    pm = tuple(p - eps * dp for p, dp in zip(params, dparams))
    Ep, _, _, _ = _solve_E(static, pp, col_log_probs, so.e_init, so.e_max_iter, 1e-12)
    Em, _, _, _ = _solve_E(static, pm, col_log_probs, so.e_init, so.e_max_iter, 1e-12)
    dE_fd = (Ep - Em) / (2 * eps)

    finite = torch.isfinite(dE_star) & torch.isfinite(dE_fd)
    a, b = dE_star[finite], dE_fd[finite]
    abs_err = float((a - b).abs().max())
    rel_err = abs_err / max(float(b.abs().max()), 1e-30)
    ok = rel_err <= 1e-4 or abs_err <= 1e-6
    print(f"[e_tangent {label}] eps={eps:.0e}  ||dE*||inf={float(a.abs().max()):.4e}")
    print(f"  vs FD: max_abs={abs_err:.3e}  max_rel={rel_err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _solve_full_fp64(static, theta, col_weights):
    out = solve_e_pi(static, theta.double(), col_weights.double())
    keys = ("E", "E_s1", "E_s2", "Ebar", "root_rows", "pi_wave", "pibar_wave",
            "pibar_row_max", "log_pS", "log_pD", "log_pL", "max_coupling", "col_log_probs")
    return dict(zip(keys, out))


def _wave_constants(sv, S):
    """Reconstruct the per-item wave-step constants exactly as pi_wave_forward does."""
    item_rows = int(sv["E"].shape[0])
    e_item = as_item_state(sv["E"], S, item_rows)
    ebar_item = as_item_state(sv["Ebar"], S, item_rows)
    e_s1_item = as_item_state(sv["E_s1"], S, item_rows)
    e_s2_item = as_item_state(sv["E_s2"], S, item_rows)
    mc_item = as_item_state(sv["max_coupling"].squeeze(-1), S, item_rows)
    pd_item = as_item_state(sv["log_pD"], S, item_rows)
    ps_item = as_item_state(sv["log_pS"], S, item_rows)
    return {
        "dl": 1.0 + pd_item + e_item,
        "ebar": ebar_item,
        "e": e_item,
        "sl1": ps_item + e_s2_item,
        "sl2": ps_item + e_s1_item,
        "mc": mc_item,
        "leaf": ps_item,
        "pd_param": as_item_param(sv["log_pD"], item_rows, S),
        "ps_param": as_item_param(sv["log_pS"], item_rows, S),
    }


def _dts_for(meta, sv, c, item_idx):
    if "sl" not in meta:
        return None
    return compute_dts_forward(
        sv["pi_wave"], sv["pibar_wave"], meta["sl"], meta["sr"], c["node_child1"], c["node_child2"],
        int(meta["W"]), meta["reduce_idx"], c["pd_param"], c["ps_param"], item_idx=item_idx,
        log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
        eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
        ge2_parent_ids=meta.get("ge2_parent_ids"), ge2_max_fanout=meta.get("ge2_max_fanout"),
        item_offset=int(meta["start"]),
    )


def check_wave_step_tangent(label: str = "small", eps: float = 1e-6, seed: int = 0, split: bool = True) -> bool:
    """FD-check one compute_wave_step application's tangent (perturb row + all constants)."""
    cap, static, theta, col_weights = load_problem(label)
    sh, wl = static.state_helpers, static.wave_layout
    S = int(sh["S"])
    item_idx = static.rate_item_idx
    topo = dict(node_child1=sh["node_child1"], node_child2=sh["node_child2"],
                node_parent=sh["node_parent"], mad=int(sh["max_ancestor_depth"]))
    sv = _solve_full_fp64(static, theta, col_weights)
    cst = _wave_constants(sv, S)
    cst.update(node_child1=topo["node_child1"], node_child2=topo["node_child2"])

    meta = next(m for m in wl["wave_metas"] if ("sl" in m) == split)
    ws, W = int(meta["start"]), int(meta["W"])
    has_splits = "sl" in meta
    has_leaf = not has_splits
    leaf_state_idx = wl["leaf_state_index"].to(torch.int32)
    dts_r = _dts_for(meta, sv, cst, item_idx)
    C = int(sv["pi_wave"].shape[0])

    def step(pi_in, dl, ebar, e, sl1, sl2, mc, leaf, dts):
        Pi_out = torch.empty_like(pi_in)
        Pibar = torch.empty_like(pi_in)
        prm = torch.empty((C,), dtype=pi_in.dtype, device=pi_in.device)
        compute_wave_step(
            pi_in, Pi_out, Pibar, ws, W, S, mc, dl, ebar, e, sl1, sl2, sv["col_log_probs"],
            topo["node_child1"], topo["node_child2"], topo["node_parent"], topo["mad"],
            dts, leaf_state_idx=leaf_state_idx, leaf_logp=leaf, item_idx=item_idx,
            pibar_row_max=prm, store_final_pibar=False, has_leaf_term=has_leaf,
            input_ws=None, use_col_weights=False,
        )
        return Pi_out.narrow(0, ws, W).clone()

    gen = torch.Generator(device=theta.device).manual_seed(seed)
    rand = lambda t: torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
    dpi = rand(sv["pi_wave"])
    dcst = {k: rand(cst[k]) for k in ("dl", "ebar", "e", "sl1", "sl2", "mc", "leaf")}
    ddts = rand(dts_r) if has_splits else None

    # analytic tangent
    dPi_out = torch.empty_like(sv["pi_wave"])
    compute_wave_step_tangent(
        sv["pi_wave"], dpi, dPi_out, ws, W, S,
        cst["mc"], dcst["mc"], cst["dl"], dcst["dl"], cst["ebar"], dcst["ebar"],
        cst["e"], dcst["e"], cst["sl1"], dcst["sl1"], cst["sl2"], dcst["sl2"],
        sv["col_log_probs"], topo["node_child1"], topo["node_child2"], topo["node_parent"], topo["mad"],
        dts_r, ddts, leaf_state_idx=leaf_state_idx, leaf_logp=cst["leaf"], dleaf_logp=dcst["leaf"],
        item_idx=item_idx, dPibar_out=None, has_leaf_term=has_leaf, input_ws=None, use_col_weights=False,
    )
    dPi_a = dPi_out.narrow(0, ws, W)

    def pert(scale):
        pi = sv["pi_wave"] + scale * dpi
        args = {k: cst[k] + scale * dcst[k] for k in ("dl", "ebar", "e", "sl1", "sl2", "mc", "leaf")}
        dts = (dts_r + scale * ddts) if has_splits else None
        return step(pi, args["dl"], args["ebar"], args["e"], args["sl1"], args["sl2"], args["mc"], args["leaf"], dts)

    dPi_fd = (pert(eps) - pert(-eps)) / (2 * eps)

    fin = torch.isfinite(dPi_a) & torch.isfinite(dPi_fd)
    a, b = dPi_a[fin], dPi_fd[fin]
    abs_err = float((a - b).abs().max())
    rel_err = abs_err / max(float(b.abs().max()), 1e-30)
    ok = rel_err <= 1e-4 or abs_err <= 1e-6
    kind = "split" if split else "leaf"
    print(f"[wave_step_tangent {label} {kind}] wave ws={ws} W={W}  ||dPi||inf={float(a.abs().max()):.4e}")
    print(f"  vs FD: max_abs={abs_err:.3e}  max_rel={rel_err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_forward_tangent(label: str = "small", eps: float = 1e-6, seed: int = 0) -> bool:
    """FD-check jvp_root_scores(v) vs central diff of the forward root scores."""
    from newton.forward_tangent import jvp_root_scores

    cap, static, theta, col_weights = load_problem(label)
    theta = theta.double()
    col_weights = col_weights.double()
    S = int(static.state_helpers["S"])
    sv = _solve_full_fp64(static, theta, col_weights)

    gen = torch.Generator(device=theta.device).manual_seed(seed)
    v = torch.randn(theta.shape, generator=gen, device=theta.device, dtype=theta.dtype)

    t_a = jvp_root_scores(static, theta, v, sv)

    def root(th):
        return solve_e_pi(static, th, col_weights)[4]  # root_rows = pi[root_row_ids]

    t_fd = (root(theta + eps * v) - root(theta - eps * v)) / (2 * eps)

    fin = torch.isfinite(t_a) & torch.isfinite(t_fd)
    a, b = t_a[fin], t_fd[fin]
    abs_err = float((a - b).abs().max())
    rel_err = abs_err / max(float(b.abs().max()), 1e-30)
    ok = rel_err <= 1e-3 or abs_err <= 1e-5
    print(f"[forward_tangent {label}] root {tuple(t_a.shape)}  ||J v||inf={float(a.abs().max()):.4e}")
    print(f"  vs FD: max_abs={abs_err:.3e}  max_rel={rel_err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_dts_tangent(label: str = "small", eps: float = 1e-6, seed: int = 0) -> bool:
    """FD-check compute_dts_tangent vs central diff of compute_dts_forward."""
    cap, static, theta, col_weights = load_problem(label)
    sh, wl = static.state_helpers, static.wave_layout
    S = int(sh["S"])
    item_idx = static.rate_item_idx
    sv = _solve_full_fp64(static, theta, col_weights)
    cst = _wave_constants(sv, S)
    c1, c2 = sh["node_child1"], sh["node_child2"]

    meta = next(m for m in wl["wave_metas"] if "sl" in m)
    ws, W = int(meta["start"]), int(meta["W"])
    pd, ps = cst["pd_param"], cst["ps_param"]

    def dts(pi, pibar, pd_, ps_):
        return compute_dts_forward(
            pi, pibar, meta["sl"], meta["sr"], c1, c2, W, meta["reduce_idx"], pd_, ps_,
            item_idx=item_idx, log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
            eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
            ge2_parent_ids=meta.get("ge2_parent_ids"), ge2_max_fanout=meta.get("ge2_max_fanout"),
            item_offset=ws,
        )

    dts_r = dts(sv["pi_wave"], sv["pibar_wave"], pd, ps)
    gen = torch.Generator(device=theta.device).manual_seed(seed)
    rand = lambda t: torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
    dpi, dpibar, dpd, dps = rand(sv["pi_wave"]), rand(sv["pibar_wave"]), rand(pd), rand(ps)

    d_a = compute_dts_tangent(
        sv["pi_wave"], sv["pibar_wave"], dpi, dpibar, meta["sl"], meta["sr"], c1, c2, W,
        meta["reduce_idx"], pd, ps, dpd, dps, dts_r, item_idx,
        log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
        eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
        ge2_parent_ids=meta.get("ge2_parent_ids"), ge2_max_fanout=meta.get("ge2_max_fanout"),
        item_offset=ws,
    )
    dp = lambda sc: dts(sv["pi_wave"] + sc * dpi, sv["pibar_wave"] + sc * dpibar, pd + sc * dpd, ps + sc * dps)
    d_fd = (dp(eps) - dp(-eps)) / (2 * eps)

    fin = torch.isfinite(d_a) & torch.isfinite(d_fd)
    a, b = d_a[fin], d_fd[fin]
    abs_err = float((a - b).abs().max())
    rel_err = abs_err / max(float(b.abs().max()), 1e-30)
    ok = rel_err <= 1e-4 or abs_err <= 1e-6
    print(f"[dts_tangent {label}] wave ws={ws} W={W}  ||d_dts||inf={float(a.abs().max()):.4e}")
    print(f"  vs FD: max_abs={abs_err:.3e}  max_rel={rel_err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_e_so(label: str = "small", eps: float = 1e-6, seed: int = 0) -> bool:
    """Local FD gate for e_step_backward_so: perturb the primal inputs of the FROZEN e-step
    backward kernels (cotangents fixed) and difference its outputs against the analytic
    second-order contraction."""
    import triton
    from kbench.core.kernels.e_step import (
        _e_step_backward_prepare_2d_kernel, _e_step_backward_finalize_2d_kernel,
        _launch_e_step_forward_2d, _tl_float_dtype,
    )
    from kbench.core.kernels.e_step_so import e_step_backward_so

    cap, static, theta, col_weights = load_problem(label)
    sh = static.state_helpers
    S = int(sh["S"])
    mad = int(sh["max_ancestor_depth"])
    parent, c1, c2 = sh["node_parent"], sh["node_child1"], sh["node_child2"]

    theta64 = theta.double()
    log_pS, log_pD, log_pL, mc, col = _base_params_uniform(static, theta64)
    so = static.solver_options
    E_star, _, _, _ = _solve_E(static, (log_pS, log_pD, log_pL, mc), col, so.e_init, so.e_max_iter, 1e-12)
    G = int(E_star.shape[0])
    item = lambda t: as_item_state(t, S, G)
    pS_m, pD_m, pL_m = item(log_pS), item(log_pD), item(log_pL)

    def fwd(E):
        return _launch_e_step_forward_2d(
            E.contiguous(), pS_m, pD_m, pL_m, item(mc), col.contiguous(),
            parent, c1, c2, mad, use_col_weights=False,
        )

    E_new0, E_s10, E_s20, Ebar0 = fwd(E_star)

    def bwd(E, E_new, E_s1, E_s2, Ebar, pS, pD, pL, g):
        g_new, g_s1, g_s2, g_ebar = g
        block_s = int(triton.next_power_of_2(S))
        grad_E, grad_pS, grad_pD, grad_pL, grad_mc, r, excl = (torch.empty_like(E) for _ in range(7))
        grad_col = torch.zeros_like(col)
        tot = torch.empty((G,), dtype=E.dtype, device=E.device)
        _e_step_backward_prepare_2d_kernel[(G,)](
            E.contiguous(), E_new.contiguous(), E_s1.contiguous(), E_s2.contiguous(),
            Ebar.contiguous(), pS.contiguous(), pD.contiguous(), pL.contiguous(),
            col.contiguous(), parent, c1, c2,
            g_new, g_s1, g_s2, g_ebar,
            grad_E, grad_pS, grad_pD, grad_pL, grad_mc, grad_col,
            r, excl, tot, S, BLOCK_S=block_s, MAX_ANCESTOR_DEPTH=mad,
            USE_COL_WEIGHTS=False, DTYPE=_tl_float_dtype(E.dtype), num_warps=8,
        )
        _e_step_backward_finalize_2d_kernel[(G,)](
            grad_E, grad_col, r, excl, tot, S, BLOCK_S=block_s, num_warps=8,
        )
        return grad_E, grad_pS, grad_pD, grad_pL, grad_mc, grad_col

    gen = torch.Generator(device=theta.device).manual_seed(seed)
    rand = lambda t: torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
    g = tuple(rand(E_star) for _ in range(4))
    dE, dE_new, dE_s1, dE_s2, dEbar = (rand(E_star) for _ in range(5))
    dpS, dpD, dpL = rand(pS_m), rand(pD_m), rand(pL_m)

    d_a = e_step_backward_so(
        E_star.contiguous(), E_new0, E_s10, E_s20, Ebar0, pS_m, pD_m, pL_m, col.contiguous(),
        parent, c1, c2, mad, *g,
        dE, dE_new, dE_s1, dE_s2, dEbar, dpS, dpD, dpL, None, use_col_weights=False,
    )

    def pert(s):
        return bwd(E_star + s * dE, E_new0 + s * dE_new, E_s10 + s * dE_s1, E_s20 + s * dE_s2,
                   Ebar0 + s * dEbar, pS_m + s * dpS, pD_m + s * dpD, pL_m + s * dpL, g)

    plus, minus = pert(eps), pert(-eps)
    names = ("d_grad_E", "d_grad_pS", "d_grad_pD", "d_grad_pL", "d_grad_mc", "d_grad_col")
    ok = True
    print(f"[e_so {label}] fixed cotangents, independent primal tangents, eps={eps:.0e}")
    for name, a, hi, lo in zip(names, d_a, plus, minus):
        fd = (hi - lo) / (2 * eps)
        fin = torch.isfinite(a) & torch.isfinite(fd)
        abs_err = float((a[fin] - fd[fin]).abs().max())
        scale = max(float(fd[fin].abs().max()), 1e-30)
        rel = abs_err / scale
        good = rel <= 1e-6 or abs_err <= 1e-9
        ok &= good
        print(f"  {name:11s} max_abs={abs_err:.3e} max_rel={rel:.3e} {'PASS' if good else 'FAIL'}")
    return ok


def check_wave_so(label: str = "small", eps: float = 1e-6, seed: int = 0, split: bool = True) -> bool:
    """Local FD gates for wave_backward_so against the FROZEN wave backward:
    (A) neumann_terms=0 exposes B^T v -> FD vs d_aw buckets;
    (B) neumann_terms=1 exposes v + A^T v -> FD vs d(A^T v)."""
    from kbench.core.kernels.wave_backward import (
        active_mask_from_rhs_absmax_fused, wave_backward_uniform_fused,
    )
    from kbench.core.kernels.wave_so import wave_backward_so
    from newton.forward_tangent import wave_step_constants

    cap, static, theta, col_weights = load_problem(label)
    sh, wl = static.state_helpers, static.wave_layout
    S = int(sh["S"])
    item_idx = static.rate_item_idx
    mad = int(sh["max_ancestor_depth"])
    c1, c2, parent = sh["node_child1"], sh["node_child2"], sh["node_parent"]
    leaf_state_idx = wl["leaf_state_index"].to(torch.int32)
    sv = _solve_full_fp64(static, theta, col_weights)
    cst = wave_step_constants(sv, S)
    prm = sv["pibar_row_max"]

    meta = next(m for m in wl["wave_metas"] if ("sl" in m) == split)
    ws, W = int(meta["start"]), int(meta["W"])
    has_splits = "sl" in meta
    has_leaf = not has_splits
    dts_r = _dts_for(meta, sv, dict(cst, node_child1=c1, node_child2=c2), item_idx) if has_splits else None

    gen = torch.Generator(device=theta.device).manual_seed(seed)
    rand = lambda t: torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
    vfix = rand(torch.empty((W, S), device=theta.device, dtype=sv["pi_wave"].dtype))
    dpi, dpibar = rand(sv["pi_wave"]), rand(sv["pibar_wave"])
    dc = {k: rand(cst[k]) for k in ("dl", "ebar", "e", "sl1", "sl2", "leaf")}
    ddts = rand(dts_r) if has_splits else None
    active = active_mask_from_rhs_absmax_fused(vfix, 0.0, use_pruning=False).contiguous()

    def run(nt, pi, pibar, dl, ebar, e, sl1, sl2, leaf, dts):
        return wave_backward_uniform_fused(
            pi, pibar, ws, W, S, dts, vfix, cst["mc"], dl, ebar, e, sl1, sl2,
            sv["col_log_probs"], c1, c2, None, neumann_terms=nt,
            leaf_state_idx=leaf_state_idx, leaf_logp=leaf, has_leaf_term=has_leaf,
            active_mask=active, node_parent=parent, max_ancestor_depth=mad,
            pibar_row_max=prm, item_idx=item_idx, item_indexed_consts=True,
            compact_level_ptr=sh["compact_level_ptr"], compact_level_parents=sh["compact_level_parents"],
            compact_level_child1=sh["compact_level_child1"], compact_level_child2=sh["compact_level_child2"],
            grad_col_log_probs=torch.zeros((S,), device=theta.device, dtype=sv["pi_wave"].dtype),
            use_col_weights=False, self_loop_solver="neumann", return_last_increment=False,
        )

    def pert(s, nt):
        return run(nt, sv["pi_wave"] + s * dpi, sv["pibar_wave"] + s * dpibar,
                   cst["dl"] + s * dc["dl"], cst["ebar"] + s * dc["ebar"], cst["e"] + s * dc["e"],
                   cst["sl1"] + s * dc["sl1"], cst["sl2"] + s * dc["sl2"],
                   cst["leaf"] + s * dc["leaf"], (dts_r + s * ddts) if has_splits else None)

    d_an = wave_backward_so(
        sv["pi_wave"], dpi, sv["pibar_wave"], dpibar, vfix, ws, W, S,
        prm, cst["mc"], cst["dl"], dc["dl"], cst["ebar"], dc["ebar"], cst["e"], dc["e"],
        cst["sl1"], dc["sl1"], cst["sl2"], dc["sl2"],
        sv["col_log_probs"], c1, c2, parent, mad, dts_r, ddts,
        leaf_state_idx=leaf_state_idx, leaf_logp=cst["leaf"], dleaf_logp=dc["leaf"],
        item_idx=item_idx, has_leaf_term=has_leaf, use_col_weights=False,
    )

    kind = "split" if split else "leaf"
    ok = True
    print(f"[wave_so {label} {kind}] wave ws={ws} W={W}")
    # gate B: d(A^T v) via neumann_terms=1 (v_k output = v + A^T v; dv part of seed = 0)
    vp, vm = pert(eps, 1)[0], pert(-eps, 1)[0]
    fd = (vp - vm) / (2 * eps)
    fin = torch.isfinite(d_an[0]) & torch.isfinite(fd)
    abs_err = float((d_an[0][fin] - fd[fin]).abs().max())
    rel = abs_err / max(float(fd[fin].abs().max()), 1e-30)
    good = rel <= 1e-6 or abs_err <= 1e-9
    ok &= good
    print(f"  d(A^T v)  max_abs={abs_err:.3e} max_rel={rel:.3e} {'PASS' if good else 'FAIL'}")
    # gate A: d(B^T v) buckets via neumann_terms=0 (aw outputs at v_k = rhs = v)
    outs_p, outs_m = pert(eps, 0), pert(-eps, 0)
    names = ("d_aw0", "d_aw1", "d_aw2", "d_aw345", "d_aw3", "d_aw4")
    for i, name in enumerate(names):
        fd = (outs_p[1 + i] - outs_m[1 + i]) / (2 * eps)
        a = d_an[1 + i]
        fin = torch.isfinite(a) & torch.isfinite(fd)
        abs_err = float((a[fin] - fd[fin]).abs().max())
        rel = abs_err / max(float(fd[fin].abs().max()), 1e-30)
        good = rel <= 1e-6 or abs_err <= 1e-9
        ok &= good
        print(f"  {name:8s}  max_abs={abs_err:.3e} max_rel={rel:.3e} {'PASS' if good else 'FAIL'}")
    return ok


def check_dts_so(label: str = "small", eps: float = 1e-6, seed: int = 0, wave_i: int = 0) -> bool:
    """Composed local FD gate for dts_backward_so against the FROZEN dts + pibar-tree kernels."""
    from kbench.core.kernels.wave_backward import (
        active_mask_from_rhs_absmax_fused, dts_cross_backward_accum_fused,
        uniform_cross_pibar_vjp_tree_from_ud_fused,
    )
    from kbench.core.kernels.dts_so import dts_backward_so
    from newton.forward_tangent import wave_step_constants

    cap, static, theta, col_weights = load_problem(label)
    sh, wl = static.state_helpers, static.wave_layout
    S = int(sh["S"])
    item_idx = static.rate_item_idx
    c1, c2, parent = sh["node_child1"], sh["node_child2"], sh["node_parent"]
    sv = _solve_full_fp64(static, theta, col_weights)
    cst = wave_step_constants(sv, S)
    prm = sv["pibar_row_max"]
    col = sv["col_log_probs"]
    C = int(sv["pi_wave"].shape[0])

    metas = [m for m in wl["wave_metas"] if "sl" in m]
    meta = metas[wave_i]
    ws, W = int(meta["start"]), int(meta["W"])
    sl, sr = meta["sl"], meta["sr"]
    N = int(sl.numel())
    lsp = meta.get("log_split_probs")
    if lsp is None:
        lsp = torch.zeros((N,), device=theta.device, dtype=sv["pi_wave"].dtype)

    gen = torch.Generator(device=theta.device).manual_seed(seed)
    rand = lambda t: torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
    vfix = rand(torch.empty((W, S), device=theta.device, dtype=sv["pi_wave"].dtype))
    active = active_mask_from_rhs_absmax_fused(vfix, 0.0, use_pruning=False).contiguous()
    dpi, dpibar = rand(sv["pi_wave"]), rand(sv["pibar_wave"])
    dpd, dps, dmc = rand(cst["pd_param"]), rand(cst["ps_param"]), rand(cst["mc"])

    def oracle(pi, pibar, pd, ps, mc_i):
        rhs = torch.zeros((C, S), device=theta.device, dtype=pi.dtype)
        gpD = torch.zeros_like(pd)
        gpS = torch.zeros_like(ps)
        gmt = torch.zeros_like(mc_i)
        gcol = torch.zeros((S,), device=theta.device, dtype=pi.dtype)
        gl, gr, side_act, _p1, _p2 = dts_cross_backward_accum_fused(
            pi, pibar, vfix, ws, sl, sr, meta["reduce_idx"], lsp, pd, ps, c1, c2, rhs, S,
            active_mask=active, merge_s_term=True, grad_log_pD=gpD, grad_log_pS=gpS,
            grad_mt=gmt, accum_param_reductions=True, accum_mt_reduction=True,
            output_pibar_ud=True, output_pibar_side_active=True, pibar_side_threshold=0.0,
            mt_squeezed=mc_i, pibar_row_max=prm,
            grad_mt_two_stage=bool(gmt.ndim == 2 and int(gmt.shape[0]) == 1),
            grad_mt_two_stage_tile_splits=128, skip_inactive_pibar_output_zero=True,
            item_idx=item_idx,
        )
        uniform_cross_pibar_vjp_tree_from_ud_fused(
            pi, col, gl, gr, sl, sr, rhs, S, active_mask=active, reduce_idx=meta["reduce_idx"],
            pibar_row_max=prm, skip_zero_sides=True, side_active=side_act,
            compact_level_ptr=sh["compact_level_ptr"], compact_level_parents=sh["compact_level_parents"],
            compact_level_child1=sh["compact_level_child1"], compact_level_child2=sh["compact_level_child2"],
            grad_col_log_probs=gcol, use_col_weights=False, side_active_threshold=0.0,
        )
        return rhs, gpD, gpS, gmt, gcol

    def pert(s):
        return oracle(sv["pi_wave"] + s * dpi, sv["pibar_wave"] + s * dpibar,
                      cst["pd_param"] + s * dpd, cst["ps_param"] + s * dps, cst["mc"] + s * dmc)

    d_rhs = torch.zeros((C, S), device=theta.device, dtype=sv["pi_wave"].dtype)
    d_gpD = torch.zeros_like(cst["pd_param"])
    d_gpS = torch.zeros_like(cst["ps_param"])
    d_gmt = torch.zeros_like(cst["mc"])
    d_gcol = torch.zeros((S,), device=theta.device, dtype=sv["pi_wave"].dtype)
    dts_backward_so(
        sv["pi_wave"], dpi, sv["pibar_wave"], dpibar, vfix, ws, meta, S,
        cst["pd_param"], cst["ps_param"], dpd, dps, cst["mc"], dmc,
        col, c1, c2, parent, int(sh["max_ancestor_depth"]), prm, item_idx,
        d_rhs, d_gpD, d_gpS, d_gmt, d_gcol,
        compact_level_ptr=sh["compact_level_ptr"],
        compact_level_parents=sh["compact_level_parents"],
        compact_level_child1=sh["compact_level_child1"],
        compact_level_child2=sh["compact_level_child2"],
        use_col_weights=False,
    )

    plus, minus = pert(eps), pert(-eps)
    names = ("d_rhs", "d_grad_pD", "d_grad_pS", "d_grad_mt", "d_grad_col")
    analytic = (d_rhs, d_gpD, d_gpS, d_gmt, d_gcol)
    ok = True
    ge2 = meta.get("ge2_parent_ids")
    print(f"[dts_so {label}] wave ws={ws} N={N} ge2={'yes' if ge2 is not None and ge2.numel() else 'no'}")
    for name, a, hi, lo in zip(names, analytic, plus, minus):
        fd = (hi - lo) / (2 * eps)
        fin = torch.isfinite(a) & torch.isfinite(fd)
        abs_err = float((a[fin] - fd[fin]).abs().max())
        rel = abs_err / max(float(fd[fin].abs().max()), 1e-30)
        good = rel <= 1e-6 or abs_err <= 1e-9
        ok &= good
        print(f"  {name:10s}  max_abs={abs_err:.3e} max_rel={rel:.3e} {'PASS' if good else 'FAIL'}")
    return ok


def check_hvp_exact(label: str = "small", seed: int = 0) -> bool:
    """Composition gate: fp64 analytic exact HVP vs fp64 FD-of-gradient, plus symmetry."""
    from newton.vg import make_value_and_grad
    from newton.newton_cg import _fd_hessian_hvp
    from newton.hvp_exact import build_point_cache, make_exact_hvp

    cap, static, theta, col_weights = load_problem(label)
    t64, cw64 = theta.double(), col_weights.double()
    sv = _solve_full_fp64(static, t64, cw64)
    gt, gc, cache = build_point_cache(static, t64, cw64, sv)
    hvp = make_exact_hvp(static, t64, cw64, sv, cache=cache)

    vg = make_value_and_grad(static, cw64)
    x = t64.reshape(-1).contiguous()
    out = vg(x)
    warm = out[3]
    del out
    fd = _fd_hessian_hvp(vg, x, warm)

    gen = torch.Generator(device=theta.device).manual_seed(seed)
    p = x.numel()
    ok = True
    Hs = []
    for i in range(2):
        u = torch.randn(p, generator=gen, device=theta.device, dtype=torch.float64)
        u /= u.norm()
        Ha = hvp(u).double()
        Hf = fd(u).double()
        Hs.append((u, Ha))
        abs_err = float((Ha - Hf).abs().max())
        rel = abs_err / max(float(Hf.abs().max()), 1e-30)
        # the production gradient comes from TRUNCATED solvers (neumann=16, bicgstab 1e-7):
        # d[truncated solve] != truncated[d solve] at ~1e-4 of scale. Component kernels and the
        # composed accumulators are separately FD-verified at 1e-9 / 1e-6 (see e_so/wave_so/
        # dts_so gates and /tmp debug); this end-to-end tolerance reflects solver truncation.
        good = rel <= 5e-4
        ok &= good
        print(f"[hvp_exact {label}] dir {i}: |Ha|={float(Ha.norm()):.4f} |Hfd|={float(Hf.norm()):.4f} "
              f"max_abs={abs_err:.3e} max_rel={rel:.3e} {'PASS' if good else 'FAIL'}")
    (u, Hu), (w, Hw) = Hs
    uHw, wHu = float(torch.dot(u, Hw)), float(torch.dot(w, Hu))
    scale = max((abs(float(torch.dot(u, Hu))) * abs(float(torch.dot(w, Hw)))) ** 0.5, 1e-30)
    sym = abs(uHw - wHu) / scale
    good = sym <= 5e-3  # truncated-solver gradient is not an exact gradient field (see above)
    ok &= good
    print(f"[hvp_exact {label}] symmetry: uHw={uHw:.6e} wHu={wHu:.6e} rel_asym={sym:.3e} "
          f"{'PASS' if good else 'FAIL'}")
    return ok


def check_ggn(label: str = "small", seed: int = 0) -> bool:
    """Regress the replicated VJP against golden, then check M symmetry + PSD (fp64)."""
    from newton.ggn import vjp_root_to_theta, make_ggn_hvp

    cap, static, theta, col_weights = load_problem(label)
    ok = True

    # (1) regression: replicated VJP (seed=-q, norm kept) == golden grad_theta
    sv32 = {k: move_to_device(v, theta.device) for k, v in cap["forward_saved"].items()}
    gt, _ = vjp_root_to_theta(static, sv32, None, theta, col_weights, drop_norm=False)
    gold = cap["golden"]["grad_theta"].to(theta.device).float()
    abs_err = float((gt.float() - gold).abs().max())
    rel_err = abs_err / max(float(gold.abs().max()), 1e-30)
    reg_ok = abs_err <= 2e-3 or rel_err <= 2e-3
    print(f"[ggn vjp-regression {label}] vs golden grad_theta: max_abs={abs_err:.3e} max_rel={rel_err:.3e} "
          f"{'PASS' if reg_ok else 'FAIL'}")
    ok &= reg_ok

    # (2,3) symmetry + PSD in fp64
    theta64, cw64 = theta.double(), col_weights.double()
    sv = _solve_full_fp64(static, theta64, cw64)
    hvp = make_ggn_hvp(static, theta64, cw64, sv)
    p = 3 * int(static.state_helpers["S"])
    gen = torch.Generator(device=theta.device).manual_seed(seed)
    v = torch.randn(p, generator=gen, device=theta.device, dtype=torch.float64)
    w = torch.randn(p, generator=gen, device=theta.device, dtype=torch.float64)
    Mv, Mw = hvp(v).double(), hvp(w).double()
    vMw, wMv = float(torch.dot(v, Mw)), float(torch.dot(w, Mv))
    vMv, wMw = float(torch.dot(v, Mv)), float(torch.dot(w, Mw))
    sym = abs(vMw - wMv) / max((vMv * wMw) ** 0.5, 1e-30)  # relative to curvature scale
    sym_ok = sym <= 5e-3  # small vs curvature; immaterial for damped Newton-CG
    psd_ok = vMv >= -1e-9 * max(abs(vMv), 1.0) and wMw >= -1e-9 * max(abs(wMw), 1.0)
    print(f"[ggn symmetry {label}] <v,Mw>={vMw:.6e} <w,Mv>={wMv:.6e} rel_asym(scale)={sym:.3e} "
          f"{'PASS' if sym_ok else 'FAIL'}")
    print(f"[ggn PSD {label}] vMv={vMv:.6e} wMw={wMw:.6e} {'PASS' if psd_ok else 'FAIL'}")
    ok &= sym_ok and psd_ok
    return ok


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "e_tangent"
    label = sys.argv[2] if len(sys.argv) > 2 else "small"
    if which == "ggn":
        ok = check_ggn(label)
    elif which == "e_so":
        ok = check_e_so(label)
    elif which == "wave_so":
        ok = check_wave_so(label, split=True) and check_wave_so(label, split=False)
    elif which == "dts_so":
        ok = check_dts_so(label, wave_i=0) and check_dts_so(label, wave_i=1)
    elif which == "hvp":
        ok = check_hvp_exact(label)
    elif which == "wave_step":
        ok = check_wave_step_tangent(label, split=True) and check_wave_step_tangent(label, split=False)
    elif which == "dts":
        ok = check_dts_tangent(label)
    elif which == "forward":
        ok = check_forward_tangent(label)
    elif which == "all":
        ok = (check_e_tangent(label) and check_wave_step_tangent(label, split=True)
              and check_wave_step_tangent(label, split=False) and check_dts_tangent(label)
              and check_forward_tangent(label))
    else:
        ok = {"e_tangent": check_e_tangent}[which](label)
    raise SystemExit(0 if ok else 1)
