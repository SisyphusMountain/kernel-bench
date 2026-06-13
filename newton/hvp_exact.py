"""Analytic exact-Hessian HVP (forward-over-reverse) — orchestrator.

Per outer Newton point: ``build_point_cache`` runs the production backward ONCE (the verified
``vjp_root_to_theta`` loop) while caching per-wave adjoints ``v_k``/``dts_r`` and the E-side
adjoint ``wE`` — theta is fixed across all CG iterations, so the cache amortizes. Each
``hvp(u)`` then costs one tangent-forward sweep + one tangent-adjoint sweep (same solve
operators, modified seeds) + the second-order contraction kernels (e_step_so / wave_so / dts_so).

Status: point-cache + gradient reproduction (build step 2). The tangent-adjoint sweep
(steps 3-5) composes on top of this cache.
"""

from __future__ import annotations

import torch

from kbench.api._implicit_grad import _bicgstab, _safe_exp2_ratio
from kbench.core.inference.logspace import logsumexp2 as _logsumexp2
from kbench.core.kernels.dts_so import dts_backward_so
from kbench.core.kernels.e_step import e_step_triton_autograd
from kbench.core.kernels.e_step_so import e_step_backward_so
from kbench.core.kernels.wave_backward import (
    dts_cross_backward_accum_fused, uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
)
from kbench.core.kernels.wave_so import wave_backward_so
from kbench.core.parameters.extract_parameters import (
    as_item_param, as_item_state, extract_parameters_weighted_cols,
)
from newton.forward_tangent import jvp_root_scores, wave_step_constants
from newton.ggn import vjp_root_to_theta

_LN2 = 0.6931471805599453


@torch.no_grad()
def build_point_cache(static, theta, col_weights, sv):
    """Run the production-configured backward once, caching per-wave (v_k, dts_r, active_mask)
    and the E-side adjoint. Returns (grad_theta, grad_col, cache)."""
    cache: dict = {}
    grad_theta, grad_col = vjp_root_to_theta(
        static, sv, None, theta, col_weights, drop_norm=False, cache=cache,
    )
    return grad_theta, grad_col, cache


def _scatter_accum(acc, item_rows_for_wave, contrib, item_rows):
    if contrib.dtype != acc.dtype:
        contrib = contrib.to(dtype=acc.dtype)
    if int(item_rows) == 1:
        if acc.ndim == 1:
            acc[0] += contrib.sum()
        elif int(acc.shape[1]) == 1:
            acc[0, 0] += contrib.sum()
        else:
            acc[0] += contrib.sum(dim=0)
        return
    if acc.ndim == 1:
        acc.index_add_(0, item_rows_for_wave, contrib.sum(dim=1))
    elif int(acc.shape[1]) == 1:
        acc[:, 0].index_add_(0, item_rows_for_wave, contrib.sum(dim=1))
    else:
        acc.index_add_(0, item_rows_for_wave, contrib)


def make_exact_hvp(static, theta, col_weights, sv, *, cache=None, debug_out=None,
                   tangent_self_iters=None):
    """Analytic exact-Hessian HVP. Builds the per-point adjoint cache once (if not given) and
    returns ``hvp(u_vec) -> H u`` (flat 3S). Runs in the dtype of ``theta``/``sv``.

    ``tangent_self_iters`` sets the FIXED per-wave self-loop iteration count for the tangent
    forward sweep (sync-free; see ``jvp_root_scores``). Resolution order: this argument, then
    the ``NEWTON_TANGENT_SELF_ITERS`` env var, then ``solver_options.pi_iters`` (matching the
    primal forward truncation). Not hardcoded — change it per run via the env var or the arg.
    """
    import os

    so = static.solver_options
    if tangent_self_iters is None:
        _env = os.environ.get("NEWTON_TANGENT_SELF_ITERS")
        if _env:
            tangent_self_iters = int(_env)
        else:
            tangent_self_iters = int(so.pi_iters) if getattr(so, "pi_iters", None) else 16
    sh, wl = static.state_helpers, static.wave_layout
    S = int(sh["S"])
    item_idx = static.rate_item_idx
    c1, c2, parent = sh["node_child1"], sh["node_child2"], sh["node_parent"]
    mad = int(sh["max_ancestor_depth"])
    leaf_state_idx = wl["leaf_state_index"].to(device=theta.device, dtype=torch.int32).contiguous()
    root_ids = wl["root_row_ids"]
    n_fam = int(root_ids.numel())
    dtype = sv["pi_wave"].dtype
    C = int(sv["pi_wave"].shape[0])
    E_star = sv["E"]
    G = int(E_star.shape[0])

    if cache is None:
        _, _, cache = build_point_cache(static, theta, col_weights, sv)
    acc = cache["accum"]
    wE = cache["e_side"]["wE"]

    cst = wave_step_constants(sv, S)
    prm = sv["pibar_row_max"]
    col = sv["col_log_probs"]
    item = lambda t: as_item_state(t, S, G)
    pS_m, pD_m, pL_m = item(sv["log_pS"]), item(sv["log_pD"]), item(sv["log_pL"])

    # e-step autograd graph at (E*, P): reused for all linear-in-cotangent transposed products
    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        E_new_g, E_s1_g, E_s2_g, Ebar_g = e_step_triton_autograd(
            E_req, sv["log_pS"], sv["log_pD"], sv["log_pL"], sv["max_coupling"], col,
            parent, c1, c2, mad, use_col_weights=False,
        )

    def jt_E(g_new):
        with torch.enable_grad():
            (out,) = torch.autograd.grad(E_new_g, E_req, grad_outputs=g_new.contiguous(),
                                         retain_graph=True)
        return out

    def aux_T(g_s1, g_s2, g_ebar):
        with torch.enable_grad():
            (out,) = torch.autograd.grad((E_s1_g, E_s2_g, Ebar_g), E_req,
                                         grad_outputs=(g_s1, g_s2, g_ebar), retain_graph=True)
        return out

    norm = (1 - torch.exp2(E_star).mean(dim=-1, keepdim=True)).clamp_min(torch.finfo(dtype).tiny)
    fam_factor = 1.0 if G == n_fam else float(n_fam)

    zeros_state = lambda: torch.zeros_like(E_star)

    # e-step head VJP at fixed cotangents (used both for the u-independent primal cotangents
    # below and, per-u, for the tangent cotangents with g_new=dwE)
    def e_bwd_params(g_new, g_ebar):
        with torch.enable_grad():
            pS_r = pS_m.detach().requires_grad_(True)
            pD_r = pD_m.detach().requires_grad_(True)
            pL_r = pL_m.detach().requires_grad_(True)
            mc_r = item(sv["max_coupling"].squeeze(-1)).detach().requires_grad_(True)
            col_r = col.detach().requires_grad_(True)
            En, _, _, Eb = e_step_triton_autograd(
                E_star.detach(), pS_r, pD_r, pL_r, mc_r, col_r,
                parent, c1, c2, mad, use_col_weights=False,
            )
            outs = torch.autograd.grad((En, Eb), (pS_r, pD_r, pL_r, mc_r, col_r),
                                       grad_outputs=(g_new, g_ebar), allow_unused=True)
        return tuple(torch.zeros_like(z) if o is None else o
                     for o, z in zip(outs, (pS_m, pD_m, pL_m, pS_m, col)))

    # ---- u-INDEPENDENT setup (theta fixed across all CG iterations): primal cotangents and the
    # smooth head graph + first-order grad g1 are built ONCE here, not per hvp(u). The head's
    # forward graph is retained (create_graph) so each hvp(u) only adds phi2 + one backward. ----
    base_p = e_bwd_params(wE, acc["grad_Ebar"])
    cot_pS = acc["grad_log_pS"] + as_item_param(base_p[0], G, S)
    cot_pD = acc["grad_log_pD"] + as_item_param(base_p[1], G, S)
    cot_pL = base_p[2]
    cot_mc = acc["grad_mc"] + base_p[3]
    cot_col = acc["grad_col"] + base_p[4]

    theta_req = theta.detach().requires_grad_(True)
    col_req = col_weights.detach().requires_grad_(True)
    _head_grad_ctx = torch.enable_grad()
    _head_grad_ctx.__enter__()
    pS_h, pD_h, pL_h, mt_h, col_h = extract_parameters_weighted_cols(
        theta_req, col_req, sh, statewise=static.statewise, itemwise=static.itemwise,
        uniform_fast=True,
    )
    pS_hp = as_item_param(pS_h, G, S)
    pD_hp = as_item_param(pD_h, G, S)
    pL_hi = as_item_state(pL_h, S, G)
    mt_hi = as_item_state(mt_h.squeeze(-1) if mt_h.ndim == pS_h.ndim + 1 else mt_h, S, G)
    phi1 = ((pS_hp * cot_pS).sum() + (pD_hp * cot_pD).sum() + (pL_hi * cot_pL).sum()
            + (mt_hi * cot_mc).sum() + (col_h * cot_col).sum())
    (g1,) = torch.autograd.grad(phi1, theta_req, create_graph=True)
    _head_grad_ctx.__exit__(None, None, None)

    def hvp(u_vec):
        u = u_vec.reshape(S, 3).to(theta.dtype)
        with torch.no_grad():
            t_root, full = jvp_root_scores(static, theta, u, sv, return_full=True,
                                           keep_d_dts=False, self_iters=tangent_self_iters)
            dcst = full["dcst"]
            dPi, dPibar = full["dPi"], full["dPibar"]
            dpS_m, dpD_m, dpL_m = item(full["dlog_pS"]), item(full["dlog_pD"]), item(full["dlog_pL"])
            dmc_m = item(full["dmax_coupling"].squeeze(-1))
            dE, dEbar_e = full["dE"], full["dEbar"]
            dE_s1, dE_s2 = full["dE_s1"], full["dE_s2"]

            # tangent of the loss seed -q on root rows
            root_Pi = sv["pi_wave"].index_select(0, root_ids)
            q = _safe_exp2_ratio(root_Pi, _logsumexp2(root_Pi, dim=-1, keepdim=True))
            d_seed = -_LN2 * q * (t_root - (q * t_root).sum(dim=-1, keepdim=True))

            d_rhs = torch.zeros((C, S), device=theta.device, dtype=dtype)
            d_rhs.index_copy_(0, root_ids, d_seed.to(dtype))

            d_gpD = torch.zeros_like(acc["grad_log_pD"])
            d_gpS = torch.zeros_like(acc["grad_log_pS"])
            d_gE, d_gEbar, d_gEs1, d_gEs2 = (zeros_state() for _ in range(4))
            d_gmc = torch.zeros_like(acc["grad_mc"])
            d_gcol = torch.zeros((S,), device=theta.device, dtype=dtype)

            from newton.vg import free_cuda_cache_if_tight

            for wave in cache["waves"]:  # already reverse order
                free_cuda_cache_if_tight()
                ws, W = wave["ws"], wave["W"]
                meta = wave["meta"]
                v_k = wave["v"]
                dts_r = wave["dts_r"]
                # recompute d_dts per wave from the cached (pruned) dts_r: storing all of them
                # would cost another Pi-sized buffer; one tangent launch per wave is cheap
                if dts_r is not None:
                    from kbench.core.kernels.dts_tangent import compute_dts_tangent
                    d_dts = compute_dts_tangent(
                        sv["pi_wave"], sv["pibar_wave"], dPi, dPibar, meta["sl"], meta["sr"],
                        c1, c2, W, meta["reduce_idx"], cst["pd_param"], cst["ps_param"],
                        dcst["dpd_param"], dcst["dps_param"], dts_r, item_idx,
                        log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
                        eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
                        ge2_parent_ids=meta.get("ge2_parent_ids"),
                        ge2_max_fanout=meta.get("ge2_max_fanout"), item_offset=ws,
                    )
                else:
                    d_dts = None
                has_leaf = wave["has_leaf_term"]
                # (a) second-order contraction at fixed v_k
                d_Av, c_aw0, c_aw1, c_aw2, c_aw345, c_aw3, c_aw4 = wave_backward_so(
                    sv["pi_wave"], dPi, sv["pibar_wave"], dPibar, v_k, ws, W, S,
                    prm, cst["mc"], cst["dl"], dcst["dDL"], cst["ebar"], dcst["dEbar"],
                    cst["e"], dcst["dE"], cst["sl1"], dcst["dSL1"], cst["sl2"], dcst["dSL2"],
                    col, c1, c2, parent, mad, dts_r, d_dts,
                    leaf_state_idx=leaf_state_idx, leaf_logp=cst["leaf"], dleaf_logp=dcst["dleaf"],
                    item_idx=item_idx, has_leaf_term=has_leaf, use_col_weights=False,
                )
                # (b) tangent-adjoint solve with the SAME operator and cached mask
                seed = d_rhs[ws:ws + W] + d_Av
                dv, l_aw0, l_aw1, l_aw2, l_aw345, l_aw3, l_aw4 = wave_backward_uniform_fused(
                    sv["pi_wave"], sv["pibar_wave"], ws, W, S, dts_r, seed, cst["mc"],
                    cst["dl"], cst["ebar"], cst["e"], cst["sl1"], cst["sl2"], col,
                    c1, c2, None, neumann_terms=int(so.neumann_terms),
                    leaf_state_idx=leaf_state_idx, leaf_logp=cst["leaf"], has_leaf_term=has_leaf,
                    active_mask=wave["active_mask"], node_parent=parent, max_ancestor_depth=mad,
                    pibar_row_max=prm, item_idx=item_idx, item_indexed_consts=True,
                    compact_level_ptr=sh["compact_level_ptr"],
                    compact_level_parents=sh["compact_level_parents"],
                    compact_level_child1=sh["compact_level_child1"],
                    compact_level_child2=sh["compact_level_child2"],
                    grad_col_log_probs=d_gcol, use_col_weights=False,
                    self_loop_solver=so.self_loop_solver, return_last_increment=False,
                )
                aw0 = c_aw0 + l_aw0
                aw1 = c_aw1 + l_aw1
                aw2 = c_aw2 + l_aw2
                aw345 = c_aw345 + l_aw345
                aw3 = c_aw3 + l_aw3
                aw4 = c_aw4 + l_aw4
                if debug_out is not None:
                    debug_out.setdefault("wave_trace", []).append(
                        (ws, float(d_Av.abs().max()), float(dv.abs().max()),
                         float(d_rhs.abs().max())))
                rows_i = item_idx[ws:ws + W]
                _scatter_accum(d_gpD, rows_i, aw0, G)
                _scatter_accum(d_gpS, rows_i, aw345, G)
                _scatter_accum(d_gE, rows_i, aw0 + aw2, G)
                _scatter_accum(d_gEbar, rows_i, aw1, G)
                _scatter_accum(d_gEs1, rows_i, aw4, G)
                _scatter_accum(d_gEs2, rows_i, aw3, G)
                _scatter_accum(d_gmc, rows_i, aw2, G)
                if dts_r is not None:
                    # C^T dv via the frozen kernels (linear in v)
                    gl, gr, side_act, _p1, _p2 = dts_cross_backward_accum_fused(
                        sv["pi_wave"], sv["pibar_wave"], dv, ws, meta["sl"], meta["sr"],
                        meta["reduce_idx"],
                        meta.get("log_split_probs", meta["sl"].new_zeros((int(meta["sl"].numel()),), dtype=dtype)),
                        cst["pd_param"], cst["ps_param"], c1, c2, d_rhs, S,
                        active_mask=wave["active_mask"], merge_s_term=True,
                        grad_log_pD=d_gpD, grad_log_pS=d_gpS, grad_mt=d_gmc,
                        accum_param_reductions=True, accum_mt_reduction=True,
                        output_pibar_ud=True, output_pibar_side_active=True,
                        pibar_side_threshold=so.pibar_side_threshold, mt_squeezed=cst["mc"],
                        pibar_row_max=prm,
                        grad_mt_two_stage=bool(d_gmc.ndim == 2 and int(d_gmc.shape[0]) == 1),
                        grad_mt_two_stage_tile_splits=128, skip_inactive_pibar_output_zero=True,
                        item_idx=item_idx,
                    )
                    uniform_cross_pibar_vjp_tree_from_ud_fused(
                        sv["pi_wave"], col, gl, gr, meta["sl"], meta["sr"], d_rhs, S,
                        active_mask=wave["active_mask"], reduce_idx=meta["reduce_idx"],
                        pibar_row_max=prm, skip_zero_sides=True, side_active=side_act,
                        compact_level_ptr=sh["compact_level_ptr"],
                        compact_level_parents=sh["compact_level_parents"],
                        compact_level_child1=sh["compact_level_child1"],
                        compact_level_child2=sh["compact_level_child2"],
                        grad_col_log_probs=d_gcol, use_col_weights=False,
                        side_active_threshold=so.pibar_side_threshold,
                    )
                    # d(C^T) v_k contraction at fixed v_k
                    dts_backward_so(
                        sv["pi_wave"], dPi, sv["pibar_wave"], dPibar, v_k, ws, meta, S,
                        cst["pd_param"], cst["ps_param"], dcst["dpd_param"], dcst["dps_param"],
                        cst["mc"], dcst["dMC"], col, c1, c2, parent, mad, prm, item_idx,
                        d_rhs, d_gpD, d_gpS, d_gmc, d_gcol,
                        compact_level_ptr=sh["compact_level_ptr"],
                        compact_level_parents=sh["compact_level_parents"],
                        compact_level_child1=sh["compact_level_child1"],
                        compact_level_child2=sh["compact_level_child2"],
                        use_col_weights=False,
                    )

            # ---- E-side ---- (the big tangent buffers are no longer needed)
            del dPi, dPibar
            full.clear()
            free_cuda_cache_if_tight()
            x_args = (E_star.contiguous(), E_star.contiguous(), sv["E_s1"], sv["E_s2"],
                      sv["Ebar"], pS_m, pD_m, pL_m, col.contiguous(),
                      parent, c1, c2, mad)
            dx = (dE, dE, dE_s1, dE_s2, dEbar_e, dpS_m, dpD_m, dpL_m, None)
            zero_g = zeros_state()
            # tangent of aux_to_e: linear part + contraction + norm-term closed form
            aux_lin = aux_T(d_gEs1, d_gEs2, d_gEbar)
            so_aux = e_step_backward_so(*x_args, zero_g, acc["grad_E_s1"], acc["grad_E_s2"],
                                        acc["grad_Ebar"], *dx, use_col_weights=False)
            e2E = torch.exp2(E_star)
            dnorm = -_LN2 * (e2E * dE).mean(dim=-1, keepdim=True)
            dg_norm = fam_factor * (-_LN2 * e2E * dE / (S * norm) + e2E * dnorm / (S * norm * norm))
            dq_E = d_gE + aux_lin + so_aux[0] + dg_norm
            # tangent E-adjoint solve: same operator, new rhs
            so_w = e_step_backward_so(*x_args, wE, zero_g, zero_g, zero_g, *dx,
                                      use_col_weights=False)
            rhs_E = (dq_E + so_w[0]).reshape(-1)
            E_shape = E_star.shape

            def AG_flat(w_flat):
                gE = jt_E(w_flat.view(E_shape))
                return (w_flat.view(E_shape) - gE).reshape(-1)

            dwE = _bicgstab(AG_flat, rhs_E, max_iter=so.bicgstab_max_iter,
                            tol=float(so.bicgstab_tol), breakdown_tol=so.bicgstab_breakdown_tol
                            ).view(E_shape)
            if debug_out is not None:
                debug_out.update(
                    d_gE=d_gE.clone(), d_gpD=d_gpD.clone(), d_gpS=d_gpS.clone(),
                    d_gmc=d_gmc.clone(), d_gEbar=d_gEbar.clone(), d_gEs1=d_gEs1.clone(),
                    d_gEs2=d_gEs2.clone(), dq_E=dq_E.clone(), dwE=dwE.clone(),
                )

            # tangent param-cotangents from the e-step head: linear (tangent cotangents,
            # g_new=dwE) + contraction at fixed cotangents (g_new=wE, g_ebar=grad_Ebar_acc).
            # e_bwd_params and the primal cotangents/head graph are hoisted (u-independent).
            lin_p = e_bwd_params(dwE, d_gEbar)
            so_p = e_step_backward_so(*x_args, wE, zero_g, zero_g, acc["grad_Ebar"], *dx,
                                      use_col_weights=False)
            # so_p outputs: (d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_mc, d_grad_col)

            d_cot_pS = d_gpS + as_item_param(lin_p[0] + so_p[1], G, S)
            d_cot_pD = d_gpD + as_item_param(lin_p[1] + so_p[2], G, S)
            d_cot_pL = lin_p[2] + so_p[3]
            d_cot_mc = d_gmc + lin_p[3] + so_p[4]
            d_cot_col = d_gcol + lin_p[4] + so_p[5]

        # ---- smooth parameter head (autograd; forward graph + g1 hoisted, retained) ----
        with torch.enable_grad():
            phi2 = ((pS_hp * d_cot_pS).sum() + (pD_hp * d_cot_pD).sum() + (pL_hi * d_cot_pL).sum()
                    + (mt_hi * d_cot_mc).sum() + (col_h * d_cot_col).sum())
            # head Hessian term + linear term in ONE backward (they share the forward graph);
            # retain_graph so the hoisted forward graph + g1 survive for the next hvp(u) call
            (out,) = torch.autograd.grad((g1 * u).sum() + phi2, theta_req, retain_graph=True)
        return out.reshape(-1)

    return hvp


def _gate_grad_reproduction(label: str = "small") -> bool:
    """Step-2 gate: the cache-collecting sweep must reproduce the golden grad_theta."""
    from kbench.runtime import move_to_device
    from newton.vg import load_problem

    cap, static, theta, col_weights = load_problem(label)
    sv = {k: move_to_device(v, theta.device) for k, v in cap["forward_saved"].items()}
    gt, gc, cache = build_point_cache(static, theta, col_weights, sv)
    gold = cap["golden"]["grad_theta"].to(theta.device).float()
    abs_err = float((gt.float() - gold).abs().max())
    rel_err = abs_err / max(float(gold.abs().max()), 1e-30)
    n_waves = len(cache["waves"])
    n_cached = sum(int(w["v"].numel()) for w in cache["waves"])
    ok = abs_err <= 2e-3 or rel_err <= 2e-3
    print(f"[hvp_exact cache {label}] waves={n_waves} cached_v_elems={n_cached}  "
          f"wE={tuple(cache['e_side']['wE'].shape)}")
    print(f"  grad vs golden: max_abs={abs_err:.3e} max_rel={rel_err:.3e} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys

    raise SystemExit(0 if _gate_grad_reproduction(sys.argv[1] if len(sys.argv) > 1 else "small") else 1)
