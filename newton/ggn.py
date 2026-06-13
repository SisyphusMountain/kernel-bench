"""Gauss-Newton / Fisher Hessian-vector product:  M v = J^T B J v.

``J = d(Pi_root)/dtheta`` (forward tangent, ``newton/forward_tangent.py``);
``B_i = ln2 (diag(q_i) - q_i q_i^T)`` is the posterior/Fisher covariance of the root softmax
``q_i = softmax2(Pi_root[i,:])`` (PSD, so M is PSD and CG always converges);
``J^T`` is the existing wave adjoint, reused here with a *custom root seed* and with the loss's
explicit E-norm term dropped (it is not part of ``d(Pi_root)/dtheta``).

``vjp_root_to_theta`` is a faithful copy of ``implicit_grad_loglik_vjp_wave``
(kbench/api/_implicit_grad.py) parametrized by ``seed_root`` and ``drop_norm``; kept in ``newton/``
so the frozen ``kbench/api`` is not edited. ``_check_vjp_matches_golden`` regresses it against the
real backward (seed = -q, norm kept).
"""

from __future__ import annotations

import math

import torch

from kbench.core.inference.logspace import logsumexp2 as _logsumexp2
from kbench.core.kernels.dts_fused import compute_dts_forward
from kbench.core.kernels.e_step import e_step_triton_autograd
from kbench.core.kernels.wave_backward import (
    active_mask_from_rhs_absmax_fused,
    dts_cross_backward_accum_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
)
from kbench.core.parameters.extract_parameters import (
    as_item_param, as_item_state, extract_parameters_weighted_cols,
)
from kbench.api._implicit_grad import _bicgstab, _safe_exp2_ratio

from newton.forward_tangent import jvp_root_scores

_LN2 = 0.6931471805599453


@torch.no_grad()
def vjp_root_to_theta(static, sv, seed_root, theta, col_weights, *, drop_norm=True,
                      neumann_terms=None, use_pruning=None, bicgstab_tol=None, cache=None):
    """J^T applied to a root-score cotangent ``seed_root`` [n_root, S] -> grad_theta [S, 3].

    With ``seed_root=None`` the loss seed ``-softmax2(Pi_root)`` is used and ``drop_norm`` should be
    False to reproduce the real gradient (regression path). ``neumann_terms``/``use_pruning``
    override the solver options so the adjoint can be made convergent + unpruned to match the
    convergent Jvp (so M = J^T B J is symmetric).
    """
    so = static.solver_options
    wave_layout = static.wave_layout
    state_helpers = static.state_helpers
    item_idx = static.rate_item_idx
    statewise, itemwise = static.statewise, static.itemwise
    neumann_terms = int(so.neumann_terms if neumann_terms is None else neumann_terms)
    use_pruning = bool(so.use_adjoint_pruning if use_pruning is None else use_pruning)
    self_loop_solver = so.self_loop_solver
    use_col_weights = False  # theta-only, uniform col_weights fixture

    Pi_star_wave = sv["pi_wave"]
    Pibar_star_wave = sv["pibar_wave"]
    E_star, E_s1, E_s2, Ebar = sv["E"], sv["E_s1"], sv["E_s2"], sv["Ebar"]
    log_pS, log_pD, log_pL = sv["log_pS"], sv["log_pD"], sv["log_pL"]
    max_coupling_mat, col_log_probs = sv["max_coupling"], sv["col_log_probs"]
    uniform_pibar_row_max = sv["pibar_row_max"]

    C, S = Pi_star_wave.shape
    device, dtype = Pi_star_wave.device, Pi_star_wave.dtype
    item_rows = int(E_star.shape[0])
    E_item, Ebar_item, log_pS_item, log_pD_item, max_coupling_item = (
        as_item_state(x, S, item_rows) for x in (E_star, Ebar, log_pS, log_pD, max_coupling_mat)
    )
    log_pD_param, log_pS_param = (as_item_param(x, item_rows, S) for x in (log_pD, log_pS))
    DL_item = 1.0 + log_pD_item + E_item
    SL1_item = log_pS_item + as_item_state(E_s2, S, item_rows)
    SL2_item = log_pS_item + as_item_state(E_s1, S, item_rows)
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_log_pD, grad_log_pS = (torch.zeros_like(x) for x in (log_pD_param, log_pS_param))
    grad_max_coupling_mat = torch.zeros_like(max_coupling_item)
    grad_col_log_probs = torch.zeros((S,), device=device, dtype=dtype)
    grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc = (
        torch.zeros_like(x) for x in (E_star, Ebar, E_star, E_star)
    )
    root_ids = wave_layout["root_row_ids"]
    root_Pi = Pi_star_wave.index_select(0, root_ids)
    root_lse = _logsumexp2(root_Pi, dim=-1, keepdim=True)
    if seed_root is None:
        seed_root = -_safe_exp2_ratio(root_Pi, root_lse)
    accumulated_rhs.index_copy_(0, root_ids, seed_root.to(dtype))

    def _scatter_accum(acc, item_rows_for_wave, contrib):
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

    node_child1 = state_helpers["node_child1"]
    node_child2 = state_helpers["node_child2"]
    compact_level_ptr = state_helpers["compact_level_ptr"]
    compact_level_parents = state_helpers["compact_level_parents"]
    compact_level_child1 = state_helpers["compact_level_child1"]
    compact_level_child2 = state_helpers["compact_level_child2"]
    leaf_state_idx = wave_layout["leaf_state_index"].to(device=device, dtype=torch.int32).contiguous()

    for meta in reversed(wave_layout["wave_metas"]):
        ws = int(meta["start"])
        W = int(meta["W"])
        rhs_k = accumulated_rhs[ws : ws + W]
        active_mask = active_mask_from_rhs_absmax_fused(
            rhs_k, so.adjoint_pruning_threshold, use_pruning=use_pruning,
        ).contiguous()
        has_splits = bool(meta.get("has_splits", "sl" in meta))
        has_leaf_term = int(meta.get("phase", 1 if not has_splits else 2)) == 1
        dts_r = (
            compute_dts_forward(
                Pi_star_wave.detach(), Pibar_star_wave.detach(), meta["sl"], meta["sr"],
                node_child1, node_child2, W, meta["reduce_idx"], log_pD_param, log_pS_param,
                item_idx=item_idx, log_split_probs=meta.get("log_split_probs"),
                n_eq1=meta.get("n_eq1"), eq1_reduce_idx=meta.get("eq1_reduce_idx"),
                ge2_ptr=meta.get("ge2_ptr"), ge2_parent_ids=meta.get("ge2_parent_ids"),
                ge2_max_fanout=meta.get("ge2_max_fanout"), active_parent_rows=active_mask,
                item_offset=ws,
            )
            if has_splits else None
        )
        v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_backward_uniform_fused(
            Pi_star_wave, Pibar_star_wave, ws, W, S, dts_r, rhs_k, max_coupling_item,
            DL_item, Ebar_item, E_item, SL1_item, SL2_item, col_log_probs,
            node_child1, node_child2, None, neumann_terms=neumann_terms,
            leaf_state_idx=leaf_state_idx, leaf_logp=log_pS_item, has_leaf_term=has_leaf_term,
            active_mask=active_mask, node_parent=state_helpers["node_parent"],
            max_ancestor_depth=int(state_helpers["max_ancestor_depth"]),
            pibar_row_max=uniform_pibar_row_max, item_idx=item_idx, item_indexed_consts=True,
            compact_level_ptr=state_helpers["compact_level_ptr"],
            compact_level_parents=compact_level_parents, compact_level_child1=compact_level_child1,
            compact_level_child2=compact_level_child2, grad_col_log_probs=grad_col_log_probs,
            use_col_weights=use_col_weights, self_loop_solver=self_loop_solver,
            return_last_increment=False,
        )
        if cache is not None:
            # per-wave adjoint state for the exact-HVP tangent sweep (theta fixed across CG).
            # With pruning on, the precompute kernel skips zeroing v_k rows for inactive rows
            # (uninitialized memory); the primal never reads them, but the second-order
            # contraction reads all rows -> sanitize with the row mask.
            row_active = (active_mask.reshape(W, -1) != 0).any(dim=1)
            v_clean = torch.where(row_active.unsqueeze(1), v_k, torch.zeros_like(v_k))
            cache.setdefault("waves", []).append(dict(
                ws=ws, W=W, v=v_clean, dts_r=dts_r, active_mask=active_mask,
                has_splits=has_splits, has_leaf_term=has_leaf_term, meta=meta,
            ))
        item_rows_for_wave = item_idx[ws : ws + W]
        _scatter_accum(grad_log_pD, item_rows_for_wave, aw0)
        _scatter_accum(grad_log_pS, item_rows_for_wave, aw345)
        _scatter_accum(grad_E_acc, item_rows_for_wave, aw0 + aw2)
        _scatter_accum(grad_Ebar_acc, item_rows_for_wave, aw1)
        _scatter_accum(grad_E_s1_acc, item_rows_for_wave, aw4)
        _scatter_accum(grad_E_s2_acc, item_rows_for_wave, aw3)
        _scatter_accum(grad_max_coupling_mat, item_rows_for_wave, aw2)
        if has_splits and dts_r is not None:
            sl, sr = meta["sl"], meta["sr"]
            grad_Pibar_l, grad_Pibar_r, pibar_side_active, _pD, _pS = dts_cross_backward_accum_fused(
                Pi_star_wave, Pibar_star_wave, v_k, ws, sl, sr, meta["reduce_idx"],
                meta.get("log_split_probs", sl.new_zeros((int(sl.numel()),), dtype=Pi_star_wave.dtype)),
                log_pD_param, log_pS_param, node_child1, node_child2, accumulated_rhs, S,
                active_mask=active_mask, merge_s_term=True, grad_log_pD=grad_log_pD,
                grad_log_pS=grad_log_pS, grad_mt=grad_max_coupling_mat, accum_param_reductions=True,
                accum_mt_reduction=True, output_pibar_ud=True, output_pibar_side_active=True,
                pibar_side_threshold=so.pibar_side_threshold, mt_squeezed=max_coupling_item,
                pibar_row_max=uniform_pibar_row_max,
                grad_mt_two_stage=bool(grad_max_coupling_mat.ndim == 2 and int(grad_max_coupling_mat.shape[0]) == 1),
                grad_mt_two_stage_tile_splits=128, skip_inactive_pibar_output_zero=True, item_idx=item_idx,
            )
            uniform_cross_pibar_vjp_tree_from_ud_fused(
                Pi_star_wave, col_log_probs, grad_Pibar_l, grad_Pibar_r, sl, sr, accumulated_rhs, S,
                active_mask=active_mask, reduce_idx=meta["reduce_idx"], pibar_row_max=uniform_pibar_row_max,
                skip_zero_sides=True, side_active=pibar_side_active, compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents, compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2, grad_col_log_probs=grad_col_log_probs,
                use_col_weights=use_col_weights, side_active_threshold=so.pibar_side_threshold,
            )

    if cache is not None:
        cache["accum"] = dict(
            grad_E=grad_E_acc, grad_Ebar=grad_Ebar_acc, grad_E_s1=grad_E_s1_acc,
            grad_E_s2=grad_E_s2_acc, grad_log_pD=grad_log_pD, grad_log_pS=grad_log_pS,
            grad_mc=grad_max_coupling_mat, grad_col=grad_col_log_probs,
        )
    return _e_adjoint_and_theta_vjp(
        E_star, log_pS, log_pD, log_pL, max_coupling_mat, col_log_probs, use_col_weights,
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_coupling_mat, grad_col_log_probs,
        int(root_ids.numel()), theta, col_weights, state_helpers,
        statewise=statewise, itemwise=itemwise, drop_norm=drop_norm,
        bicgstab_max_iter=so.bicgstab_max_iter,
        bicgstab_tol=float(so.bicgstab_tol if bicgstab_tol is None else bicgstab_tol),
        bicgstab_breakdown_tol=so.bicgstab_breakdown_tol,
        cache=cache,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_coupling_mat, col_log_probs, use_col_weights,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2, grad_log_pD, grad_log_pS, grad_max_coupling_mat,
    grad_col_log_probs, n_fam, theta, col_weights, state_helpers, *, statewise, itemwise,
    drop_norm, bicgstab_max_iter=500, bicgstab_tol=1e-7, bicgstab_breakdown_tol=1e-30,
    cache=None,
):
    topology_args = (
        state_helpers["node_parent"], state_helpers["node_child1"], state_helpers["node_child2"],
        int(state_helpers["max_ancestor_depth"]),
    )
    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        triton_E_from_E, E_s1_from_E, E_s2_from_E, Ebar_from_E = e_step_triton_autograd(
            E_req, log_pS, log_pD, log_pL, max_coupling_mat, col_log_probs, *topology_args,
            use_col_weights=use_col_weights,
        )
        aux_outputs = (E_s1_from_E, E_s2_from_E, Ebar_from_E)
        aux_grads = (grad_E_s1, grad_E_s2, grad_Ebar)
        if not drop_norm:
            norm = (1 - torch.exp2(E_req).mean(dim=-1)).clamp_min(torch.finfo(E_req.dtype).tiny)
            denom = torch.log2(norm)
            direct_obj = denom.sum() if E_req.shape[0] == n_fam else (n_fam * denom).sum()
            aux_outputs = (direct_obj, *aux_outputs)
            aux_grads = (torch.ones_like(direct_obj), *aux_grads)
        (aux_to_e,) = torch.autograd.grad(aux_outputs, E_req, grad_outputs=aux_grads, retain_graph=True)
    q_E = grad_E + aux_to_e
    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        with torch.enable_grad():
            (gE,) = torch.autograd.grad(triton_E_from_E, E_req, grad_outputs=wE.clone(), retain_graph=True)
        return (wE - gE).reshape(-1)

    wE = _bicgstab(AG_flat, q_flat, max_iter=bicgstab_max_iter, tol=bicgstab_tol,
                   breakdown_tol=bicgstab_breakdown_tol).view(E_shape)
    if cache is not None:
        cache["e_side"] = dict(q_E=q_E, wE=wE, aux_to_e=aux_to_e)

    theta_req = theta.detach().requires_grad_(True)
    col_req = col_weights.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, mt_r, col_log_probs_r = extract_parameters_weighted_cols(
            theta_req, col_req, state_helpers, statewise=statewise, itemwise=itemwise,
            uniform_fast=not use_col_weights,
        )
        S = int(state_helpers["S"])
        item_rows = int(E_star.shape[0])
        log_pS_param = as_item_param(log_pS_r, item_rows, S)
        log_pD_param = as_item_param(log_pD_r, item_rows, S)
        param_loss = (
            (log_pS_param * grad_log_pS).sum() + (log_pD_param * grad_log_pD).sum()
            + (mt_r * grad_max_coupling_mat).sum() + (col_log_probs_r * grad_col_log_probs).sum()
        )
        E_from_params, _, _, Ebar_from_params = e_step_triton_autograd(
            E_star.detach(), log_pS_r, log_pD_r, log_pL_r, mt_r, col_log_probs_r, *topology_args,
            use_col_weights=use_col_weights,
        )
        grad_theta, grad_col = torch.autograd.grad(
            (param_loss, Ebar_from_params, E_from_params), (theta_req, col_req),
            grad_outputs=(torch.ones_like(param_loss), grad_Ebar, wE),
        )
    return grad_theta, grad_col


def make_ggn_hvp(static, theta, col_weights, sv, *, self_tol=None, self_max_iter=200,
                 vjp_neumann_terms=None, vjp_use_pruning=None, vjp_bicgstab_tol=None):
    """Return hvp(v_vec) computing the GGN/Fisher product M v in theta-space (flat 3S).

    Defaults use the solver's production adjoint settings (neumann_terms, pruning, bicgstab tol);
    the wave self-loop already converges within those terms, so M is unchanged vs the convergent
    settings (pass overrides only to force a convergent/unpruned adjoint for symmetry checks).
    """
    S = int(static.state_helpers["S"])
    root_ids = static.wave_layout["root_row_ids"]
    root_Pi = sv["pi_wave"].index_select(0, root_ids)
    root_lse = _logsumexp2(root_Pi, dim=-1, keepdim=True)
    q = _safe_exp2_ratio(root_Pi, root_lse)  # posterior softmax2 per root row

    def hvp(v_vec):
        v = v_vec.reshape(S, 3).to(theta.dtype)
        t = jvp_root_scores(static, theta, v, sv, self_tol=self_tol, self_max_iter=self_max_iter)
        u = _LN2 * q * (t - (q * t).sum(dim=-1, keepdim=True))  # B t  (PSD Fisher covariance)
        gt, _gc = vjp_root_to_theta(static, sv, u, theta, col_weights, drop_norm=True,
                                    neumann_terms=vjp_neumann_terms, use_pruning=vjp_use_pruning,
                                    bicgstab_tol=vjp_bicgstab_tol)
        return gt.reshape(-1)

    return hvp
