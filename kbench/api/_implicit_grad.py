import torch

from kbench.core.inference.logspace import logsumexp2 as _logsumexp2
from kbench.core.kernels.dts_fused import compute_dts_forward
from kbench.core.kernels.wave_backward import (
    active_mask_from_rhs_absmax_fused,
    dts_cross_backward_accum_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
)
from kbench.core.parameters.extract_parameters import (
    as_item_param,
    as_item_state,
    extract_parameters_weighted_cols,
)
from kbench.core.kernels.e_step import e_step_triton_autograd

_NEG_INF = float("-inf")


def _safe_exp2_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    neg_inf = a == _NEG_INF
    a_safe = torch.where(neg_inf, torch.zeros_like(a), a)
    b_safe = torch.where(neg_inf, torch.zeros_like(b), b)
    return torch.where(neg_inf, torch.zeros_like(a), torch.exp2(a_safe - b_safe))


@torch.no_grad()
def _bicgstab(
    Av,
    b: torch.Tensor,
    *,
    max_iter: int = 500,
    tol: float = 1e-7,
    breakdown_tol: float = 1e-30,
):
    max_iter = int(max_iter)
    tol = float(tol)
    breakdown_tol = float(breakdown_tol)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if breakdown_tol <= 0.0:
        raise ValueError("breakdown_tol must be positive")

    x = torch.zeros_like(b)
    r = b - Av(x)
    bnorm = max(float(torch.linalg.vector_norm(b).detach().cpu()), 1.0)
    rel_res = float(torch.linalg.vector_norm(r).detach().cpu()) / bnorm
    if rel_res <= tol:
        return x

    r_hat = r.clone()
    rho_old = torch.ones((), dtype=b.dtype, device=b.device)
    alpha = torch.ones((), dtype=b.dtype, device=b.device)
    omega = torch.ones((), dtype=b.dtype, device=b.device)
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    for k in range(1, max_iter + 1):
        rho = torch.dot(r_hat, r)
        if float(rho.abs().detach().cpu()) <= breakdown_tol:
            break

        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = Av(p)
        denom = torch.dot(r_hat, v)
        if float(denom.abs().detach().cpu()) <= breakdown_tol:
            break

        alpha = rho / denom
        s = r - alpha * v
        rel_s = float(torch.linalg.vector_norm(s).detach().cpu()) / bnorm
        if rel_s <= tol:
            return x + alpha * p

        t = Av(s)
        tt = torch.dot(t, t)
        if float(tt.abs().detach().cpu()) <= breakdown_tol:
            break

        omega = torch.dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        rel_res = float(torch.linalg.vector_norm(r).detach().cpu()) / bnorm
        if rel_res <= tol:
            return x
        if float(omega.abs().detach().cpu()) <= breakdown_tol:
            break
        rho_old = rho

    raise RuntimeError(f"E-adjoint BiCGSTAB solve failed after {k} iterations (relative residual {rel_res:.3e})")


@torch.no_grad()
def implicit_grad_loglik_vjp_wave(
    wave_layout, state_helpers, *, Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor, E_star: torch.Tensor, E_s1: torch.Tensor,
    E_s2: torch.Tensor, Ebar: torch.Tensor, log_pS: torch.Tensor,
    log_pD: torch.Tensor, log_pL: torch.Tensor, max_coupling_mat: torch.Tensor,
    col_log_probs: torch.Tensor,
    use_col_weights: bool,
    theta: torch.Tensor, col_weights: torch.Tensor, uniform_pibar_row_max: torch.Tensor,
    item_idx: torch.Tensor,
    statewise: bool = False,
    itemwise: bool = False,
    neumann_terms: int = 3,
    self_loop_solver: str = "neumann",
    bicgstab_max_iter: int = 500,
    bicgstab_tol: float = 1e-7,
    bicgstab_breakdown_tol: float = 1e-30,
    adjoint_pruning_threshold: float = 1e-6,
    use_adjoint_pruning: bool = True,
    pibar_side_threshold: float = 0.0,
    collect_backward_relres: bool = False,
):
    neumann_terms = int(neumann_terms)
    if neumann_terms < 0:
        raise ValueError("neumann_terms must be non-negative")
    self_loop_solver = str(self_loop_solver).strip().lower()
    if self_loop_solver not in ("neumann", "gmres"):
        raise ValueError("self_loop_solver must be one of: neumann, gmres")
    adjoint_pruning_threshold = float(adjoint_pruning_threshold)
    if adjoint_pruning_threshold < 0.0:
        raise ValueError("adjoint_pruning_threshold must be non-negative")
    pibar_side_threshold = float(pibar_side_threshold)
    if pibar_side_threshold < 0.0:
        raise ValueError("pibar_side_threshold must be non-negative")

    C, S = Pi_star_wave.shape
    device = Pi_star_wave.device
    dtype = Pi_star_wave.dtype
    item_rows = int(E_star.shape[0])
    E_item, Ebar_item, log_pS_item, log_pD_item, max_coupling_item = (
        as_item_state(x, S, item_rows)
        for x in (E_star, Ebar, log_pS, log_pD, max_coupling_mat)
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
    accumulated_rhs.index_copy_(
        0,
        root_ids,
        -_safe_exp2_ratio(root_Pi, root_lse),
    )
    def _scatter_accum(acc: torch.Tensor, item_rows_for_wave: torch.Tensor, contrib: torch.Tensor) -> None:
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



    backward_relres = None
    backward_vk_mag = None
    if collect_backward_relres:
        row_item = wave_layout["item_idx"].to(device=device, dtype=torch.long)
        n_fam = int(row_item.max().item()) + 1 if row_item.numel() else 0
        backward_relres = torch.zeros(n_fam, device=device, dtype=torch.float32)
        backward_vk_mag = torch.zeros(n_fam, device=device, dtype=torch.float32)

    for meta in reversed(wave_layout["wave_metas"]):
        ws = int(meta["start"])
        W = int(meta["W"])
        rhs_k = accumulated_rhs[ws : ws + W]
        active_mask = active_mask_from_rhs_absmax_fused(
            rhs_k,
            adjoint_pruning_threshold,
            use_pruning=bool(use_adjoint_pruning),
        ).contiguous()
        has_splits = bool(meta.get("has_splits", "sl" in meta))
        has_leaf_term = int(meta.get("phase", 1 if not has_splits else 2)) == 1
        dts_r = (
            compute_dts_forward(
                Pi_star_wave.detach(), Pibar_star_wave.detach(), meta["sl"], meta["sr"],
                node_child1,
                node_child2,
                W,
                meta["reduce_idx"],
                log_pD_param,
                log_pS_param,
                item_idx=item_idx,
                log_split_probs=meta.get("log_split_probs"),
                n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"),
                ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"),
                ge2_max_fanout=meta.get("ge2_max_fanout"),
                active_parent_rows=active_mask,
                item_offset=ws,
            )
            if has_splits
            else None
        )
        backward_out = wave_backward_uniform_fused(
            Pi_star_wave,
            Pibar_star_wave,
            ws,
            W,
            S,
            dts_r,
            rhs_k,
            max_coupling_item,
            DL_item,
            Ebar_item,
            E_item,
            SL1_item,
            SL2_item,
            col_log_probs,
            node_child1,
            node_child2,
            None,
            neumann_terms=neumann_terms,
            leaf_state_idx=leaf_state_idx,
            leaf_logp=log_pS_item,
            has_leaf_term=has_leaf_term,
            active_mask=active_mask,
            node_parent=state_helpers["node_parent"],
            max_ancestor_depth=int(state_helpers["max_ancestor_depth"]),
            pibar_row_max=uniform_pibar_row_max,
            item_idx=item_idx,
            item_indexed_consts=True,
            compact_level_ptr=state_helpers["compact_level_ptr"],
            compact_level_parents=compact_level_parents,
            compact_level_child1=compact_level_child1,
            compact_level_child2=compact_level_child2,
            grad_col_log_probs=grad_col_log_probs,
            use_col_weights=use_col_weights,
            self_loop_solver=self_loop_solver,
            return_last_increment=collect_backward_relres,
        )
        if collect_backward_relres:
            v_k, aw0, aw1, aw2, aw345, aw3, aw4, last_relres = backward_out
            wave_item = row_item[ws : ws + W]
            row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            vk_norm = torch.where(
                row_active, v_k.float().norm(dim=1), torch.zeros(W, device=device, dtype=torch.float32)
            )
            backward_vk_mag.scatter_reduce_(
                0,
                wave_item,
                vk_norm,
                reduce="amax",
                include_self=True,
            )
            if last_relres is not None:
                backward_relres.scatter_reduce_(
                    0,
                    wave_item,
                    last_relres.to(dtype=torch.float32),
                    reduce="amax",
                    include_self=True,
                )
        else:
            v_k, aw0, aw1, aw2, aw345, aw3, aw4 = backward_out
        item_rows_for_wave = item_idx[ws : ws + W]
        _scatter_accum(grad_log_pD, item_rows_for_wave, aw0)
        _scatter_accum(grad_log_pS, item_rows_for_wave, aw345)
        _scatter_accum(grad_E_acc, item_rows_for_wave, aw0 + aw2)
        _scatter_accum(grad_Ebar_acc, item_rows_for_wave, aw1)
        _scatter_accum(grad_E_s1_acc, item_rows_for_wave, aw4)
        _scatter_accum(grad_E_s2_acc, item_rows_for_wave, aw3)
        _scatter_accum(grad_max_coupling_mat, item_rows_for_wave, aw2)
        if has_splits and dts_r is not None:
            sl = meta["sl"]
            sr = meta["sr"]
            grad_Pibar_l, grad_Pibar_r, pibar_side_active, _param_pD, _param_pS = dts_cross_backward_accum_fused(
                Pi_star_wave,
                Pibar_star_wave,
                v_k,
                ws,
                sl,
                sr,
                meta["reduce_idx"],
                meta.get("log_split_probs", sl.new_zeros((int(sl.numel()),), dtype=Pi_star_wave.dtype)),
                log_pD_param,
                log_pS_param,
                node_child1,
                node_child2,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                merge_s_term=True,
                grad_log_pD=grad_log_pD,
                grad_log_pS=grad_log_pS,
                grad_mt=grad_max_coupling_mat,
                accum_param_reductions=True,
                accum_mt_reduction=True,
                output_pibar_ud=True,
                output_pibar_side_active=True,
                pibar_side_threshold=pibar_side_threshold,
                mt_squeezed=max_coupling_item,
                pibar_row_max=uniform_pibar_row_max,
                grad_mt_two_stage=bool(grad_max_coupling_mat.ndim == 2 and int(grad_max_coupling_mat.shape[0]) == 1),
                grad_mt_two_stage_tile_splits=128,
                skip_inactive_pibar_output_zero=True,
                item_idx=item_idx,
            )
            uniform_cross_pibar_vjp_tree_from_ud_fused(
                Pi_star_wave,
                col_log_probs,
                grad_Pibar_l,
                grad_Pibar_r,
                sl,
                sr,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                reduce_idx=meta["reduce_idx"],
                pibar_row_max=uniform_pibar_row_max,
                skip_zero_sides=True,
                side_active=pibar_side_active,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
                grad_col_log_probs=grad_col_log_probs,
                use_col_weights=use_col_weights,
                side_active_threshold=pibar_side_threshold,
            )
    if collect_backward_relres:


        return backward_relres, backward_vk_mag
    return _e_adjoint_and_theta_vjp(
        E_star, log_pS, log_pD, log_pL, max_coupling_mat,
        col_log_probs,
        use_col_weights,
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_coupling_mat, grad_col_log_probs,
        int(root_ids.numel()), theta, col_weights, state_helpers,
        statewise=statewise,
        itemwise=itemwise,
        bicgstab_max_iter=bicgstab_max_iter,
        bicgstab_tol=bicgstab_tol,
        bicgstab_breakdown_tol=bicgstab_breakdown_tol,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_coupling_mat, col_log_probs, use_col_weights,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2,
    grad_log_pD, grad_log_pS, grad_max_coupling_mat, grad_col_log_probs,
    n_fam, theta, col_weights, state_helpers, *, statewise, itemwise,
    bicgstab_max_iter: int = 500,
    bicgstab_tol: float = 1e-7,
    bicgstab_breakdown_tol: float = 1e-30,
):
    topology_args = (
        state_helpers["node_parent"],
        state_helpers["node_child1"],
        state_helpers["node_child2"],
        int(state_helpers["max_ancestor_depth"]),
    )

    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        triton_E_from_E, E_s1_from_E, E_s2_from_E, Ebar_from_E = e_step_triton_autograd(
            E_req,
            log_pS,
            log_pD,
            log_pL,
            max_coupling_mat,
            col_log_probs,
            *topology_args,
            use_col_weights=use_col_weights,
        )
        norm = (1 - torch.exp2(E_req).mean(dim=-1)).clamp_min(torch.finfo(E_req.dtype).tiny)
        denom = torch.log2(norm)
        direct_obj = denom.sum() if E_req.shape[0] == n_fam else (n_fam * denom).sum()
        (aux_to_e,) = torch.autograd.grad(
            (direct_obj, E_s1_from_E, E_s2_from_E, Ebar_from_E),
            E_req,
            grad_outputs=(
                torch.ones_like(direct_obj),
                grad_E_s1,
                grad_E_s2,
                grad_Ebar,
            ),
            retain_graph=True,
        )
    q_E = grad_E + aux_to_e

    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        with torch.enable_grad():
            (gE,) = torch.autograd.grad(
                triton_E_from_E,
                E_req,
                grad_outputs=wE.clone(),
                retain_graph=True,
            )
        return (wE - gE).reshape(-1)

    wE = _bicgstab(
        AG_flat,
        q_flat,
        max_iter=bicgstab_max_iter,
        tol=bicgstab_tol,
        breakdown_tol=bicgstab_breakdown_tol,
    ).view(E_shape)

    theta_req = theta.detach().requires_grad_(True)
    col_req = col_weights.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, mt_r, col_log_probs_r = extract_parameters_weighted_cols(
            theta_req,
            col_req,
            state_helpers,
            statewise=statewise,
            itemwise=itemwise,
            uniform_fast=not use_col_weights,
        )
        S = int(state_helpers["S"])
        item_rows = int(E_star.shape[0])
        log_pS_param = as_item_param(log_pS_r, item_rows, S)
        log_pD_param = as_item_param(log_pD_r, item_rows, S)
        param_loss = (
            (log_pS_param * grad_log_pS).sum()
            + (log_pD_param * grad_log_pD).sum()
            + (mt_r * grad_max_coupling_mat).sum()
            + (col_log_probs_r * grad_col_log_probs).sum()
        )
        E_from_params, _, _, Ebar_from_params = e_step_triton_autograd(
            E_star.detach(),
            log_pS_r,
            log_pD_r,
            log_pL_r,
            mt_r,
            col_log_probs_r,
            *topology_args,
            use_col_weights=use_col_weights,
        )
        grad_theta, grad_col = torch.autograd.grad(
            (param_loss, Ebar_from_params, E_from_params),
            (theta_req, col_req),
            grad_outputs=(torch.ones_like(param_loss), grad_Ebar, wE),
        )
    return grad_theta, grad_col
