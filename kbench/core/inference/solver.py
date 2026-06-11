import math

import torch

from kbench.core.inference.forward import pi_wave_forward
from kbench.core.inference.logspace import logsumexp2
from kbench.core.kernels.e_step import e_fixed_point_triton
from kbench.core.parameters.extract_parameters import extract_parameters_uniform, extract_parameters_weighted_cols


def col_weights_are_uniform(col_weights: torch.Tensor) -> bool:
    flat = col_weights.detach().reshape(-1)
    return bool(torch.all(flat == flat[0]).item())


def solve_e_pi(
    static,
    theta: torch.Tensor,
    col_weights: torch.Tensor,
    *,
    warm_start_E: torch.Tensor | None = None,
    pi_iters: int | None = None,
    pi_residual_out: torch.Tensor | None = None,
):
    solver_options = static.solver_options
    solver_options.validate()
    use_col_weights = not col_weights_are_uniform(col_weights)
    S = int(static.state_helpers["S"])
    if use_col_weights:
        log_p_s, log_p_d, log_p_l, max_coupling, col_log_probs = extract_parameters_weighted_cols(
            theta.detach(),
            col_weights.detach(),
            static.state_helpers,
            statewise=static.statewise,
            itemwise=static.itemwise,
        )
    else:
        log_p_s, log_p_d, log_p_l, max_coupling = extract_parameters_uniform(
            theta.detach(),
            static.state_helpers["unnorm_row_max"].to(device=theta.device, dtype=theta.dtype),
            statewise=static.statewise,
            itemwise=static.itemwise,
        )
        col_log_probs = theta.new_full((S,), -math.log2(S))
    e_shape = (int(static.wave_layout["root_row_ids"].numel()) if static.itemwise else 1, S)
    E0 = (
        warm_start_E.detach().to(theta).contiguous()
        if warm_start_E is not None
        else theta.new_full(e_shape, float(solver_options.e_init))
    )
    E, E_s1, E_s2, Ebar = e_fixed_point_triton(
        E0,
        log_pS=log_p_s,
        log_pD=log_p_d,
        log_pL=log_p_l,
        max_coupling=max_coupling,
        col_log_probs=col_log_probs,
        use_col_weights=use_col_weights,
        node_parent=static.state_helpers["node_parent"],
        node_child1=static.state_helpers["node_child1"],
        node_child2=static.state_helpers["node_child2"],
        max_ancestor_depth=int(static.state_helpers["max_ancestor_depth"]),
        max_iter=solver_options.e_max_iter,
        tol=solver_options.e_tol,
    )
    root_rows, pi_wave, pibar_wave, pibar_row_max = pi_wave_forward(
        wave_layout=static.wave_layout,
        state_helpers=static.state_helpers,
        e=E,
        e_bar=Ebar,
        e_s1=E_s1,
        e_s2=E_s2,
        log_p_s=log_p_s,
        log_p_d=log_p_d,
        max_coupling_mat=max_coupling,
        col_log_probs=col_log_probs,
        use_col_weights=use_col_weights,
        item_idx=static.rate_item_idx,
        pi_iters=solver_options.pi_iters if pi_iters is None else int(pi_iters),
        pi_residual_out=pi_residual_out,
    )
    return (
        E,
        E_s1,
        E_s2,
        Ebar,
        root_rows,
        pi_wave,
        pibar_wave,
        pibar_row_max,
        log_p_s,
        log_p_d,
        log_p_l,
        max_coupling,
        col_log_probs,
    )


def solve_forward_residual(
    static,
    theta: torch.Tensor,
    col_weights: torch.Tensor,
    *,
    pi_iters: int,
    warm_start_E: torch.Tensor | None = None,
):
    """Per-item forward Pi convergence residual at a (high) ``pi_iters``.

    Runs the forward solve and captures the final iteration's per-row
    ``max_s |Pi_new - Pi_old|`` (= the size of the last Pi update). Returns a 1-D
    float tensor of length ``n_items_in_batch`` (batch-local index), holding the
    max residual over each item's rows. Diagnostic only; meaningful only at a
    converged forward, hence the caller supplies a high ``pi_iters``.
    """
    C = int(static.wave_layout["leaf_state_index"].numel())
    pi_residual = torch.zeros(C, device=theta.device, dtype=torch.float32)
    solve_e_pi(
        static,
        theta,
        col_weights,
        warm_start_E=warm_start_E,
        pi_iters=pi_iters,
        pi_residual_out=pi_residual,
    )
    fam_local = static.wave_layout["item_idx"].to(device=pi_residual.device, dtype=torch.long)
    n_fam = int(static.item_index_tensor.numel())
    per_item = torch.zeros(n_fam, device=pi_residual.device, dtype=torch.float32)
    per_item.scatter_reduce_(0, fam_local, pi_residual, reduce="amax", include_self=True)
    return per_item


def nll_vector_from_root_rows(root_rows: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    norm = (1 - torch.exp2(E).mean(dim=-1)).clamp_min(torch.finfo(E.dtype).tiny)
    return -(
        logsumexp2(root_rows, dim=-1)
        - math.log2(root_rows.shape[-1])
        - torch.log2(norm)
    )


def nll_from_root_rows(root_rows: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return nll_vector_from_root_rows(root_rows, E).sum()
