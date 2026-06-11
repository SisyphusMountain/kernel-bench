import torch

from ..kernels.dts_fused import compute_dts_forward
from ..kernels.wave_step import compute_leaf_initial_wave_step, compute_wave_step
from ..parameters.extract_parameters import as_item_param, as_item_state


def pi_wave_forward(
    wave_layout,
    state_helpers,
    e,
    e_bar,
    e_s1,
    e_s2,
    log_p_s,
    log_p_d,
    max_coupling_mat,
    col_log_probs,
    use_col_weights: bool = True,
    *,
    item_idx: torch.Tensor,
    pi_iters: int = 6,
    pi_residual_out: torch.Tensor | None = None,
):
    pi_iters = int(pi_iters)
    if pi_iters < 2 or pi_iters % 2 != 0:
        raise ValueError("pi_iters must be an even integer at least 2")

    C = int(wave_layout["leaf_state_index"].numel())
    S = int(state_helpers["S"])
    device = e.device
    dtype = e.dtype

    pi = torch.empty((C, S), dtype=dtype, device=device)
    pibar = torch.empty((C, S), dtype=dtype, device=device)

    item_rows = int(e.shape[0])
    e_item = as_item_state(e, S, item_rows)
    e_bar_item = as_item_state(e_bar, S, item_rows)
    e_s1_item = as_item_state(e_s1, S, item_rows)
    e_s2_item = as_item_state(e_s2, S, item_rows)
    max_coupling_item = as_item_state(max_coupling_mat.squeeze(-1), S, item_rows)
    log_p_d_param = as_item_param(log_p_d, item_rows, S)
    log_p_s_param = as_item_param(log_p_s, item_rows, S)
    log_p_d_item = as_item_state(log_p_d, S, item_rows)
    log_p_s_item = as_item_state(log_p_s, S, item_rows)
    uniform_pibar_row_max = torch.empty((C,), dtype=dtype, device=device)

    node_child1 = state_helpers["node_child1"]
    node_child2 = state_helpers["node_child2"]
    node_parent = state_helpers["node_parent"]
    node_subtree_start = state_helpers["node_subtree_start"]
    node_subtree_end = state_helpers["node_subtree_end"]
    max_ancestor_depth = int(state_helpers["max_ancestor_depth"])

    dl_const = 1.0 + log_p_d_item + e_item
    sl1_const = log_p_s_item + e_s2_item
    sl2_const = log_p_s_item + e_s1_item

    for meta in wave_layout["wave_metas"]:
        ws = meta["start"]
        W = meta["W"]
        dts_r = (
            compute_dts_forward(
                pi, pibar, meta["sl"], meta["sr"], node_child1, node_child2,
                W, meta["reduce_idx"], log_p_d_param, log_p_s_param,
                item_idx=item_idx,
                log_split_probs=meta.get("log_split_probs"),
                n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"),
                ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"),
                ge2_max_fanout=meta.get("ge2_max_fanout"),
                item_offset=ws,
            )
            if "sl" in meta
            else None
        )
        has_leaf_term = "sl" not in meta
        for local_iter in range(pi_iters):
            pi_in = pi if (local_iter % 2 == 0) else pibar
            pi_out = pibar if (local_iter % 2 == 0) else pi
            if local_iter == 0 and not has_leaf_term:
                continue
            elif local_iter == 0:
                compute_leaf_initial_wave_step(
                    pi_out, ws, W, S,
                    max_coupling_item, dl_const, e_bar_item, e_item, sl1_const, sl2_const,
                    col_log_probs,
                    node_child1, node_child2, node_subtree_start, node_subtree_end,
                    wave_layout["leaf_state_index"],
                    log_p_s_item,
                    item_idx=item_idx,
                    use_col_weights=use_col_weights,
                )
            else:
                step_input_ws = 0 if local_iter == 1 and not has_leaf_term else None
                compute_wave_step(
                    dts_r if step_input_ws == 0 else pi_in, pi_out, pibar, ws, W, S,
                    max_coupling_item, dl_const, e_bar_item, e_item, sl1_const, sl2_const,
                    col_log_probs,
                    node_child1, node_child2, node_parent, max_ancestor_depth,
                    dts_r,
                    leaf_state_idx=wave_layout["leaf_state_index"],
                    leaf_logp=log_p_s_item,
                    item_idx=item_idx,
                    pibar_row_max=uniform_pibar_row_max,
                    store_final_pibar=local_iter == pi_iters - 1,
                    has_leaf_term=has_leaf_term,
                    input_ws=step_input_ws,
                    use_col_weights=use_col_weights,
                    pi_residual_out=(
                        pi_residual_out if local_iter == pi_iters - 1 else None
                    ),
                )

    return pi[wave_layout["root_row_ids"]], pi, pibar, uniform_pibar_row_max


Pi_wave_forward = pi_wave_forward
