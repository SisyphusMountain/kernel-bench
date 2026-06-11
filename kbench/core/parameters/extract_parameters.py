import math

import torch

_LN2 = 0.6931471805599453


def as_item_param(t, item_rows, S=None):
    if t.ndim == 0:
        return t.reshape(1, 1).expand(int(item_rows), 1).contiguous()
    if t.ndim == 1:
        if S is not None and int(item_rows) == 1 and int(t.numel()) == int(S):
            return t.reshape(1, int(S)).contiguous()
        return t.reshape(int(item_rows), 1).contiguous()
    return t.contiguous()


def as_item_state(t, S, item_rows):
    param = as_item_param(t, item_rows, S)
    if int(param.shape[1]) == int(S):
        return param.contiguous()
    return param.expand(int(item_rows), int(S)).contiguous()


def extract_parameters_uniform(theta, unnorm_row_max, *, statewise=False, itemwise=False):
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = torch.log_softmax(logits * _LN2, dim=-1) / _LN2
    log_pT = result[..., 3]
    if statewise and not itemwise:
        max_coupling = log_pT + unnorm_row_max
    else:
        max_coupling = log_pT.unsqueeze(-1) + unnorm_row_max
    return result[..., 0], result[..., 1], result[..., 2], max_coupling


def col_log_probs_from_weights(col_weights: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(col_weights, dim=-1) / _LN2


def col_valid_log_normalizer(
    col_log_probs: torch.Tensor,
    node_parent: torch.Tensor,
    max_ancestor_depth: int,
) -> torch.Tensor:
    S = int(col_log_probs.numel())
    col_probs = torch.exp2(col_log_probs)
    parent = node_parent.to(device=col_log_probs.device, dtype=torch.long)
    cur = torch.arange(S, device=col_log_probs.device, dtype=torch.long)
    ancestor_mass = torch.zeros((S,), device=col_log_probs.device, dtype=col_log_probs.dtype)
    zero = torch.zeros((), device=col_log_probs.device, dtype=col_log_probs.dtype)

    for _ in range(max(1, int(max_ancestor_depth))):
        valid = (cur >= 0) & (cur < S)
        safe_cur = cur.clamp(0, max(S - 1, 0))
        ancestor_mass = ancestor_mass + torch.where(valid, col_probs.index_select(0, safe_cur), zero)
        next_cur = parent.index_select(0, safe_cur)
        cur = torch.where(valid, next_cur, torch.full_like(cur, -1))

    valid_mass = 1.0 - ancestor_mass
    return torch.where(
        valid_mass > 0.0,
        -torch.log2(valid_mass.clamp_min(torch.finfo(col_log_probs.dtype).tiny)),
        col_log_probs.new_full((S,), float("-inf")),
    )


def extract_parameters_weighted_cols(
    theta,
    col_weights,
    state_helpers,
    *,
    statewise=False,
    itemwise=False,
    uniform_fast=False,
):
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = torch.log_softmax(logits * _LN2, dim=-1) / _LN2
    log_pT = result[..., 3]
    col_log_probs = col_log_probs_from_weights(col_weights.to(device=theta.device, dtype=theta.dtype))
    col_norm = col_valid_log_normalizer(
        col_log_probs,
        state_helpers["node_parent"],
        int(state_helpers["max_ancestor_depth"]),
    )
    if statewise and not itemwise:
        max_coupling = log_pT + col_norm
    else:
        max_coupling = log_pT.unsqueeze(-1) + col_norm
    if uniform_fast:
        max_coupling = max_coupling - math.log2(int(state_helpers["S"]))
    return result[..., 0], result[..., 1], result[..., 2], max_coupling, col_log_probs
