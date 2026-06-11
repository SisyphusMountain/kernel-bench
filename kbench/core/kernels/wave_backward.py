"""Fused Triton kernels for the retained wave-backward fast path.

This module also contains private standalone diagnostics/helpers used by that
path.  In particular, ``active_mask_from_rhs_absmax_fused()`` accepts bf16
inputs for standalone row-mask experiments, but the retained public
``Pi_wave_backward`` path rejects bf16 before this helper is reached.
"""

import torch
import triton
import triton.language as tl

import os

from kbench.core.kernels._dts_layout_contract import dts_backward_param_layout
from kbench.core.kernels.wave_step import _get_jumps, _get_levels, _pathsum_doubling, _pathsum_walk
from kbench.core.memory_policy import proposal0_memory_gate

_JT_WARPS = int(os.environ.get("KBENCH_JT_WARPS", "2"))
_JT_MODE = int(os.environ.get("KBENCH_JT_MODE", "1"))  # 0=scratch+level-walk 1=register cumsum
_JT_REG_WARPS = int(os.environ.get("KBENCH_JT_REG_WARPS", "4"))
_VJP_MODE = int(os.environ.get("KBENCH_VJP_MODE", "0"))  # 0=level-walk 1=register cumsum (slower on 4090)

# DFS tables for register-resident subtree sums: subtree(s) is a contiguous interval in
# DFS order, so corr[s] = cumsum(u_d in DFS order)[end-1] - cumsum[start-1]. Any DFS
# numbering works; derived once from parent/child arrays and cached by storage pointer.
_DFS_CACHE: dict = {}


def _get_dfs_tables(node_parent, node_child1, node_child2, S: int):
    key = (node_parent.data_ptr(), int(S), node_parent.device.index)
    hit = _DFS_CACHE.get(key)
    if hit is not None:
        return hit
    parent = node_parent.detach().to("cpu", torch.int64).tolist()[:S]
    c1 = node_child1.detach().to("cpu", torch.int64).tolist()[:S]
    c2 = node_child2.detach().to("cpu", torch.int64).tolist()[:S]
    roots = [s for s in range(S) if not (0 <= parent[s] < S)]
    dfs_node, start, end = [0] * S, [0] * S, [0] * S
    pos = 0
    for r in roots:
        stack = [(r, False)]
        while stack:
            node, done = stack.pop()
            if done:
                end[node] = pos
                continue
            start[node] = pos
            dfs_node[pos] = node
            pos += 1
            stack.append((node, True))
            for ch in (c2[node], c1[node]):
                if 0 <= ch < S:
                    stack.append((ch, False))
    assert pos == S, f"DFS visited {pos} of {S} nodes; not a forest?"
    device = node_parent.device
    entry = (
        torch.tensor(dfs_node, dtype=torch.int32, device=device),
        torch.tensor([start[s] - 1 for s in range(S)], dtype=torch.int32, device=device),
        torch.tensor([end[s] - 1 for s in range(S)], dtype=torch.int32, device=device),
    )
    _DFS_CACHE[key] = entry
    return entry


def _get_dfs_tables_from_compact(compact_parents, compact_child1, compact_child2, S: int):
    """Same DFS tables, derived from the compact internal-node arrays (the only tree
    description the pibar-VJP wrapper receives)."""
    key = ("compact", compact_parents.data_ptr(), int(S), compact_parents.device.index)
    hit = _DFS_CACHE.get(key)
    if hit is not None:
        return hit
    par_list = compact_parents.detach().to("cpu", torch.int64).tolist()
    c1_list = compact_child1.detach().to("cpu", torch.int64).tolist()
    c2_list = compact_child2.detach().to("cpu", torch.int64).tolist()
    parent = [-1] * S
    child1 = [S] * S
    child2 = [S] * S
    for p, c1, c2 in zip(par_list, c1_list, c2_list):
        if 0 <= c1 < S:
            parent[c1] = p
            child1[p] = c1
        if 0 <= c2 < S:
            parent[c2] = p
            child2[p] = c2
    parent_t = torch.tensor(parent, dtype=torch.int32, device=compact_parents.device)
    child1_t = torch.tensor(child1, dtype=torch.int32, device=compact_parents.device)
    child2_t = torch.tensor(child2, dtype=torch.int32, device=compact_parents.device)
    entry = _get_dfs_tables(parent_t, child1_t, child2_t, S)
    # keep parent_t alive: the inner cache is keyed by its data_ptr
    _DFS_CACHE[key] = entry
    _DFS_CACHE[key + ("keepalive",)] = (parent_t, child1_t, child2_t)
    return entry

_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64, torch.bfloat16)


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _device_scalar_param(param, *, device, dtype):
    """Return a one-element device tensor without extracting CUDA scalars."""
    if torch.is_tensor(param):
        if param.numel() != 1:
            raise ValueError("fused DTS backward scalar parameters must have one element")
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        return param.reshape(1).contiguous()
    return torch.tensor([param], device=device, dtype=dtype)


def _dts_layout_param_args(log_pD, log_pS, *, item_idx, S, device, dtype):
    """Return DTS parameter tensors plus a Triton addressing layout.

    With ``item_idx`` present, retained backward treats a one-dimensional
    tensor as item scalar rows before considering a shared ``[S]`` state
    vector.  Direct callers that need forward/backward parity when ``G == S``
    should use ``[G, 1]`` for item scalar rows or ``[G, S]`` for
    item/state rows.

    Layouts:
      0: shared scalar, tensor [1]
      1: shared state vector, tensor [S]
      2: item scalar, tensor [G] addressed by item_idx[parent]
      3: item state, tensor [G, S] addressed by item_idx[parent], s
    """

    def _normalize(param):
        if not torch.is_tensor(param):
            return _device_scalar_param(param, device=device, dtype=dtype), 0
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        try:
            layout = dts_backward_param_layout(
                param,
                S=S,
                item_indexed=item_idx is not None,
            )
        except ValueError as exc:
            raise ValueError(
                "DTS parameters must be scalar, [S], [G], [G, 1], or [G, S] "
                "for the fused DTS backward path"
            ) from exc
        layout_code = int(layout.code)
        if layout_code == 0:
            return param.reshape(1).contiguous(), 0
        if layout_code == 1:
            return param.contiguous(), 1
        if layout_code == 2:
            if param.ndim == 2:
                return param.reshape(int(param.shape[0])).contiguous(), 2
            return param.contiguous(), 2
        if layout_code == 3:
            return param.contiguous(), 3
        raise AssertionError("validated DTS backward layout reached unreachable branch")

    pD, layout_D = _normalize(log_pD)
    pS, layout_S = _normalize(log_pS)
    if layout_D != layout_S:
        raise ValueError("log_pD/log_pS must use the same DTS parameter layout")
    return pD, pS, layout_D


def _dts_grad_layout(grad, *, item_idx, S):
    """Return gradient addressing layout matching _dts_layout_param_args."""
    try:
        return int(
            dts_backward_param_layout(
                grad,
                S=S,
                item_indexed=item_idx is not None,
            ).code
        )
    except ValueError as exc:
        raise ValueError("unsupported DTS gradient layout") from exc


def _uniform_backward_const_layout(const_tensor, item_idx, item_indexed):
    """Return addressing mode for self-loop constants.

    Modes:
      0: shared [S]
      1: row-expanded [W, S]
      2: item-indexed [G, S] addressed through item_idx[C]
    """
    if item_indexed:
        if item_idx is None:
            raise ValueError("item-indexed backward constants require item_idx")
        if const_tensor.ndim != 2:
            raise ValueError("item-indexed backward constants require [G, S] tensors")
        return 2
    if const_tensor.ndim == 2:
        return 1
    return 0


def _uniform_backward_leaf_logp_mode(use_leaf_index, leaf_logp, item_idx, item_indexed):
    """Return addressing mode for leaf log-probabilities in the self-loop."""
    if not use_leaf_index:
        return 0
    if item_indexed:
        if item_idx is None:
            raise ValueError("item-indexed leaf log-probabilities require item_idx")
        if leaf_logp.ndim == 1:
            return 1
        if leaf_logp.ndim == 2:
            if int(leaf_logp.shape[1]) == 1:
                raise ValueError("item-indexed [G, 1] leaf_logp should be expanded to [G, S]")
            return 2
        raise ValueError("item-indexed leaf_logp must have shape [G] or [G, S]")
    if leaf_logp.numel() == 1:
        return 3
    return 0


@triton.jit
def _active_mask_from_rhs_absmax_kernel(
    rhs_ptr,
    active_mask_ptr,
    threshold,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    STRICT_GT: tl.constexpr,
    DTYPE: tl.constexpr,
):
    w = tl.program_id(0)
    row_base = w * stride
    row_max = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        rhs_val = tl.load(rhs_ptr + row_base + s_offs, mask=mask, other=0.0)
        tile_max = tl.max(tl.abs(rhs_val), axis=0)
        row_max = tl.maximum(row_max, tile_max)

    if STRICT_GT:
        active = row_max > threshold
    else:
        active = row_max >= threshold
    lane = tl.arange(0, 1)
    tl.store(active_mask_ptr + w + lane, active)


def active_mask_from_rhs_absmax_fused(rhs, threshold, *, use_pruning=True):
    """Build the row activity mask for backward pruning in one Triton launch.

    This is a private retained-kernel helper, not a public dtype policy.  The
    helper accepts fp32/fp64/bf16 CUDA tensors for standalone mask experiments;
    the public ``Pi_wave_backward`` path still supports only fp32/fp64 and
    rejects bf16 before calling this helper.
    """
    if rhs.ndim != 2:
        raise ValueError("rhs must be a 2D tensor")
    if rhs.device.type != "cuda":
        raise ValueError("active_mask_from_rhs_absmax_fused requires a CUDA tensor")
    if rhs.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise ValueError(
            "active_mask_from_rhs_absmax_fused supports fp32/fp64/bf16 tensors"
        )

    W, S = rhs.shape
    active_mask = torch.empty((W,), device=rhs.device, dtype=torch.bool)
    if W == 0:
        return active_mask

    BLOCK_S = min(256, triton.next_power_of_2(S))
    _active_mask_from_rhs_absmax_kernel[(W,)](
        rhs,
        active_mask,
        float(threshold),
        S,
        rhs.stride(0),
        BLOCK_S,
        STRICT_GT=bool(not use_pruning),
        DTYPE=_tl_float_dtype(rhs.dtype),
    )
    return active_mask

@triton.jit
def _wave_backward_uniform_2d_precompute_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    Pibar_row_max_ptr,
    dts_r_ptr,
    has_splits: tl.constexpr,
    rhs_ptr,
    active_mask_ptr,
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    col_log_probs_ptr,
    node_child1_ptr, node_child2_ptr, node_parent_ptr,
    leaf_term_ptr,
    leaf_state_ptr,
    leaf_logp_ptr,
    item_idx_ptr,
    v_k_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    sl2_ptr,
    jump_ptr,
    ws,
    W,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    K_ROUNDS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_SCRATCH_ZERO: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
):
    """Precompute self-loop J^T coefficients for a block of rows and all state."""
    NEG_LARGE: tl.constexpr = -float("inf")

    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    state_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = state_valid[:, None] & row_mask[None, :]
    if SKIP_INACTIVE_SCRATCH_ZERO:
        store_mask = mask
    else:
        store_mask = state_valid[:, None] & row_valid[None, :]

    row_global = ws + rows
    pi_offsets = row_global[None, :] * stride + s_offs[:, None]
    out_offsets = rows[None, :] * S + s_offs[:, None]

    row_max = tl.load(Pibar_row_max_ptr + row_global, mask=row_valid, other=NEG_LARGE).to(DTYPE)
    pi_w = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    pibar_w = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    if USE_COL_WEIGHTS:
        col_logp = tl.load(col_log_probs_ptr + s_offs, mask=state_valid, other=NEG_LARGE).to(DTYPE)
        p_prime = tl.exp2(col_logp[:, None] + pi_w - row_max_safe[None, :])
    else:
        p_prime = tl.exp2(pi_w - row_max_safe[None, :])
    row_sum = tl.sum(tl.where(mask, p_prime, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)), axis=0)

    item = tl.full([BLOCK_W], value=0, dtype=tl.int64)
    const_base = tl.zeros([BLOCK_W], dtype=tl.int64)
    if CONST_LAYOUT == 1:
        const_offsets = out_offsets
    elif CONST_LAYOUT == 2:
        item = tl.load(item_idx_ptr + row_global, mask=row_valid, other=0).to(tl.int64)
        const_base = item * stride
        const_offsets = const_base[None, :] + s_offs[:, None]
    else:
        const_offsets = s_offs[:, None]

    if CONST_LAYOUT == 0:
        const_mask = state_valid[:, None]
    else:
        const_mask = mask
    dl_c = tl.load(DL_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    e_val = tl.load(E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl1_c = tl.load(SL1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl2_c = tl.load(SL2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)

    c1 = tl.load(node_child1_ptr + s_offs, mask=state_valid, other=0)
    c2 = tl.load(node_child2_ptr + s_offs, mask=state_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(state_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)
    pi_s2 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(state_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)

    t0 = dl_c + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_c + pi_s1
    t4 = sl2_c + pi_s2
    if USE_LEAF_INDEX:
        leaf_state = tl.load(leaf_state_ptr + row_global, mask=row_valid, other=-1)
        leaf_hit = mask & (leaf_state[None, :] == s_offs[:, None])
        if LEAF_LOGP_MODE == 3:
            leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + item, mask=row_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            leaf_logp = tl.load(
                leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                mask=leaf_hit,
                other=NEG_LARGE,
            ).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=state_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        t5 = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    else:
        t5 = tl.full([BLOCK_S, BLOCK_W], value=NEG_LARGE, dtype=DTYPE)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m != NEG_LARGE, m, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(dts_l_sum > 0.0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

    if has_splits:
        dts_r = tl.load(dts_r_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m != NEG_LARGE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l != NEG_LARGE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
    else:
        w_L = tl.full([BLOCK_S, BLOCK_W], value=1.0, dtype=DTYPE)

    # Ancestor-or-self path sums of p_prime via in-register binary lifting (p_prime is
    # exactly exp2(colw + pi - row_max), the term the old per-column ancestor chase
    # accumulated). BLOCK_W is 1, so the [BLOCK_S, 1] tile reshapes to a flat row.
    tl.static_assert(BLOCK_W == 1)
    val = tl.reshape(p_prime, [BLOCK_S])
    s_flat = tl.arange(0, BLOCK_S)
    flat_valid = s_flat < S
    for _k in tl.static_range(K_ROUNDS):
        jmp = tl.load(jump_ptr + _k * S + s_flat, mask=flat_valid, other=-1)
        jv = jmp >= 0
        g = tl.gather(val, tl.where(jv, jmp, 0), axis=0)
        val += tl.where(jv, g, 0.0)
    ancestor_sum = tl.reshape(val, [BLOCK_S, BLOCK_W])
    denom = row_sum[None, :] - ancestor_sum
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, tl.zeros_like(denom))

    diag_wt = w_L * (e0 + e1) * inv_sum
    pibar_u_coeff = w_L * e2 * inv_sum * inv_denom
    sl1_wt = w_L * e3 * inv_sum
    sl2_wt = w_L * e4 * inv_sum

    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    rhs_val = tl.load(rhs_ptr + out_offsets, mask=mask, other=0.0).to(DTYPE)
    tl.store(v_k_ptr + out_offsets, tl.where(mask, rhs_val, zero), mask=store_mask)
    tl.store(diag_ptr + out_offsets, tl.where(mask, diag_wt, zero), mask=store_mask)
    tl.store(pibar_coeff_ptr + out_offsets, tl.where(mask, pibar_u_coeff, zero), mask=store_mask)
    tl.store(p_prime_ptr + out_offsets, tl.where(mask, p_prime, zero), mask=store_mask)
    if USE_CHILD_EDGE_SELF_LOOP:
        child1_offsets = rows[None, :] * S + c1[:, None]
        child2_offsets = rows[None, :] * S + c2[:, None]
        child1_mask = (state_valid & c1_valid)[:, None] & row_mask[None, :]
        child2_mask = (state_valid & c2_valid)[:, None] & row_mask[None, :]
        tl.store(sl1_ptr + child1_offsets, sl1_wt, mask=child1_mask)
        tl.store(sl1_ptr + child2_offsets, sl2_wt, mask=child2_mask)
    else:
        tl.store(sl1_ptr + out_offsets, tl.where(mask, sl1_wt, zero), mask=store_mask)
        tl.store(sl2_ptr + out_offsets, tl.where(mask, sl2_wt, zero), mask=store_mask)


@triton.jit
def _wave_backward_uniform_2d_jt_kernel(
    term_in_ptr,
    term_out_ptr,
    rhs_update_ptr,
    active_mask_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    sl2_ptr,
    node_child1_ptr,
    node_child2_ptr,
    node_parent_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    pibar_corr_ptr,
    v_k_ptr,
    W,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_SCRATCH_ZERO: tl.constexpr,
    FIXED_POINT_UPDATE: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
    OUTPUT_A: tl.constexpr,
    ACCUMULATE_V: tl.constexpr,
):
    """Apply one self-loop J^T term using in-program bottom-up tree reduction."""
    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    state_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = state_valid[:, None] & row_mask[None, :]
    if SKIP_INACTIVE_SCRATCH_ZERO:
        store_mask = mask
    else:
        store_mask = state_valid[:, None] & row_valid[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]

    term_val = tl.load(term_in_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    pibar_u_coeff = tl.load(pibar_coeff_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    u_d = term_val * pibar_u_coeff
    A = tl.sum(tl.where(mask, u_d, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)), axis=0)
    tl.store(pibar_corr_ptr + offsets, tl.where(mask, u_d, tl.zeros_like(u_d)), mask=store_mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=0)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            reduce_mask = node_mask[:, None] & row_mask[None, :]
            row_base = rows[None, :] * S
            parent_val = tl.load(
                pibar_corr_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ).to(DTYPE)
            c1_val = tl.load(
                pibar_corr_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            c2_val = tl.load(
                pibar_corr_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            tl.store(
                pibar_corr_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    tl.debug_barrier()

    corr = tl.load(pibar_corr_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    diag_wt = tl.load(diag_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    p_prime = tl.load(p_prime_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    base = term_val * diag_wt + p_prime * (A[None, :] - corr)

    if USE_CHILD_EDGE_SELF_LOOP:
        parent = tl.load(node_parent_ptr + s_offs, mask=state_valid, other=-1)
        parent_valid = state_valid & (parent >= 0) & (parent < S)
        row_base = rows[None, :] * S
        parent_mask = parent_valid[:, None] & row_mask[None, :]
        parent_term = tl.load(
            term_in_ptr + row_base + parent[:, None],
            mask=parent_mask,
            other=0.0,
        ).to(DTYPE)
        edge_wt = tl.load(sl1_ptr + offsets, mask=parent_mask, other=0.0).to(DTYPE)
        result = base + parent_term * edge_wt
    else:
        tl.store(term_out_ptr + offsets, tl.where(mask, base, tl.zeros_like(base)), mask=store_mask)

        tl.debug_barrier()

        c1 = tl.load(node_child1_ptr + s_offs, mask=state_valid, other=S)
        c2 = tl.load(node_child2_ptr + s_offs, mask=state_valid, other=S)
        sl1_wt = tl.load(sl1_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        sl2_wt = tl.load(sl2_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        row_base = rows[None, :] * S
        c1_mask = (state_valid & (c1 < S))[:, None] & row_mask[None, :]
        c2_mask = (state_valid & (c2 < S))[:, None] & row_mask[None, :]
        c1_cur = tl.load(term_out_ptr + row_base + c1[:, None], mask=c1_mask, other=0.0).to(DTYPE)
        c2_cur = tl.load(term_out_ptr + row_base + c2[:, None], mask=c2_mask, other=0.0).to(DTYPE)
        tl.store(term_out_ptr + row_base + c1[:, None], c1_cur + term_val * sl1_wt, mask=c1_mask)
        tl.store(term_out_ptr + row_base + c2[:, None], c2_cur + term_val * sl2_wt, mask=c2_mask)

        tl.debug_barrier()

        result = tl.load(term_out_ptr + offsets, mask=mask, other=0.0).to(DTYPE)

    out_val = term_val - result if OUTPUT_A else result
    tl.store(
        term_out_ptr + offsets,
        tl.where(mask, out_val, tl.zeros_like(out_val)),
        mask=store_mask,
    )

    if FIXED_POINT_UPDATE:
        rhs_val = tl.load(rhs_update_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        tl.store(
            v_k_ptr + offsets,
            tl.where(mask, rhs_val + result, tl.zeros_like(result)),
            mask=store_mask,
        )
    elif ACCUMULATE_V:
        v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        tl.store(v_k_ptr + offsets, v_prev + result, mask=mask)


@triton.jit
def _wave_backward_jt_neumann_reg_kernel(
    rhs_ptr,
    v_k_ptr,
    active_mask_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    node_parent_ptr,
    dfs_node_ptr,
    start_m1_ptr,
    end_m1_ptr,
    term_scratch_ptr,
    neumann_terms,
    W,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    STORE_LAST_TERM: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Register-resident Neumann loop (one row per program, BLOCK_S = whole state row).

    The per-term subtree reduction runs as cumsum over DFS order + interval difference
    (tl.gather/tl.cumsum stay in registers/SMEM): no scratch buffers, no barriers.
    """
    row = tl.program_id(0)
    if USE_ACTIVE_MASK:
        if tl.load(active_mask_ptr + row) == 0:
            return

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    base = row * S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    pibar_u_coeff = tl.load(pibar_coeff_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
    diag_wt = tl.load(diag_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
    p_prime = tl.load(p_prime_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
    parent = tl.load(node_parent_ptr + s_offs, mask=mask, other=-1)
    pvalid = mask & (parent >= 0) & (parent < S)
    edge_wt = tl.load(sl1_ptr + base + s_offs, mask=pvalid, other=0.0).to(DTYPE)
    dfs_node = tl.load(dfs_node_ptr + s_offs, mask=mask, other=0)
    start_m1 = tl.load(start_m1_ptr + s_offs, mask=mask, other=-1)
    end_m1 = tl.load(end_m1_ptr + s_offs, mask=mask, other=0)
    parent_safe = tl.where(pvalid, parent, 0)
    start_safe = tl.where(start_m1 >= 0, start_m1, 0)
    end_safe = tl.where(end_m1 >= 0, end_m1, 0)

    term = tl.load(rhs_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
    v_acc = term

    for _n in range(0, neumann_terms):
        u_d = tl.where(mask, term * pibar_u_coeff, zero)
        A = tl.sum(u_d, axis=0)
        u_dfs = tl.where(mask, tl.gather(u_d, dfs_node, axis=0), zero)
        cum = tl.cumsum(u_dfs, axis=0)
        ce = tl.gather(cum, end_safe, axis=0)
        cs = tl.where(start_m1 >= 0, tl.gather(cum, start_safe, axis=0), zero)
        corr = ce - cs
        base_val = term * diag_wt + p_prime * (A - corr)
        parent_term = tl.where(pvalid, tl.gather(term, parent_safe, axis=0), zero)
        result = base_val + parent_term * edge_wt
        v_acc += result
        term = result

    tl.store(v_k_ptr + base + s_offs, tl.where(mask, v_acc, zero), mask=mask)
    if STORE_LAST_TERM:
        tl.store(term_scratch_ptr + base + s_offs, tl.where(mask, term, zero), mask=mask)


@triton.jit
def _wave_backward_uniform_2d_jt_neumann_fused_kernel(
    rhs_ptr,
    v_k_ptr,
    active_mask_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    node_parent_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    pibar_corr_ptr,
    term_scratch_ptr,
    neumann_terms,
    W,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    STORE_LAST_TERM: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """All Neumann self-loop terms in one launch (rows are independent).

    Replays the exact per-term op sequence of `_wave_backward_uniform_2d_jt_kernel`
    (USE_CHILD_EDGE_SELF_LOOP=True, ACCUMULATE_V semantics) in an in-program loop so
    the per-row coefficient arrays load once instead of once per term.
    """
    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    state_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = state_valid[:, None] & row_mask[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]
    row_base = rows[None, :] * S
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)

    pibar_u_coeff = tl.load(pibar_coeff_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    diag_wt = tl.load(diag_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    p_prime = tl.load(p_prime_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    parent = tl.load(node_parent_ptr + s_offs, mask=state_valid, other=-1)
    parent_valid = state_valid & (parent >= 0) & (parent < S)
    parent_mask = parent_valid[:, None] & row_mask[None, :]
    edge_wt = tl.load(sl1_ptr + offsets, mask=parent_mask, other=0.0).to(DTYPE)

    term_val = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    v_acc = term_val

    for _n in range(0, neumann_terms):
        u_d = term_val * pibar_u_coeff
        A = tl.sum(tl.where(mask, u_d, zero), axis=0)
        tl.store(pibar_corr_ptr + offsets, tl.where(mask, u_d, zero), mask=mask)
        tl.store(term_scratch_ptr + offsets, tl.where(mask, term_val, zero), mask=mask)

        tl.debug_barrier()

        for level in range(0, N_LEVELS):
            level_start = tl.load(compact_level_ptr + level)
            level_end = tl.load(compact_level_ptr + level + 1)
            node_start = level_start
            while node_start < level_end:
                node_offs = node_start + tl.arange(0, BLOCK_NODES)
                node_mask = node_offs < level_end
                node_parent_c = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=0)
                c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
                c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
                reduce_mask = node_mask[:, None] & row_mask[None, :]
                parent_val = tl.load(
                    pibar_corr_ptr + row_base + node_parent_c[:, None],
                    mask=reduce_mask,
                    other=0.0,
                ).to(DTYPE)
                c1_val = tl.load(
                    pibar_corr_ptr + row_base + c1[:, None],
                    mask=reduce_mask & (c1 < S)[:, None],
                    other=0.0,
                ).to(DTYPE)
                c2_val = tl.load(
                    pibar_corr_ptr + row_base + c2[:, None],
                    mask=reduce_mask & (c2 < S)[:, None],
                    other=0.0,
                ).to(DTYPE)
                tl.store(
                    pibar_corr_ptr + row_base + node_parent_c[:, None],
                    parent_val + c1_val + c2_val,
                    mask=reduce_mask,
                )
                node_start += BLOCK_NODES
            tl.debug_barrier()

        tl.debug_barrier()

        corr = tl.load(pibar_corr_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        base = term_val * diag_wt + p_prime * (A[None, :] - corr)
        parent_term = tl.load(
            term_scratch_ptr + row_base + parent[:, None],
            mask=parent_mask,
            other=0.0,
        ).to(DTYPE)
        result = base + parent_term * edge_wt
        v_acc += result
        term_val = result

    tl.store(v_k_ptr + offsets, tl.where(mask, v_acc, zero), mask=mask)
    if STORE_LAST_TERM:
        tl.store(term_scratch_ptr + offsets, tl.where(mask, term_val, zero), mask=mask)


@torch.no_grad()
def _gmres_solve_wave_self_loop(
    apply_a,
    rhs: torch.Tensor,
    *,
    max_iter: int,
) -> torch.Tensor:
    """Solve ``A v = rhs`` for one wave with fixed-iteration unrestarted GMRES."""
    max_iter = int(max_iter)
    if max_iter < 1:
        return torch.zeros_like(rhs)

    b_norm_t = torch.linalg.vector_norm(rhs)
    if float(b_norm_t.detach().cpu()) == 0.0:
        return torch.zeros_like(rhs)

    return _gmres_solve_wave_self_loop_fixed_cgs2(
        apply_a,
        rhs,
        max_iter=max_iter,
        b_norm_t=b_norm_t,
    )


def _gmres_solve_wave_self_loop_fixed_cgs2(
    apply_a,
    rhs: torch.Tensor,
    *,
    max_iter: int,
    b_norm_t: torch.Tensor,
) -> torch.Tensor:
    """Fixed-m GMRES Arnoldi using batched CGS with one reorthogonalization."""
    basis = torch.empty(
        (max_iter + 1, *rhs.shape),
        dtype=rhs.dtype,
        device=rhs.device,
    )
    basis_2d = basis.reshape(max_iter + 1, -1)
    basis_2d[0].copy_(rhs.reshape(-1) / b_norm_t)
    hessenberg = torch.zeros(
        (max_iter + 1, max_iter),
        dtype=rhs.dtype,
        device=rhs.device,
    )
    e1 = torch.zeros((max_iter + 1,), dtype=rhs.dtype, device=rhs.device)
    e1[0] = b_norm_t
    coeff_buf = torch.empty((max_iter,), dtype=rhs.dtype, device=rhs.device)
    coeff2_buf = torch.empty((max_iter,), dtype=rhs.dtype, device=rhs.device)
    work = torch.empty_like(rhs).reshape(-1)
    work2 = torch.empty_like(rhs).reshape(-1)

    effective_iter = max_iter
    breakdown_tol = torch.finfo(rhs.dtype).eps * torch.clamp(b_norm_t, min=1.0)
    for j in range(max_iter):
        w = apply_a(basis[j]).reshape(-1)
        q = basis_2d[: j + 1]
        coeff = coeff_buf[: j + 1]
        torch.mv(q, w, out=coeff)
        hessenberg[: j + 1, j].copy_(coeff)
        torch.addmv(w, q.t(), coeff, beta=1.0, alpha=-1.0, out=work)

        coeff2 = coeff2_buf[: j + 1]
        torch.mv(q, work, out=coeff2)
        hessenberg[: j + 1, j].add_(coeff2)
        torch.addmv(work, q.t(), coeff2, beta=1.0, alpha=-1.0, out=work2)

        next_norm_t = torch.linalg.vector_norm(work2)
        hessenberg[j + 1, j] = next_norm_t
        if bool((next_norm_t <= breakdown_tol).detach().cpu()):
            effective_iter = j + 1
            break
        if j + 1 < max_iter:
            denom = torch.clamp(next_norm_t, min=torch.finfo(rhs.dtype).tiny)
            torch.div(work2, denom, out=basis_2d[j + 1])

    h_sub = hessenberg[: effective_iter + 1, :effective_iter]
    rhs_sub = e1[: effective_iter + 1]
    y = torch.linalg.lstsq(h_sub, rhs_sub).solution
    out = torch.empty_like(rhs)
    torch.mv(basis_2d[:effective_iter].t(), y, out=out.reshape(-1))
    return out


@triton.jit
def _col_grad_from_pibar_self_loop_kernel(
    v_k_ptr,
    active_mask_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    pibar_corr_ptr,
    grad_col_log_probs_ptr,
    W,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    state_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = state_valid[:, None] & row_mask[None, :]
    store_mask = state_valid[:, None] & row_valid[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]

    term_val = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    pibar_u_coeff = tl.load(pibar_coeff_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    u_d = term_val * pibar_u_coeff
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    A = tl.sum(tl.where(mask, u_d, zero), axis=0)
    tl.store(pibar_corr_ptr + offsets, tl.where(mask, u_d, zero), mask=store_mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=0)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            reduce_mask = node_mask[:, None] & row_mask[None, :]
            row_base = rows[None, :] * S
            parent_val = tl.load(
                pibar_corr_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ).to(DTYPE)
            c1_val = tl.load(
                pibar_corr_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            c2_val = tl.load(
                pibar_corr_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            tl.store(
                pibar_corr_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    corr = tl.load(pibar_corr_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    p_prime = tl.load(p_prime_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    contrib = p_prime * (A[None, :] - corr)
    state_contrib = tl.sum(tl.where(mask, contrib, zero), axis=1)
    tl.atomic_add(
        grad_col_log_probs_ptr + s_offs,
        state_contrib,
        sem="relaxed",
        mask=state_valid,
    )


@triton.jit
def _col_grad_from_pibar_self_loop_reg_kernel(
    v_k_ptr,
    active_mask_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    dfs_node_ptr,
    start_m1_ptr,
    end_m1_ptr,
    grad_col_log_probs_ptr,
    W,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Register-resident column-gradient kernel (subtree sums via DFS cumsum)."""
    row = tl.program_id(0)
    if USE_ACTIVE_MASK:
        if tl.load(active_mask_ptr + row) == 0:
            return
    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    base = row * S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    term = tl.load(v_k_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
    coeff = tl.load(pibar_coeff_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
    p_prime = tl.load(p_prime_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
    dfs_node = tl.load(dfs_node_ptr + s_offs, mask=mask, other=0)
    start_m1 = tl.load(start_m1_ptr + s_offs, mask=mask, other=-1)
    end_m1 = tl.load(end_m1_ptr + s_offs, mask=mask, other=0)

    u_d = tl.where(mask, term * coeff, zero)
    A = tl.sum(u_d, axis=0)
    u_dfs = tl.where(mask, tl.gather(u_d, dfs_node, axis=0), zero)
    cum = tl.cumsum(u_dfs, axis=0)
    ce = tl.gather(cum, tl.where(end_m1 >= 0, end_m1, 0), axis=0)
    cs = tl.where(start_m1 >= 0, tl.gather(cum, tl.where(start_m1 >= 0, start_m1, 0), axis=0), zero)
    corr = ce - cs
    contrib = tl.where(mask, p_prime * (A - corr), zero)
    tl.atomic_add(grad_col_log_probs_ptr + s_offs, contrib, sem="relaxed", mask=mask)


@triton.jit
def _wave_backward_uniform_param_store_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    dts_r_ptr,
    has_splits: tl.constexpr,
    v_k_ptr,
    active_mask_ptr,
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    node_child1_ptr, node_child2_ptr,
    leaf_term_ptr,
    leaf_state_ptr,
    leaf_logp_ptr,
    item_idx_ptr,
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_E_ptr,
    grad_Ebar_ptr,
    grad_E_s1_ptr,
    grad_E_s2_ptr,
    grad_mt_ptr,
    aw0_ptr,
    aw1_ptr,
    aw2_ptr,
    aw345_ptr,
    aw3_ptr,
    aw4_ptr,
    ws,
    W,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    ACCUM_GRADS: tl.constexpr,
    PARAM_GRAD_VECTOR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Store per-element self-loop parameter VJP contributions after Neumann."""
    NEG_LARGE: tl.constexpr = -float("inf")

    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    state_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = state_valid[:, None] & row_mask[None, :]
    store_mask = state_valid[:, None] & row_valid[None, :]
    row_global = ws + rows
    pi_offsets = row_global[None, :] * stride + s_offs[:, None]
    out_offsets = rows[None, :] * S + s_offs[:, None]

    item = tl.full([BLOCK_W], value=0, dtype=tl.int64)
    const_base = tl.zeros([BLOCK_W], dtype=tl.int64)
    if CONST_LAYOUT == 1:
        const_offsets = out_offsets
    elif CONST_LAYOUT == 2:
        item = tl.load(item_idx_ptr + row_global, mask=row_valid, other=0).to(tl.int64)
        const_base = item * stride
        const_offsets = const_base[None, :] + s_offs[:, None]
    else:
        const_offsets = s_offs[:, None]

    const_mask = state_valid[:, None] if CONST_LAYOUT == 0 else mask
    pi_w = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    pibar_w = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    v_k_val = tl.load(v_k_ptr + out_offsets, mask=mask, other=0.0).to(DTYPE)
    dl_c = tl.load(DL_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    e_val = tl.load(E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl1_c = tl.load(SL1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl2_c = tl.load(SL2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)

    c1 = tl.load(node_child1_ptr + s_offs, mask=state_valid, other=0)
    c2 = tl.load(node_child2_ptr + s_offs, mask=state_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(state_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)
    pi_s2 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(state_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)

    t0 = dl_c + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_c + pi_s1
    t4 = sl2_c + pi_s2
    if USE_LEAF_INDEX:
        leaf_state = tl.load(leaf_state_ptr + row_global, mask=row_valid, other=-1)
        leaf_hit = mask & (leaf_state[None, :] == s_offs[:, None])
        if LEAF_LOGP_MODE == 3:
            leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + item, mask=row_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            leaf_logp = tl.load(
                leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                mask=leaf_hit,
                other=NEG_LARGE,
            ).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=state_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        t5 = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    else:
        t5 = tl.full([BLOCK_S, BLOCK_W], value=NEG_LARGE, dtype=DTYPE)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m != NEG_LARGE, m, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(dts_l_sum > 0.0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

    if has_splits:
        dts_r = tl.load(dts_r_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m != NEG_LARGE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l != NEG_LARGE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
    else:
        w_L = tl.full([BLOCK_S, BLOCK_W], value=1.0, dtype=DTYPE)

    alpha = v_k_val * w_L
    _aw0 = alpha * e0 * inv_sum
    _aw1 = alpha * e1 * inv_sum
    _aw2 = alpha * e2 * inv_sum
    _aw3 = alpha * e3 * inv_sum
    _aw4 = alpha * e4 * inv_sum
    _aw5 = alpha * e5 * inv_sum
    _aw345 = _aw3 + _aw4 + _aw5
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    if ACCUM_GRADS:
        aw0_s = tl.sum(tl.where(mask, _aw0, zero), axis=1)
        aw1_s = tl.sum(tl.where(mask, _aw1, zero), axis=1)
        aw2_s = tl.sum(tl.where(mask, _aw2, zero), axis=1)
        aw345_s = tl.sum(tl.where(mask, _aw345, zero), axis=1)
        aw3_s = tl.sum(tl.where(mask, _aw3, zero), axis=1)
        aw4_s = tl.sum(tl.where(mask, _aw4, zero), axis=1)
        if PARAM_GRAD_VECTOR:
            tl.atomic_add(
                grad_log_pD_ptr + s_offs,
                aw0_s,
                sem="relaxed",
                mask=state_valid,
            )
            tl.atomic_add(
                grad_log_pS_ptr + s_offs,
                aw345_s,
                sem="relaxed",
                mask=state_valid,
            )
        else:
            tl.atomic_add(grad_log_pD_ptr, tl.sum(aw0_s, axis=0), sem="relaxed")
            tl.atomic_add(grad_log_pS_ptr, tl.sum(aw345_s, axis=0), sem="relaxed")
        tl.atomic_add(
            grad_E_ptr + s_offs,
            aw0_s + aw2_s,
            sem="relaxed",
            mask=state_valid,
        )
        tl.atomic_add(
            grad_Ebar_ptr + s_offs,
            aw1_s,
            sem="relaxed",
            mask=state_valid,
        )
        tl.atomic_add(
            grad_E_s1_ptr + s_offs,
            aw4_s,
            sem="relaxed",
            mask=state_valid,
        )
        tl.atomic_add(
            grad_E_s2_ptr + s_offs,
            aw3_s,
            sem="relaxed",
            mask=state_valid,
        )
        tl.atomic_add(
            grad_mt_ptr + s_offs,
            aw2_s,
            sem="relaxed",
            mask=state_valid,
        )
    else:
        tl.store(aw0_ptr + out_offsets, tl.where(mask, _aw0, zero), mask=store_mask)
        tl.store(aw1_ptr + out_offsets, tl.where(mask, _aw1, zero), mask=store_mask)
        tl.store(aw2_ptr + out_offsets, tl.where(mask, _aw2, zero), mask=store_mask)
        tl.store(aw345_ptr + out_offsets, tl.where(mask, _aw345, zero), mask=store_mask)
        tl.store(aw3_ptr + out_offsets, tl.where(mask, _aw3, zero), mask=store_mask)
        tl.store(aw4_ptr + out_offsets, tl.where(mask, _aw4, zero), mask=store_mask)


def _wave_backward_uniform_2d(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    col_log_probs,
    node_child1, node_child2, leaf_term_wt,
    *,
    neumann_terms,
    leaf_state_idx,
    leaf_logp,
    has_leaf_term,
    active_mask,
    node_parent,
    max_ancestor_depth,
    pibar_row_max,
    item_idx,
    const_layout,
    leaf_logp_mode,
    use_leaf_index,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
    grad_col_log_probs=None,
    use_col_weights=True,
    self_loop_grad_targets=None,
    initial_v=None,
    self_loop_solver="neumann",
    return_last_increment=False,
):
    """Retained 2D row-block/full-state tree-reduction self-loop."""
    if Pi_star.device.type != "cuda":
        raise RuntimeError("GPUREC self-loop 2D fast path requires CUDA tensors")
    if Pi_star.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "2D self-loop fast path currently supports fp32/fp64 only",
        )
    ok, required_bytes, budget_bytes = proposal0_memory_gate(
        W,
        S,
        Pi_star.dtype,
        device=Pi_star.device,
    )
    if not ok:
        raise RuntimeError(
            "2D self-loop fast path estimated scratch "
            f"{required_bytes / (1024 ** 3):.2f} GiB above memory budget "
            f"{(budget_bytes or 0) / (1024 ** 3):.2f} GiB",
        )
    if const_layout not in (0, 1, 2):
        raise RuntimeError("unsupported self-loop constant layout")
    if use_leaf_index and leaf_logp_mode not in (0, 1, 2, 3):
        raise RuntimeError("unsupported leaf log-probability layout")

    device = Pi_star.device
    dtype = Pi_star.dtype
    col_log_probs = col_log_probs.to(device=device, dtype=dtype).contiguous()
    if node_parent is None:
        raise ValueError("node_parent is required for the retained 2D self-loop path")
    node_parent = node_parent.to(device=device, dtype=torch.int32).contiguous()
    if max_ancestor_depth is None:
        raise ValueError("max_ancestor_depth is required for the retained 2D self-loop path")
    max_ancestor_depth = max(1, int(max_ancestor_depth))

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact state levels are required for the retained 2D self-loop path")
    compact_level_ptr = compact_level_ptr.to(device=device, dtype=torch.long).contiguous()
    compact_level_parents = compact_level_parents.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child1 = compact_level_child1.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child2 = compact_level_child2.to(device=device, dtype=torch.int32).contiguous()

    block_w = 1
    block_s = triton.next_power_of_2(S)
    block_nodes = 128
    n_row_blocks = triton.cdiv(W, block_w)
    scratch_shape = (W, S)
    jump_table, k_rounds = _get_jumps(node_parent, S)

    v_k = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw0 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw1 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw2 = torch.empty(scratch_shape, device=device, dtype=dtype)
    accum_self_loop_grads = self_loop_grad_targets is not None
    aw345 = None if accum_self_loop_grads else torch.empty(scratch_shape, device=device, dtype=dtype)
    aw3 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw4 = torch.empty(scratch_shape, device=device, dtype=dtype)
    spec_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    term_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    pibar_corr = torch.empty(scratch_shape, device=device, dtype=dtype)

    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained 2D self-loop path")
    pibar_row_max = pibar_row_max.to(device=device, dtype=dtype).contiguous()
    skip_inactive_scratch_zero = True
    if item_idx is not None:
        item_idx = item_idx.to(device=device, dtype=torch.long).contiguous()
    else:
        item_idx = node_parent
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = bool(use_leaf_index and requested_has_leaf_term)
    has_materialized_leaf_term = leaf_term_wt is not None
    if leaf_term_wt is None:
        leaf_term_wt = leaf_logp if use_leaf_index else Pi_star
    has_leaf_term = bool(
        requested_has_leaf_term
        and (use_leaf_index or has_materialized_leaf_term)
    )
    leaf_state_arg = leaf_state_idx if use_leaf_index else node_child1
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt
    use_child_edge_self_loop = True

    launch_options = {"num_warps": 8}

    _wave_backward_uniform_2d_precompute_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        pibar_row_max,
        dts_r if dts_r is not None else Pi_star,
        dts_r is not None,
        rhs,
        active_mask if active_mask is not None else rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        col_log_probs,
        node_child1,
        node_child2,
        node_parent,
        leaf_term_wt,
        leaf_state_arg,
        leaf_logp_arg,
        item_idx,
        v_k,
        aw0,
        aw1,
        aw2,
        aw3,
        aw4,
        jump_table,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        K_ROUNDS=k_rounds,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
        CONST_LAYOUT=int(const_layout),
        DTYPE=_tl_float_dtype(dtype),
        USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
        USE_COL_WEIGHTS=bool(use_col_weights),
        **launch_options,
    )

    self_loop_solver = str(self_loop_solver).strip().lower()
    jt_options = {"num_warps": 2}
    if self_loop_solver == "gmres":
        if initial_v is not None:
            raise ValueError("GMRES self-loop solve does not support initial_v")
        gmres_a_buf = torch.empty_like(v_k)
        gmres_rhs = rhs
        gmres_active_mask = active_mask
        if active_mask is not None:
            gmres_active_mask = active_mask.to(device=device, dtype=torch.bool).contiguous()
            gmres_rhs = rhs * gmres_active_mask[:, None].to(dtype=dtype)

        def _apply_a(term_in: torch.Tensor) -> torch.Tensor:
            _wave_backward_uniform_2d_jt_kernel[(n_row_blocks,)](
                term_in,
                gmres_a_buf,
                rhs,
                gmres_active_mask if gmres_active_mask is not None else rhs,
                aw0,
                aw1,
                aw2,
                aw3,
                aw4,
                node_child1,
                node_child2,
                node_parent,
                compact_level_ptr,
                compact_level_parents,
                compact_level_child1,
                compact_level_child2,
                pibar_corr,
                v_k,
                W,
                S,
                block_w,
                block_s,
                block_nodes,
                compact_level_ptr.numel() - 1,
                USE_ACTIVE_MASK=bool(gmres_active_mask is not None),
                SKIP_INACTIVE_SCRATCH_ZERO=False,
                FIXED_POINT_UPDATE=False,
                DTYPE=_tl_float_dtype(dtype),
                USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
                OUTPUT_A=True,
                ACCUMULATE_V=False,
                **jt_options,
            )
            return gmres_a_buf

        v_k.copy_(
            _gmres_solve_wave_self_loop(
                _apply_a,
                gmres_rhs,
                max_iter=int(neumann_terms),
            )
        )
    elif self_loop_solver == "neumann" and initial_v is not None:
        if tuple(initial_v.shape) != scratch_shape:
            raise ValueError(
                f"initial_v shape {tuple(initial_v.shape)} does not match "
                f"wave scratch shape {scratch_shape}"
            )
        v_k.copy_(initial_v.to(device=device, dtype=dtype).contiguous())
        for _n in range(int(neumann_terms)):
            _wave_backward_uniform_2d_jt_kernel[(n_row_blocks,)](
                v_k,
                spec_buf,
                rhs,
                active_mask if active_mask is not None else rhs,
                aw0,
                aw1,
                aw2,
                aw3,
                aw4,
                node_child1,
                node_child2,
                node_parent,
                compact_level_ptr,
                compact_level_parents,
                compact_level_child1,
                compact_level_child2,
                pibar_corr,
                v_k,
                W,
                S,
                block_w,
                block_s,
                block_nodes,
                compact_level_ptr.numel() - 1,
                USE_ACTIVE_MASK=bool(active_mask is not None),
                SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
                FIXED_POINT_UPDATE=True,
                DTYPE=_tl_float_dtype(dtype),
                USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
                OUTPUT_A=False,
                ACCUMULATE_V=True,
                **jt_options,
            )
    elif self_loop_solver == "neumann" and _JT_MODE == 1:
        # Register-resident Neumann: all terms in one launch, subtree sums via
        # DFS cumsum + interval difference. No scratch traffic, no barriers.
        dfs_node, start_m1, end_m1 = _get_dfs_tables(node_parent, node_child1, node_child2, S)
        _wave_backward_jt_neumann_reg_kernel[(n_row_blocks,)](
            rhs,
            v_k,
            active_mask if active_mask is not None else rhs,
            aw0,
            aw1,
            aw2,
            aw3,
            node_parent,
            dfs_node,
            start_m1,
            end_m1,
            spec_buf,
            int(neumann_terms),
            W,
            S,
            block_s,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            STORE_LAST_TERM=bool(return_last_increment),
            DTYPE=_tl_float_dtype(dtype),
            num_warps=_JT_REG_WARPS,
        )
    elif self_loop_solver == "neumann":
        # All Neumann terms fused into one launch: rows are independent, so each
        # program iterates its own terms; coefficient arrays load once.
        _wave_backward_uniform_2d_jt_neumann_fused_kernel[(n_row_blocks,)](
            rhs,
            v_k,
            active_mask if active_mask is not None else rhs,
            aw0,
            aw1,
            aw2,
            aw3,
            node_parent,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            pibar_corr,
            spec_buf,
            int(neumann_terms),
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            STORE_LAST_TERM=bool(return_last_increment),
            DTYPE=_tl_float_dtype(dtype),
            num_warps=_JT_WARPS,
        )
    else:
        raise ValueError(f"unsupported self-loop solver {self_loop_solver!r}")



    last_increment_relres = None
    if (
        return_last_increment
        and self_loop_solver == "neumann"
        and initial_v is None
        and int(neumann_terms) > 0
    ):
        last_buf = spec_buf  # fused Neumann kernel stores the last term here
        eps = torch.finfo(torch.float32).tiny
        num = last_buf.float().norm(dim=1)
        den = v_k.float().norm(dim=1).clamp_min(eps)
        relres = num / den


        if active_mask is not None:
            row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            relres = torch.where(row_active, relres, torch.zeros_like(relres))
        last_increment_relres = relres

    if accum_self_loop_grads:
        (
            grad_log_pD_ptr,
            grad_log_pS_ptr,
            grad_E_ptr,
            grad_Ebar_ptr,
            grad_E_s1_ptr,
            grad_E_s2_ptr,
            grad_mt_ptr,
            param_grad_vector,
        ) = self_loop_grad_targets
        aw345_ptr = aw0
    else:
        grad_log_pD_ptr = aw0
        grad_log_pS_ptr = aw0
        grad_E_ptr = aw0
        grad_Ebar_ptr = aw0
        grad_E_s1_ptr = aw0
        grad_E_s2_ptr = aw0
        grad_mt_ptr = aw0
        param_grad_vector = False
        aw345_ptr = aw345

    if grad_col_log_probs is not None and _JT_MODE == 1:
        dfs_node, start_m1, end_m1 = _get_dfs_tables(node_parent, node_child1, node_child2, S)
        _col_grad_from_pibar_self_loop_reg_kernel[(n_row_blocks,)](
            v_k,
            active_mask if active_mask is not None else rhs,
            aw1,
            aw2,
            dfs_node,
            start_m1,
            end_m1,
            grad_col_log_probs,
            W,
            S,
            block_s,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            DTYPE=_tl_float_dtype(dtype),
            num_warps=_JT_REG_WARPS,
        )
    elif grad_col_log_probs is not None:
        _col_grad_from_pibar_self_loop_kernel[(n_row_blocks,)](
            v_k,
            active_mask if active_mask is not None else rhs,
            aw1,
            aw2,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            pibar_corr,
            grad_col_log_probs,
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            DTYPE=_tl_float_dtype(dtype),
            **jt_options,
        )

    _wave_backward_uniform_param_store_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        dts_r if dts_r is not None else Pi_star,
        dts_r is not None,
        v_k,
        active_mask if active_mask is not None else rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        node_child1,
        node_child2,
        leaf_term_wt,
        leaf_state_arg,
        leaf_logp_arg,
        item_idx,
        grad_log_pD_ptr,
        grad_log_pS_ptr,
        grad_E_ptr,
        grad_Ebar_ptr,
        grad_E_s1_ptr,
        grad_E_s2_ptr,
        grad_mt_ptr,
        aw0,
        aw1,
        aw2,
        aw345_ptr,
        aw3,
        aw4,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        CONST_LAYOUT=int(const_layout),
        ACCUM_GRADS=bool(accum_self_loop_grads),
        PARAM_GRAD_VECTOR=bool(param_grad_vector),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if accum_self_loop_grads:
        base = (v_k, None, None, None, None, None, None)
    else:
        base = (v_k, aw0, aw1, aw2, aw345, aw3, aw4)
    if return_last_increment:
        return (*base, last_increment_relres)
    return base


def wave_backward_uniform_fused(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    col_log_probs,
    node_child1, node_child2, leaf_term_wt,
    neumann_terms=3,
    leaf_state_idx=None,
    leaf_logp=None,
    has_leaf_term=True,
    active_mask=None,
    node_parent=None,
    max_ancestor_depth=None,
    pibar_row_max=None,
    item_idx=None,
    item_indexed_consts=False,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
    grad_col_log_probs=None,
    use_col_weights=True,
    self_loop_grad_targets=None,
    initial_v=None,
    self_loop_solver="neumann",
    return_last_increment=False,
):
    """Fused backward: precompute + Neumann + param VJP in one kernel per wave.

    Args:
        Pi_star: [C, S] converged Pi
        Pibar_star: [C, S] converged Pibar
        ws: wave start offset
        W: wave size
        S: number of state
        dts_r: [W, S] or None
        rhs: [W, S] incoming adjoint
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const:
            [S], [W, S], or [G, S] when item_indexed_consts=True
        node_child1, node_child2: [S] long
        leaf_term_wt: [W, S]
        neumann_terms: int
        leaf_state_idx: optional [C] row -> state leaf index, -1 for non-leaves
        leaf_logp: optional [S], [G], or [G, S] log_pS values used with
            leaf_state_idx

    Returns:
        v_k: [W, S] Neumann-solved adjoint
        aw0, aw1, aw2, aw345, aw3, aw4: [W, S] per-element param grad contributions
    """
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = (
        requested_has_leaf_term
        and leaf_state_idx is not None
        and leaf_logp is not None
    )
    const_layout = _uniform_backward_const_layout(
        DL_const, item_idx, bool(item_indexed_consts)
    )
    if bool(item_indexed_consts) and use_leaf_index:
        if leaf_logp.ndim == 1:
            leaf_logp = leaf_logp.unsqueeze(-1).expand(-1, S).contiguous()
        elif leaf_logp.ndim == 2 and int(leaf_logp.shape[1]) == 1:
            leaf_logp = leaf_logp.expand(-1, S).contiguous()
        else:
            leaf_logp = leaf_logp.contiguous()
    leaf_logp_mode = _uniform_backward_leaf_logp_mode(
        use_leaf_index, leaf_logp, item_idx, bool(item_indexed_consts)
    )
    if item_idx is not None:
        item_idx = item_idx.to(device=Pi_star.device, dtype=torch.long).contiguous()
    if node_parent is None:
        raise ValueError("node_parent is required for the retained backward fast path")
    node_parent = node_parent.to(device=Pi_star.device).contiguous()
    if Pi_star.device.type == "cuda" and node_parent.dtype != torch.int32:
        node_parent = node_parent.to(dtype=torch.int32)
    if max_ancestor_depth is None:
        raise ValueError("max_ancestor_depth is required for the retained backward fast path")
    max_ancestor_depth = max(1, int(max_ancestor_depth))
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained backward fast path")
    pibar_row_max = pibar_row_max.to(device=Pi_star.device, dtype=Pi_star.dtype).contiguous()

    return _wave_backward_uniform_2d(
        Pi_star,
        Pibar_star,
        ws,
        W,
        S,
        dts_r,
        rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        col_log_probs,
        node_child1,
        node_child2,
        leaf_term_wt,
        neumann_terms=neumann_terms,
        leaf_state_idx=leaf_state_idx,
        leaf_logp=leaf_logp,
        has_leaf_term=requested_has_leaf_term,
        active_mask=active_mask,
        node_parent=node_parent,
        max_ancestor_depth=max_ancestor_depth,
        pibar_row_max=pibar_row_max,
        item_idx=item_idx,
        const_layout=const_layout,
        leaf_logp_mode=leaf_logp_mode,
        use_leaf_index=use_leaf_index,
        compact_level_ptr=compact_level_ptr,
        compact_level_parents=compact_level_parents,
        compact_level_child1=compact_level_child1,
        compact_level_child2=compact_level_child2,
        grad_col_log_probs=grad_col_log_probs,
        use_col_weights=use_col_weights,
        self_loop_grad_targets=self_loop_grad_targets,
        initial_v=initial_v,
        self_loop_solver=self_loop_solver,
        return_last_increment=return_last_increment,
    )






@triton.jit
def _dts_cross_backward_accum_kernel(

    Pi_star_ptr,
    Pibar_star_ptr,

    v_k_ptr,
    active_mask_ptr,

    sl_ptr,
    sr_ptr,
    reduce_idx_ptr,
    wlsp_ptr,

    log_pD_arg,
    log_pS_arg,
    item_idx_ptr,

    node_child1_ptr,
    node_child2_ptr,

    accumulated_rhs_ptr,
    grad_Pibar_l_ptr,
    grad_Pibar_r_ptr,
    param_pD_ptr,
    param_pS_ptr,
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_mt_ptr,
    grad_mt_partial_ptr,
    pibar_ud_ptr,
    pibar_A_ptr,
    pibar_side_active_ptr,
    mt_ptr,
    pibar_row_max_ptr,
    side_active_threshold_ptr,

    ws,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
    MERGE_S_TERM: tl.constexpr,
    DEVICE_SCALAR_PARAMS: tl.constexpr,
    PARAM_LAYOUT: tl.constexpr,
    PARAM_GRAD_LAYOUT: tl.constexpr,
    MT_LAYOUT: tl.constexpr,
    GRAD_MT_LAYOUT: tl.constexpr,
    ACCUM_PARAM_REDUCTIONS: tl.constexpr,
    ACCUM_MT_REDUCTION: tl.constexpr,
    GRAD_MT_SCALAR: tl.constexpr,
    GRAD_MT_TWO_STAGE: tl.constexpr,
    GRAD_MT_TILE_SPLITS: tl.constexpr,
    OUTPUT_PIBAR_UD: tl.constexpr,
    OUTPUT_SIDE_ACTIVE: tl.constexpr,
    SIDE_ACTIVE_THRESHOLD_ENABLED: tl.constexpr,
    SKIP_INACTIVE_PIBAR_OUTPUT_ZERO: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """DTS cross-row backward with direct accumulation of Pi adjoints.

    It writes direct Pi contributions into accumulated_rhs instead of materializing
    grad_Pi_l/grad_Pi_r and relying on two PyTorch index_add_ calls.
    Pibar adjoints are still materialized because they feed the uniform Pibar
    VJP kernel.
    """
    NEG_LARGE: tl.constexpr = -float("inf")

    i = tl.program_id(0)

    sl = tl.load(sl_ptr + i).to(tl.int64)
    sr = tl.load(sr_ptr + i).to(tl.int64)
    parent_w = tl.load(reduce_idx_ptr + i).to(tl.int64)
    wlsp = tl.load(wlsp_ptr + i).to(DTYPE)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            out_base = i * S
            ud_l_base = i * S
            ud_r_base = (tl.program_id(0) + 0 + tl.num_programs(0)) * S
            zero_scalar = tl.zeros((1,), dtype=DTYPE)
            _scalar_off = tl.arange(0, 1)
            if not ACCUM_PARAM_REDUCTIONS:
                tl.store(param_pD_ptr + i + _scalar_off, zero_scalar)
                tl.store(param_pS_ptr + i + _scalar_off, zero_scalar)
            if OUTPUT_PIBAR_UD:
                if OUTPUT_SIDE_ACTIVE:
                    tl.store(pibar_side_active_ptr + i + _scalar_off, 0)
                    tl.store(pibar_side_active_ptr + tl.num_programs(0) + i + _scalar_off, 0)
                if SKIP_INACTIVE_PIBAR_OUTPUT_ZERO:
                    return
                tl.store(pibar_A_ptr + i + _scalar_off, zero_scalar)
                tl.store(pibar_A_ptr + tl.num_programs(0) + i + _scalar_off, zero_scalar)
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                if OUTPUT_PIBAR_UD:
                    tl.store(pibar_ud_ptr + ud_l_base + s_offs, zero, mask=mask)
                    tl.store(pibar_ud_ptr + ud_r_base + s_offs, zero, mask=mask)
                else:
                    tl.store(grad_Pibar_l_ptr + out_base + s_offs, zero, mask=mask)
                    tl.store(grad_Pibar_r_ptr + out_base + s_offs, zero, mask=mask)
            return
    else:
        parent_active = True

    parent_global = ws + parent_w
    if (
        PARAM_LAYOUT == 2
        or PARAM_LAYOUT == 3
        or PARAM_GRAD_LAYOUT == 2
        or PARAM_GRAD_LAYOUT == 3
    ):
        parent_item = tl.load(item_idx_ptr + parent_global).to(tl.int64)
    else:
        parent_item = 0

    if MT_LAYOUT == 1 or GRAD_MT_LAYOUT == 1:
        item_l = tl.load(item_idx_ptr + sl).to(tl.int64)
        item_r = tl.load(item_idx_ptr + sr).to(tl.int64)
    else:
        item_l = 0
        item_r = 0

    if PARAM_LAYOUT == 0 and DEVICE_SCALAR_PARAMS:
        log_pD = tl.load(log_pD_arg).to(DTYPE)
        log_pS = tl.load(log_pS_arg).to(DTYPE)
    elif PARAM_LAYOUT == 0:
        log_pD = log_pD_arg
        log_pS = log_pS_arg
    elif PARAM_LAYOUT == 2:
        log_pD = tl.load(log_pD_arg + parent_item).to(DTYPE)
        log_pS = tl.load(log_pS_arg + parent_item).to(DTYPE)
    else:
        log_pD = tl.zeros((1,), dtype=DTYPE)
        log_pS = tl.zeros((1,), dtype=DTYPE)

    pi_l_base = sl * stride_C
    pi_r_base = sr * stride_C
    pibar_l_base = sl * stride_C
    pibar_r_base = sr * stride_C
    parent_pi_base = (ws + parent_w) * stride_C
    parent_vk_base = parent_w * S
    out_base = i * S

    sum_pD = tl.zeros((1,), dtype=DTYPE)
    sum_pS = tl.zeros((1,), dtype=DTYPE)
    sum_ud_l = tl.zeros((1,), dtype=DTYPE)
    sum_ud_r = tl.zeros((1,), dtype=DTYPE)
    _scalar_off = tl.arange(0, 1)
    if OUTPUT_PIBAR_UD:
        row_max_l = tl.load(pibar_row_max_ptr + sl).to(DTYPE)
        row_max_r = tl.load(pibar_row_max_ptr + sr).to(DTYPE)
        side_nonzero_l = tl.full((1,), value=0, dtype=tl.int32)
        side_nonzero_r = tl.full((1,), value=0, dtype=tl.int32)
        side_abs_bound_l = tl.zeros((1,), dtype=DTYPE)
        side_abs_bound_r = tl.zeros((1,), dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        Pibar_l = tl.load(Pibar_star_ptr + pibar_l_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        Pibar_r = tl.load(Pibar_star_ptr + pibar_r_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)

        c1 = tl.load(node_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(node_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)

        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0).to(DTYPE)

        if PARAM_LAYOUT == 1:
            log_pD_s = tl.load(log_pD_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            log_pS_s = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
        elif PARAM_LAYOUT == 3:
            param_base = parent_item * S
            log_pD_s = tl.load(log_pD_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            log_pS_s = tl.load(log_pS_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
        else:
            log_pD_s = log_pD
            log_pS_s = log_pS

        d0 = log_pD_s + Pi_l + Pi_r
        d1 = Pi_l + Pibar_r
        d2 = Pi_r + Pibar_l
        d3 = log_pS_s + Pi_l_s1 + Pi_r_s2
        d4 = log_pS_s + Pi_r_s1 + Pi_l_s2

        parent_valid = Pi_parent != NEG_LARGE
        w0 = tl.where(parent_valid, tl.exp2(wlsp + d0 - Pi_parent), tl.zeros_like(d0))
        w1 = tl.where(parent_valid, tl.exp2(wlsp + d1 - Pi_parent), tl.zeros_like(d1))
        w2 = tl.where(parent_valid, tl.exp2(wlsp + d2 - Pi_parent), tl.zeros_like(d2))
        w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
        w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))

        vd0 = v_k_val * w0
        vd1 = v_k_val * w1
        vd2 = v_k_val * w2
        vd3 = v_k_val * w3
        vd4 = v_k_val * w4

        pi_l_out = accumulated_rhs_ptr + pi_l_base + s_offs
        pi_r_out = accumulated_rhs_ptr + pi_r_base + s_offs
        if USE_ATOMICS:
            tl.atomic_add(pi_l_out, vd0 + vd1, sem="relaxed", mask=mask)
            tl.atomic_add(pi_r_out, vd0 + vd2, sem="relaxed", mask=mask)
        else:
            pi_l_cur = tl.load(pi_l_out, mask=mask, other=0.0).to(DTYPE)
            pi_r_cur = tl.load(pi_r_out, mask=mask, other=0.0).to(DTYPE)
            tl.store(pi_l_out, pi_l_cur + vd0 + vd1, mask=mask)
            tl.store(pi_r_out, pi_r_cur + vd0 + vd2, mask=mask)
        if OUTPUT_PIBAR_UD:
            if MT_LAYOUT == 1:
                mt_l = tl.load(mt_ptr + item_l * S + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
                mt_r = tl.load(mt_ptr + item_r * S + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
            else:
                mt = tl.load(mt_ptr + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
                mt_l = mt
                mt_r = mt
            finite_l = (Pibar_l != NEG_LARGE) & mask
            finite_r = (Pibar_r != NEG_LARGE) & mask
            inv_denom_l = tl.where(
                finite_l,
                tl.exp2(row_max_l + mt_l - Pibar_l),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            inv_denom_r = tl.where(
                finite_r,
                tl.exp2(row_max_r + mt_r - Pibar_r),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            ud_l = vd2 * inv_denom_l
            ud_r = vd1 * inv_denom_r
            tl.store(pibar_ud_ptr + i * S + s_offs, ud_l, mask=valid_mask)
            tl.store(pibar_ud_ptr + (tl.num_programs(0) + i) * S + s_offs, ud_r, mask=valid_mask)
            sum_ud_l += tl.sum(tl.where(mask, ud_l, 0.0), axis=0)
            sum_ud_r += tl.sum(tl.where(mask, ud_r, 0.0), axis=0)
            if OUTPUT_SIDE_ACTIVE:
                if SIDE_ACTIVE_THRESHOLD_ENABLED:
                    side_abs_bound_l += tl.sum(tl.where(mask, tl.abs(ud_l), 0.0), axis=0)
                    side_abs_bound_r += tl.sum(tl.where(mask, tl.abs(ud_r), 0.0), axis=0)
                else:
                    side_nonzero_l += tl.where(tl.max(tl.abs(ud_l), axis=0) != 0.0, 1, 0)
                    side_nonzero_r += tl.where(tl.max(tl.abs(ud_r), axis=0) != 0.0, 1, 0)
        else:
            tl.store(grad_Pibar_l_ptr + out_base + s_offs, vd2, mask=valid_mask)
            tl.store(grad_Pibar_r_ptr + out_base + s_offs, vd1, mask=valid_mask)

        if ACCUM_PARAM_REDUCTIONS and PARAM_GRAD_LAYOUT == 1:
            tl.atomic_add(grad_log_pD_ptr + s_offs, vd0, sem="relaxed", mask=mask)
            tl.atomic_add(grad_log_pS_ptr + s_offs, vd3 + vd4, sem="relaxed", mask=mask)
        elif ACCUM_PARAM_REDUCTIONS and PARAM_GRAD_LAYOUT == 3:
            grad_param_base = parent_item * S
            tl.atomic_add(grad_log_pD_ptr + grad_param_base + s_offs, vd0, sem="relaxed", mask=mask)
            tl.atomic_add(grad_log_pS_ptr + grad_param_base + s_offs, vd3 + vd4, sem="relaxed", mask=mask)
        else:
            sum_pD += tl.sum(vd0, axis=0)
            sum_pS += tl.sum(vd3 + vd4, axis=0)
        if ACCUM_MT_REDUCTION:
            mt_contrib = vd1 + vd2
            if GRAD_MT_LAYOUT == 1:
                tl.atomic_add(
                    grad_mt_ptr + item_l * S + s_offs,
                    vd2,
                    sem="relaxed",
                    mask=mask,
                )
                tl.atomic_add(
                    grad_mt_ptr + item_r * S + s_offs,
                    vd1,
                    sem="relaxed",
                    mask=mask,
                )
            elif GRAD_MT_SCALAR:
                tl.atomic_add(
                    grad_mt_ptr + _scalar_off,
                    tl.sum(tl.where(mask, mt_contrib, 0.0), axis=0),
                    sem="relaxed",
                )
            elif GRAD_MT_TWO_STAGE:
                mt_tile = i // GRAD_MT_TILE_SPLITS
                tl.atomic_add(
                    grad_mt_partial_ptr + mt_tile * S + s_offs,
                    mt_contrib,
                    sem="relaxed",
                    mask=mask,
                )
            else:
                tl.atomic_add(
                    grad_mt_ptr + s_offs,
                    mt_contrib,
                    sem="relaxed",
                    mask=mask,
                )

        if MERGE_S_TERM:
            pi_l_c1_out = accumulated_rhs_ptr + pi_l_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + pi_r_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + pi_r_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + pi_l_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, vd3, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, vd4, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, vd3, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, vd4, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0)
                tl.store(pi_l_c1_out, pi_l_c1_cur + vd3, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + vd4, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + vd3, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + vd4, mask=c2_valid)

    if ACCUM_PARAM_REDUCTIONS:
        if PARAM_GRAD_LAYOUT == 0:
            tl.atomic_add(grad_log_pD_ptr + _scalar_off, sum_pD, sem="relaxed")
            tl.atomic_add(grad_log_pS_ptr + _scalar_off, sum_pS, sem="relaxed")
        elif PARAM_GRAD_LAYOUT == 2:
            tl.atomic_add(
                grad_log_pD_ptr + parent_item + _scalar_off,
                sum_pD,
                sem="relaxed",
            )
            tl.atomic_add(
                grad_log_pS_ptr + parent_item + _scalar_off,
                sum_pS,
                sem="relaxed",
            )
    else:
        tl.store(param_pD_ptr + i + _scalar_off, sum_pD)
        tl.store(param_pS_ptr + i + _scalar_off, sum_pS)
    if OUTPUT_PIBAR_UD:
        tl.store(pibar_A_ptr + i + _scalar_off, sum_ud_l)
        tl.store(pibar_A_ptr + tl.num_programs(0) + i + _scalar_off, sum_ud_r)
        if OUTPUT_SIDE_ACTIVE:
            if SIDE_ACTIVE_THRESHOLD_ENABLED:
                threshold = tl.load(side_active_threshold_ptr).to(DTYPE)
                bound_l = side_abs_bound_l
                bound_r = side_abs_bound_r
                tl.store(pibar_side_active_ptr + i + _scalar_off, bound_l > threshold)
                tl.store(
                    pibar_side_active_ptr + tl.num_programs(0) + i + _scalar_off,
                    bound_r > threshold,
                )
            else:
                tl.store(pibar_side_active_ptr + i + _scalar_off, side_nonzero_l != 0)
                tl.store(
                    pibar_side_active_ptr + tl.num_programs(0) + i + _scalar_off,
                    side_nonzero_r != 0,
                )

    if not MERGE_S_TERM:
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & parent_active

            c1 = tl.load(node_child1_ptr + s_offs, mask=mask, other=0)
            c2 = tl.load(node_child2_ptr + s_offs, mask=mask, other=0)
            c1_valid = (c1 < S) & mask
            c2_valid = (c2 < S) & mask

            Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
            Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)
            Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
            Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)

            Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
            v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0).to(DTYPE)

            if PARAM_LAYOUT == 1:
                log_pS_s = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            elif PARAM_LAYOUT == 3:
                log_pS_s = tl.load(log_pS_arg + parent_item * S + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            else:
                log_pS_s = log_pS

            d3 = log_pS_s + Pi_l_s1 + Pi_r_s2
            d4 = log_pS_s + Pi_r_s1 + Pi_l_s2

            parent_valid = Pi_parent != NEG_LARGE
            w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
            w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))
            vd3 = v_k_val * w3
            vd4 = v_k_val * w4

            pi_l_c1_out = accumulated_rhs_ptr + pi_l_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + pi_r_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + pi_r_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + pi_l_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, vd3, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, vd4, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, vd3, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, vd4, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0).to(DTYPE)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0).to(DTYPE)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0).to(DTYPE)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0).to(DTYPE)
                tl.store(pi_l_c1_out, pi_l_c1_cur + vd3, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + vd4, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + vd3, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + vd4, mask=c2_valid)


@triton.jit
def _dts_grad_mt_two_stage_reduce_kernel(
    partial_ptr,
    grad_mt_ptr,
    n_tiles: tl.constexpr,
    S: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Reduce split-tile DTS grad_mt partials by state."""
    s_block = tl.program_id(0)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_s = s_offs < S
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)

    tile_start = 0
    while tile_start < n_tiles:
        tile_offs = tile_start + tl.arange(0, BLOCK_TILES)
        mask = (tile_offs[:, None] < n_tiles) & valid_s[None, :]
        vals = tl.load(
            partial_ptr + tile_offs[:, None] * S + s_offs[None, :],
            mask=mask,
            other=0.0,
        )
        acc += tl.sum(vals, axis=0)
        tile_start += BLOCK_TILES

    current = tl.load(grad_mt_ptr + s_offs, mask=valid_s, other=0.0)
    tl.store(grad_mt_ptr + s_offs, current + acc, mask=valid_s)


def dts_cross_backward_accum_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    node_child1, node_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    use_atomics=True,
    merge_s_term=False,
    grad_log_pD=None,
    grad_log_pS=None,
    grad_mt=None,
    accum_param_reductions=False,
    accum_mt_reduction=False,
    output_pibar_ud=False,
    output_pibar_side_active=False,
    pibar_side_threshold=0.0,
    mt_squeezed=None,
    pibar_row_max=None,
    grad_mt_two_stage=False,
    grad_mt_two_stage_tile_splits=128,
    skip_inactive_pibar_output_zero=False,
    item_idx=None,
):
    """Fused DTS backward with direct Pi-adjoint accumulation."""
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp
    item_idx_arg = None
    if item_idx is not None:
        item_idx_arg = item_idx.to(device=device, dtype=torch.long).contiguous()
    log_pD_arg, log_pS_arg, param_layout = _dts_layout_param_args(
        log_pD, log_pS, item_idx=item_idx_arg, S=S, device=device, dtype=dtype
    )
    device_scalar_params = False
    if param_layout == 0:
        device_scalar_params = True

    if accum_param_reductions and (grad_log_pD is None or grad_log_pS is None):
        raise ValueError("grad_log_pD/grad_log_pS are required when accumulating DTS scalar reductions")
    if accum_param_reductions:
        param_grad_layout = _dts_grad_layout(grad_log_pD, item_idx=item_idx_arg, S=S)
        if param_grad_layout != _dts_grad_layout(grad_log_pS, item_idx=item_idx_arg, S=S):
            raise ValueError("grad_log_pD/grad_log_pS must use the same DTS gradient layout")
    else:
        param_grad_layout = 0
    if accum_mt_reduction and grad_mt is None:
        raise ValueError("grad_mt is required when accumulating DTS mt reductions")
    if accum_mt_reduction:
        if grad_mt.numel() == 1 or (grad_mt.ndim == 1 and int(grad_mt.shape[0]) == int(S)):
            grad_mt_layout = 0
        elif item_idx_arg is not None and grad_mt.ndim == 2 and int(grad_mt.shape[1]) == int(S):
            grad_mt_layout = 1
        else:
            raise ValueError("DTS mt reduction target must be scalar, [S], or [G, S]")
    else:
        grad_mt_layout = 0
    if output_pibar_ud and (mt_squeezed is None or pibar_row_max is None):
        raise ValueError("mt_squeezed and pibar_row_max are required when outputting Pibar u_d")
    if output_pibar_ud:
        if mt_squeezed.ndim == 1 and int(mt_squeezed.shape[0]) == int(S):
            mt_layout = 0
        elif item_idx_arg is not None and mt_squeezed.ndim == 2 and int(mt_squeezed.shape[1]) == int(S):
            mt_layout = 1
        else:
            raise ValueError("mt_squeezed must have shape [S] or [G, S] when outputting Pibar u_d")
    else:
        mt_layout = 0
    if output_pibar_ud and pibar_row_max.numel() < Pi_star.shape[0]:
        raise ValueError("pibar_row_max must contain one row-max value per Pi row")
    if output_pibar_side_active and not output_pibar_ud:
        raise ValueError("output_pibar_side_active requires output_pibar_ud")
    if torch.is_tensor(pibar_side_threshold):
        if pibar_side_threshold.numel() != 1:
            raise ValueError("pibar_side_threshold tensor must contain one value")
        side_threshold_enabled = bool(output_pibar_side_active)
        side_active_threshold_arg = _device_scalar_param(
            pibar_side_threshold, device=device, dtype=dtype
        )
    else:
        pibar_side_threshold = float(pibar_side_threshold)
        if pibar_side_threshold < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
        side_threshold_enabled = bool(output_pibar_side_active and pibar_side_threshold > 0.0)
        side_active_threshold_arg = (
            torch.tensor([pibar_side_threshold], device=device, dtype=dtype)
            if side_threshold_enabled
            else None
        )

    if output_pibar_ud:
        grad_Pibar_l = None
        grad_Pibar_r = None
    else:
        grad_Pibar_l = torch.empty((n_ws, S), device=device, dtype=dtype)
        grad_Pibar_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    if output_pibar_ud:
        pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype)
        pibar_A = torch.empty((2 * n_ws,), device=device, dtype=dtype)
    else:
        pibar_ud = None
        pibar_A = None
    pibar_side_active = (
        torch.empty((2 * n_ws,), device=device, dtype=torch.bool)
        if output_pibar_side_active
        else None
    )
    if accum_param_reductions:
        param_pD = None
        param_pS = None
    else:
        param_pD = torch.empty(n_ws, device=device, dtype=dtype)
        param_pS = torch.empty(n_ws, device=device, dtype=dtype)
    param_pD_arg = grad_log_pD if accum_param_reductions else param_pD
    param_pS_arg = grad_log_pS if accum_param_reductions else param_pS
    dummy = pibar_ud if output_pibar_ud else grad_Pibar_l
    grad_log_pD_arg = grad_log_pD if accum_param_reductions else dummy
    grad_log_pS_arg = grad_log_pS if accum_param_reductions else dummy
    grad_mt_arg = grad_mt if accum_mt_reduction else dummy
    pibar_ud_arg = pibar_ud if output_pibar_ud else dummy
    pibar_A_arg = pibar_A if output_pibar_ud else dummy
    pibar_side_active_arg = pibar_side_active if output_pibar_side_active else dummy
    mt_arg = mt_squeezed.contiguous() if output_pibar_ud and not mt_squeezed.is_contiguous() else mt_squeezed
    pibar_row_max_arg = (
        pibar_row_max.contiguous()
        if output_pibar_ud and not pibar_row_max.is_contiguous()
        else pibar_row_max
    )
    mt_arg = mt_arg if output_pibar_ud else dummy
    pibar_row_max_arg = pibar_row_max_arg if output_pibar_ud else dummy
    side_active_threshold_arg = side_active_threshold_arg if side_threshold_enabled else dummy
    item_idx_kernel_arg = item_idx_arg if item_idx_arg is not None else sl
    grad_mt_scalar = bool(accum_mt_reduction and grad_mt.numel() == 1)
    use_grad_mt_two_stage = bool(
        grad_mt_two_stage
        and accum_mt_reduction
        and grad_mt_layout == 0
        and not grad_mt_scalar
        and grad_mt.numel() == S
    )
    grad_mt_two_stage_tile_splits = max(1, int(grad_mt_two_stage_tile_splits))
    n_grad_mt_tiles = triton.cdiv(n_ws, grad_mt_two_stage_tile_splits)
    if use_grad_mt_two_stage:
        grad_mt_partial = torch.empty((n_grad_mt_tiles, S), device=device, dtype=dtype)
        grad_mt_partial.zero_()
    else:
        grad_mt_partial = dummy

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    launch_options = {"num_warps": 8}

    _dts_cross_backward_accum_kernel[(n_ws,)](
        Pi_star, Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl, sr, reduce_idx, wlsp_flat,
        log_pD_arg, log_pS_arg, item_idx_kernel_arg,
        node_child1, node_child2,
        accumulated_rhs,
        grad_Pibar_l if grad_Pibar_l is not None else pibar_ud,
        grad_Pibar_r if grad_Pibar_r is not None else pibar_ud,
        param_pD_arg, param_pS_arg,
        grad_log_pD_arg, grad_log_pS_arg, grad_mt_arg,
        grad_mt_partial,
        pibar_ud_arg, pibar_A_arg, pibar_side_active_arg, mt_arg, pibar_row_max_arg,
        side_active_threshold_arg,
        ws, S, stride_C, BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ATOMICS=bool(use_atomics),
        MERGE_S_TERM=bool(merge_s_term),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        PARAM_LAYOUT=int(param_layout),
        PARAM_GRAD_LAYOUT=int(param_grad_layout),
        MT_LAYOUT=int(mt_layout),
        GRAD_MT_LAYOUT=int(grad_mt_layout),
        ACCUM_PARAM_REDUCTIONS=bool(accum_param_reductions),
        ACCUM_MT_REDUCTION=bool(accum_mt_reduction),
        GRAD_MT_SCALAR=bool(grad_mt_scalar),
        GRAD_MT_TWO_STAGE=bool(use_grad_mt_two_stage),
        GRAD_MT_TILE_SPLITS=grad_mt_two_stage_tile_splits,
        OUTPUT_PIBAR_UD=bool(output_pibar_ud),
        OUTPUT_SIDE_ACTIVE=bool(output_pibar_side_active),
        SIDE_ACTIVE_THRESHOLD_ENABLED=side_threshold_enabled,
        SKIP_INACTIVE_PIBAR_OUTPUT_ZERO=bool(skip_inactive_pibar_output_zero),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if use_grad_mt_two_stage:
        mt_block_s = min(64, triton.next_power_of_2(S))
        mt_block_tiles = 16
        _dts_grad_mt_two_stage_reduce_kernel[(triton.cdiv(S, mt_block_s),)](
            grad_mt_partial,
            grad_mt,
            n_grad_mt_tiles,
            S,
            mt_block_tiles,
            mt_block_s,
            DTYPE=_tl_float_dtype(dtype),
            num_warps=4,
        )

    if output_pibar_ud:
        if output_pibar_side_active:
            return pibar_ud, pibar_A, pibar_side_active, param_pD, param_pS
        return pibar_ud, pibar_A, param_pD, param_pS
    return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS






@triton.jit
def _pibar_ud_side_active_kernel(
    pibar_ud_ptr,
    side_active_ptr,
    side_active_threshold_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SIDE_ACTIVE_THRESHOLD_ENABLED: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Mark split-side rows whose staged u_d should run Pibar tree work."""
    row = tl.program_id(0)
    row_base = row * S
    row_absmax = tl.full([1], value=0.0, dtype=DTYPE)
    row_abssum = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        vals = tl.load(pibar_ud_ptr + row_base + s_offs, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        row_absmax = tl.maximum(row_absmax, tl.max(abs_vals, axis=0))
        row_abssum += tl.sum(tl.where(mask, abs_vals, 0.0), axis=0)

    lane = tl.arange(0, 1)
    if SIDE_ACTIVE_THRESHOLD_ENABLED:
        threshold = tl.load(side_active_threshold_ptr).to(DTYPE)
        tl.store(side_active_ptr + row + lane, row_abssum > threshold)
    else:
        tl.store(side_active_ptr + row + lane, row_absmax != 0.0)


@triton.jit
def _uniform_cross_pibar_vjp_tree_from_ud_compact_kernel(
    Pi_star_ptr,
    col_log_probs_ptr,
    pibar_ud_ptr,
    pibar_A_ptr,
    side_active_ptr,
    sl_ptr,
    sr_ptr,
    reduce_idx_ptr,
    active_mask_ptr,
    pibar_row_max_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    accumulated_rhs_ptr,
    grad_col_log_probs_ptr,
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    ACCUM_COL_GRAD: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar from-u_d tree correction using compact per-level nodes."""
    NEG_LARGE: tl.constexpr = -float("inf")

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws
    if USE_SIDE_ACTIVE:
        side_active = tl.load(side_active_ptr + row)
        if side_active == 0:
            return

    child_l = tl.load(sl_ptr + split_i).to(tl.int64)
    child_r = tl.load(sr_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i).to(tl.int64)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return
    else:
        row_active = True

    pi_base = child * stride_C
    row_base = row * S
    row_max = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    A = tl.load(pibar_A_ptr + row).to(DTYPE)

    tl.debug_barrier()
    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        p_start = level_start
        while p_start < level_end:
            node_offs = p_start + tl.arange(0, BLOCK_S)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=-1)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            parent_valid = node_mask & (parent >= 0) & (parent < S) & row_active
            c1_valid = node_mask & (c1 >= 0) & (c1 < S) & row_active
            c2_valid = node_mask & (c2 >= 0) & (c2 < S) & row_active

            parent_val = tl.load(pibar_ud_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(pibar_ud_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(pibar_ud_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(pibar_ud_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
            p_start += BLOCK_S
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        if USE_COL_WEIGHTS:
            col_logp = tl.load(col_log_probs_ptr + s_offs, mask=valid_mask, other=NEG_LARGE)
            p_prime = tl.exp2(col_logp + pi_val - row_max_safe)
        else:
            p_prime = tl.exp2(pi_val - row_max_safe)
        subtree_sum = tl.load(pibar_ud_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)
        if ACCUM_COL_GRAD:
            tl.atomic_add(
                grad_col_log_probs_ptr + s_offs,
                contrib,
                sem="relaxed",
                mask=mask,
            )


@triton.jit
def _uniform_cross_pibar_vjp_tree_from_ud_reg_kernel(
    Pi_star_ptr,
    col_log_probs_ptr,
    pibar_ud_ptr,
    pibar_A_ptr,
    side_active_ptr,
    sl_ptr,
    sr_ptr,
    reduce_idx_ptr,
    active_mask_ptr,
    pibar_row_max_ptr,
    dfs_node_ptr,
    start_m1_ptr,
    end_m1_ptr,
    accumulated_rhs_ptr,
    grad_col_log_probs_ptr,
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    ACCUM_COL_GRAD: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Register-resident Pibar-from-u_d tree correction (subtree sums via DFS cumsum).

    Also writes the subtree sums back into pibar_ud, matching the in-place mutation of
    the level-walk version (part of the captured wrapper contract)."""
    NEG_LARGE: tl.constexpr = -float("inf")

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws
    if USE_SIDE_ACTIVE:
        if tl.load(side_active_ptr + row) == 0:
            return
    child_l = tl.load(sl_ptr + split_i).to(tl.int64)
    child_r = tl.load(sr_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i).to(tl.int64)
        if tl.load(active_mask_ptr + parent_w) == 0:
            return

    pi_base = child * stride_C
    row_base = row * S
    row_max = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    A = tl.load(pibar_A_ptr + row).to(DTYPE)

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)
    u_d = tl.load(pibar_ud_ptr + row_base + s_offs, mask=mask, other=0.0).to(DTYPE)
    dfs_node = tl.load(dfs_node_ptr + s_offs, mask=mask, other=0)
    start_m1 = tl.load(start_m1_ptr + s_offs, mask=mask, other=-1)
    end_m1 = tl.load(end_m1_ptr + s_offs, mask=mask, other=0)
    u_dfs = tl.where(mask, tl.gather(u_d, dfs_node, axis=0), zero)
    cum = tl.cumsum(u_dfs, axis=0)
    ce = tl.gather(cum, tl.where(end_m1 >= 0, end_m1, 0), axis=0)
    cs = tl.where(start_m1 >= 0, tl.gather(cum, tl.where(start_m1 >= 0, start_m1, 0), axis=0), zero)
    subtree_sum = ce - cs
    tl.store(pibar_ud_ptr + row_base + s_offs, subtree_sum, mask=mask)

    pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
    if USE_COL_WEIGHTS:
        col_logp = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(col_logp + pi_val - row_max_safe)
    else:
        p_prime = tl.exp2(pi_val - row_max_safe)
    contrib = p_prime * (A - subtree_sum)
    tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)
    if ACCUM_COL_GRAD:
        tl.atomic_add(grad_col_log_probs_ptr + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_tree_from_ud_fused(
    Pi_star,
    col_log_probs,
    pibar_ud,
    pibar_A,
    sl,
    sr,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    pibar_row_max=None,
    skip_zero_sides=False,
    side_active=None,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
    grad_col_log_probs=None,
    use_col_weights=True,
    side_active_threshold=0.0,
):
    """Uniform-Pibar VJP tree correction from DTS-staged u_d."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for DTS-staged Pibar VJP")
    if torch.is_tensor(side_active_threshold):
        if side_active_threshold.numel() != 1:
            raise ValueError("side_active_threshold tensor must contain one value")
        side_active_threshold_enabled = True
        side_active_threshold_arg = _device_scalar_param(
            side_active_threshold, device=Pi_star.device, dtype=Pi_star.dtype
        )
    else:
        side_active_threshold = float(side_active_threshold)
        if side_active_threshold < 0.0:
            raise ValueError("side_active_threshold must be non-negative")
        side_active_threshold_enabled = side_active_threshold > 0.0
        side_active_threshold_arg = (
            torch.tensor([side_active_threshold], device=Pi_star.device, dtype=Pi_star.dtype)
            if side_active_threshold_enabled
            else None
        )

    BLOCK_S = min(256, triton.next_power_of_2(S))
    launch_options = {"num_warps": 4}
    stride_C = Pi_star.stride(0)
    if side_active is not None:
        if side_active.numel() != 2 * n_ws:
            raise ValueError("side_active must have one entry per split side")
        side_active = side_active.contiguous()
    elif skip_zero_sides:
        side_active = torch.empty((2 * n_ws,), device=Pi_star.device, dtype=torch.bool)
        _pibar_ud_side_active_kernel[(2 * n_ws,)](
            pibar_ud,
            side_active,
            side_active_threshold_arg if side_active_threshold_enabled else pibar_ud,
            S,
            BLOCK_S,
            SIDE_ACTIVE_THRESHOLD_ENABLED=bool(side_active_threshold_enabled),
            DTYPE=_tl_float_dtype(Pi_star.dtype),
            **launch_options,
        )

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact state levels are required for Pibar VJP")
    if compact_level_ptr.numel() < 2:
        raise ValueError("compact_level_ptr must contain at least start and end offsets")
    compact_level_ptr = compact_level_ptr.contiguous()
    compact_level_parents = compact_level_parents.contiguous()
    compact_level_child1 = compact_level_child1.contiguous()
    compact_level_child2 = compact_level_child2.contiguous()
    col_log_probs = col_log_probs.to(device=Pi_star.device, dtype=Pi_star.dtype).contiguous()
    col_grad_arg = (
        grad_col_log_probs
        if grad_col_log_probs is not None
        else pibar_A
    )
    if _VJP_MODE == 1:
        dfs_node, start_m1, end_m1 = _get_dfs_tables_from_compact(
            compact_level_parents, compact_level_child1, compact_level_child2, S
        )
        _uniform_cross_pibar_vjp_tree_from_ud_reg_kernel[(2 * n_ws,)](
            Pi_star,
            col_log_probs,
            pibar_ud,
            pibar_A,
            side_active if side_active is not None else pibar_A,
            sl,
            sr,
            reduce_idx if reduce_idx is not None else sl,
            active_mask if active_mask is not None else pibar_ud,
            pibar_row_max,
            dfs_node,
            start_m1,
            end_m1,
            accumulated_rhs,
            col_grad_arg,
            n_ws,
            S,
            stride_C,
            int(triton.next_power_of_2(S)),
            USE_ACTIVE_MASK=bool(active_mask is not None),
            USE_SIDE_ACTIVE=bool(side_active is not None),
            ACCUM_COL_GRAD=bool(grad_col_log_probs is not None),
            USE_COL_WEIGHTS=bool(use_col_weights),
            DTYPE=_tl_float_dtype(Pi_star.dtype),
            num_warps=_JT_REG_WARPS,
        )
        return
    _uniform_cross_pibar_vjp_tree_from_ud_compact_kernel[(2 * n_ws,)](
        Pi_star,
        col_log_probs,
        pibar_ud,
        pibar_A,
        side_active if side_active is not None else pibar_A,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else pibar_ud,
        pibar_row_max,
        compact_level_ptr,
        compact_level_parents,
        compact_level_child1,
        compact_level_child2,
        accumulated_rhs,
        col_grad_arg,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        N_LEVELS=compact_level_ptr.numel() - 1,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_SIDE_ACTIVE=bool(side_active is not None),
        ACCUM_COL_GRAD=bool(grad_col_log_probs is not None),
        USE_COL_WEIGHTS=bool(use_col_weights),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        **launch_options,
    )
    return side_active
