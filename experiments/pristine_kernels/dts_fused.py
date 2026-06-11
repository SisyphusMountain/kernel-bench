import torch
import triton
import triton.language as tl


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


@triton.jit
def _load_rate(param, item, s_offs, mask, S: tl.constexpr, ROW_STRIDE: tl.constexpr, BY_STATE: tl.constexpr, BLOCK_S: tl.constexpr, DTYPE: tl.constexpr):
    NEG_INF: tl.constexpr = -float("inf")
    if BY_STATE:
        return tl.load(param + item * ROW_STRIDE + s_offs, mask=mask, other=NEG_INF)
    return tl.load(param + item * ROW_STRIDE) + tl.zeros([BLOCK_S], dtype=DTYPE)


@triton.jit
def _dts_eq1_kernel(
    Pi, Pibar, lefts, rights, node_child1, node_child2, log_pD, log_pS,
    log_split_probs, eq1_reduce_idx, active_rows, out, item_idx,
    item_offset, S: tl.constexpr, BLOCK_S: tl.constexpr,
    ROW_STRIDE: tl.constexpr, BY_STATE: tl.constexpr, USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_w = tl.load(eq1_reduce_idx + n).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_w) == 0:
            tl.store(out + parent_w * S + s_offs, tl.full([BLOCK_S], NEG_INF, dtype=DTYPE), mask=mask)
            return

    item = tl.load(item_idx + item_offset + parent_w).to(tl.int64)
    left = tl.load(lefts + n).to(tl.int64)
    right = tl.load(rights + n).to(tl.int64)
    base_l = left * S
    base_r = right * S
    pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG_INF)
    pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG_INF)
    pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG_INF)
    pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG_INF)
    log_d = _load_rate(log_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    log_s = _load_rate(log_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    c1 = tl.load(node_child1 + s_offs, mask=mask, other=S)
    c2 = tl.load(node_child2 + s_offs, mask=mask, other=S)
    c1_valid = c1 < S
    c2_valid = c2 < S
    lsp = tl.load(log_split_probs + n)

    t0 = lsp + log_d + pi_l + pi_r
    t1 = lsp + pi_l + pibar_r
    t2 = lsp + pi_r + pibar_l
    t3 = lsp + log_s + tl.load(Pi + base_l + c1, mask=mask & c1_valid, other=NEG_INF) + tl.load(Pi + base_r + c2, mask=mask & c2_valid, other=NEG_INF)
    t4 = lsp + log_s + tl.load(Pi + base_r + c1, mask=mask & c1_valid, other=NEG_INF) + tl.load(Pi + base_l + c2, mask=mask & c2_valid, other=NEG_INF)
    m = tl.maximum(tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3)), t4)
    m_safe = tl.where(m != NEG_INF, m, tl.zeros_like(m))
    acc = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe) + tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe)
    tl.store(out + parent_w * S + s_offs, tl.log2(acc) + m, mask=mask)


@triton.jit
def _dts_ge2_stage1_kernel(
    Pi, Pibar, lefts, rights, node_child1, node_child2, log_pD, log_pS,
    log_split_probs, ge2_ptr, ge2_parent_ids, active_rows, partial_max,
    partial_sum, item_idx, item_offset, split_offset, MAX_TILES: tl.constexpr,
    S: tl.constexpr, TILE_SPLITS: tl.constexpr, BLOCK_S: tl.constexpr,
    ROW_STRIDE: tl.constexpr, BY_STATE: tl.constexpr, USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    group = tl.program_id(0)
    tile_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_w = tl.load(ge2_parent_ids + group).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_w) == 0:
            return

    item = tl.load(item_idx + item_offset + parent_w).to(tl.int64)
    start = tl.load(ge2_ptr + group)
    end = tl.load(ge2_ptr + group + 1)
    tile_start = start + tile_id * TILE_SPLITS
    if tile_start >= end:
        return
    tile_end = tl.minimum(tile_start + TILE_SPLITS, end)

    m = tl.full([BLOCK_S], NEG_INF, dtype=DTYPE)
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)
    split_rel = tile_start
    while split_rel < tile_end:
        split_i = split_offset + split_rel
        left = tl.load(lefts + split_i).to(tl.int64)
        right = tl.load(rights + split_i).to(tl.int64)
        base_l = left * S
        base_r = right * S
        pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG_INF)
        pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG_INF)
        pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG_INF)
        pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG_INF)
        log_d = _load_rate(log_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
        log_s = _load_rate(log_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
        c1 = tl.load(node_child1 + s_offs, mask=mask, other=S)
        c2 = tl.load(node_child2 + s_offs, mask=mask, other=S)
        c1_valid = c1 < S
        c2_valid = c2 < S
        lsp = tl.load(log_split_probs + split_i)

        v0 = lsp + log_d + pi_l + pi_r
        v1 = lsp + pi_l + pibar_r
        v2 = lsp + pi_r + pibar_l
        v3 = lsp + log_s + tl.load(Pi + base_l + c1, mask=mask & c1_valid, other=NEG_INF) + tl.load(Pi + base_r + c2, mask=mask & c2_valid, other=NEG_INF)
        v4 = lsp + log_s + tl.load(Pi + base_r + c1, mask=mask & c1_valid, other=NEG_INF) + tl.load(Pi + base_l + c2, mask=mask & c2_valid, other=NEG_INF)
        split_m = tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)), v4)
        split_m_safe = tl.where(split_m != NEG_INF, split_m, tl.zeros_like(split_m))
        split_sum = tl.exp2(v0 - split_m_safe) + tl.exp2(v1 - split_m_safe) + tl.exp2(v2 - split_m_safe) + tl.exp2(v3 - split_m_safe) + tl.exp2(v4 - split_m_safe)

        new_m = tl.maximum(m, split_m)
        new_m_safe = tl.where(new_m != NEG_INF, new_m, tl.zeros_like(new_m))
        acc = tl.where(m != NEG_INF, acc * tl.exp2(m - new_m_safe), tl.zeros_like(acc)) + split_sum * tl.exp2(split_m_safe - new_m_safe)
        m = new_m
        split_rel += 1

    partial_row = group * MAX_TILES + tile_id
    tl.store(partial_max + partial_row * S + s_offs, m, mask=mask)
    tl.store(partial_sum + partial_row * S + s_offs, acc, mask=mask)


@triton.jit
def _dts_ge2_stage2_kernel(
    ge2_ptr, ge2_parent_ids, active_rows, partial_max, partial_sum, out,
    MAX_TILES: tl.constexpr, S: tl.constexpr, TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr, USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    group = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_w = tl.load(ge2_parent_ids + group).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_w) == 0:
            tl.store(out + parent_w * S + s_offs, tl.full([BLOCK_S], NEG_INF, dtype=DTYPE), mask=mask)
            return

    start = tl.load(ge2_ptr + group)
    end = tl.load(ge2_ptr + group + 1)
    n_tiles = tl.cdiv(end - start, TILE_SPLITS)
    m = tl.full([BLOCK_S], NEG_INF, dtype=DTYPE)
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)
    tile_id = 0
    while tile_id < n_tiles:
        partial_row = group * MAX_TILES + tile_id
        pm = tl.load(partial_max + partial_row * S + s_offs, mask=mask, other=NEG_INF)
        ps = tl.load(partial_sum + partial_row * S + s_offs, mask=mask, other=0.0)
        new_m = tl.maximum(m, pm)
        new_m_safe = tl.where(new_m != NEG_INF, new_m, tl.zeros_like(new_m))
        acc = tl.where(m != NEG_INF, acc * tl.exp2(m - new_m_safe), tl.zeros_like(acc)) + tl.where(pm != NEG_INF, ps * tl.exp2(pm - new_m_safe), tl.zeros_like(acc))
        m = new_m
        tile_id += 1

    tl.store(out + parent_w * S + s_offs, tl.log2(acc) + m, mask=mask)


def compute_dts_forward(
    Pi, Pibar, lefts, rights, node_child1, node_child2, W, reduce_idx,
    log_pD_vec, log_pS_vec, item_idx, *, log_split_probs=None,
    n_eq1=None, eq1_reduce_idx=None, ge2_ptr=None, ge2_parent_ids=None,
    ge2_max_fanout=None, active_parent_rows=None, item_offset=0,
):
    N = int(lefts.shape[0])
    S = int(Pi.shape[1])
    out = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype)
    if N == 0:
        return out
    if log_split_probs is None:
        log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        log_split_probs = log_split_probs.reshape(N).contiguous()
    if n_eq1 is None:
        n_eq1 = N
        eq1_reduce_idx = reduce_idx
        ge2_parent_ids = reduce_idx[:0]
        ge2_ptr = reduce_idx.new_zeros((1,), dtype=torch.long)
        ge2_max_fanout = 0

    by_state = log_pD_vec.ndim == 2 and int(log_pD_vec.shape[1]) != 1
    row_stride = 0 if int(log_pD_vec.shape[0]) == 1 else int(log_pD_vec.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    active = active_parent_rows if active_parent_rows is not None else reduce_idx

    if int(n_eq1) > 0:
        _dts_eq1_kernel[(int(n_eq1), triton.cdiv(S, block_s))](
            Pi, Pibar, lefts, rights, node_child1, node_child2, log_pD_vec, log_pS_vec,
            log_split_probs, eq1_reduce_idx, active, out, item_idx, int(item_offset),
            S, BLOCK_S=block_s, ROW_STRIDE=row_stride, BY_STATE=bool(by_state),
            USE_ACTIVE=bool(active_parent_rows is not None),
            DTYPE=_tl_float_dtype(Pi.dtype),
        )

    if ge2_parent_ids is None or int(ge2_parent_ids.numel()) == 0:
        return out
    tile_splits = 64
    if ge2_max_fanout is None:
        ge2_max_fanout = int((ge2_ptr[1:] - ge2_ptr[:-1]).max().item())
    max_tiles = max(1, triton.cdiv(int(ge2_max_fanout), tile_splits))
    n_groups = int(ge2_parent_ids.numel())
    partial_shape = (n_groups * max_tiles, S)
    partial_max = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    partial_sum = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    _dts_ge2_stage1_kernel[(n_groups, max_tiles, triton.cdiv(S, block_s))](
        Pi, Pibar, lefts, rights, node_child1, node_child2, log_pD_vec, log_pS_vec,
        log_split_probs, ge2_ptr, ge2_parent_ids, active, partial_max, partial_sum,
        item_idx, int(item_offset), split_offset=int(n_eq1), MAX_TILES=max_tiles,
        S=S, TILE_SPLITS=tile_splits, BLOCK_S=block_s, ROW_STRIDE=row_stride,
        BY_STATE=bool(by_state), USE_ACTIVE=bool(active_parent_rows is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    _dts_ge2_stage2_kernel[(n_groups, triton.cdiv(S, block_s))](
        ge2_ptr, ge2_parent_ids, active, partial_max, partial_sum, out,
        MAX_TILES=max_tiles, S=S, TILE_SPLITS=tile_splits, BLOCK_S=block_s,
        USE_ACTIVE=bool(active_parent_rows is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    return out
