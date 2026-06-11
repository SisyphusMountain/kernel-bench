import os

import torch
import triton
import triton.language as tl

# Experiment knobs (swept via env, defaults = current best).
_WALK_MODE = int(os.environ.get("KBENCH_WS_WALK", "4"))  # 0=levels 1=doubling 2=fused-expw 3=hops 4=register-gather
_WS_WARPS = int(os.environ.get("KBENCH_WS_WARPS", "8"))
_WS_BLOCK_S = int(os.environ.get("KBENCH_WS_BLOCK_S", "512"))  # tile width cap
_WS_MAXNREG = int(os.environ.get("KBENCH_WS_MAXNREG", "0"))  # 0 = compiler default
_WS_REG_MIN_S = int(os.environ.get("KBENCH_WS_REG_MIN_S", "0"))  # >0: pristine kernel below this S


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _prepare_wave_launch(S: int, const_tensor) -> tuple[int, int]:
    const_row_stride = 0 if int(const_tensor.shape[0]) == 1 else int(const_tensor.stride(0))
    return int(min(256, triton.next_power_of_2(S))), const_row_stride


# Binary-lifting jump tables for the ancestor path-sum, derived once from node_parent
# and cached: jump[k][s] is the ancestor 2^k steps above s (-1 past the root). With them
# pathsum[s] = sum of expw over ancestors-or-self computes in ceil(log2(depth)) full-width
# rounds instead of a per-column depth-deep pointer chase. Keyed by the parent tensor's
# storage; node_parent lives in the static problem data, so its data_ptr is stable for
# the lifetime of the capture.
_JUMPS_CACHE: dict = {}
# Ping-pong per-row fp scratch for the path-sum rounds. Keyed by (device, S, dtype).
_PATHSUM_SCRATCH: dict = {}


def _get_jumps(node_parent: torch.Tensor, S: int):
    key = (node_parent.data_ptr(), int(S), node_parent.device.index)
    hit = _JUMPS_CACHE.get(key)
    if hit is not None:
        return hit
    parent = node_parent.detach().to("cpu", torch.int64).tolist()[:S]
    j0 = [p if 0 <= p < S else -1 for p in parent]
    jumps = [j0]
    while any(j >= 0 for j in jumps[-1]):
        prev = jumps[-1]
        jumps.append([prev[prev[s]] if prev[s] >= 0 else -1 for s in range(S)])
    if len(jumps) > 1 and all(j < 0 for j in jumps[-1]):
        jumps.pop()  # last table is all -1 -> that round would be a no-op
    k_rounds = len(jumps)
    device = node_parent.device
    entry = (
        torch.tensor(jumps, dtype=torch.int32, device=device).contiguous(),
        k_rounds,
    )
    _JUMPS_CACHE[key] = entry
    return entry


_LEVELS_CACHE: dict = {}


def _get_levels(node_parent: torch.Tensor, S: int):
    """Top-down level schedule: per level d>=1, scratch[s] += scratch[parent[s]]."""
    key = (node_parent.data_ptr(), int(S), node_parent.device.index)
    hit = _LEVELS_CACHE.get(key)
    if hit is not None:
        return hit
    parent = node_parent.detach().to("cpu", torch.int64).tolist()[:S]
    depth = [-1] * S
    for s in range(S):
        cur, chain = s, []
        while 0 <= cur < S and depth[cur] < 0:
            chain.append(cur)
            cur = parent[cur]
        base = depth[cur] + 1 if 0 <= cur < S else 0
        for i, node in enumerate(reversed(chain)):
            depth[node] = base + i
    n_levels = max(depth) + 1
    by_level: list = [[] for _ in range(n_levels)]
    for s in range(S):
        by_level[depth[s]].append(s)
    lvl_nodes, lvl_parent, lvl_ptr = [], [], [0]
    for d in range(1, n_levels):
        for s in by_level[d]:
            lvl_nodes.append(s)
            lvl_parent.append(parent[s])
        lvl_ptr.append(len(lvl_nodes))
    device = node_parent.device
    max_width = max((lvl_ptr[i + 1] - lvl_ptr[i] for i in range(len(lvl_ptr) - 1)), default=1)
    entry = (
        torch.tensor(lvl_nodes, dtype=torch.int32, device=device),
        torch.tensor(lvl_parent, dtype=torch.int32, device=device),
        torch.tensor(lvl_ptr, dtype=torch.int64, device=device),
        len(lvl_ptr) - 1,
        int(min(256, triton.next_power_of_2(max(max_width, 1)))),
    )
    _LEVELS_CACHE[key] = entry
    return entry


_HOPS_CACHE: dict = {}
_MAX_HOPS = 8


def _get_hop_schedule(node_parent: torch.Tensor, S: int, block_nodes: int = 256):
    """Round schedule for the hop walk: levels are greedily grouped into rounds of at
    most ``block_nodes`` nodes spanning at most ``_MAX_HOPS`` levels. A node at depth d
    in a round starting at depth d0 sums expw over its ancestors at depths [d0, d]
    (its "hops", at most _MAX_HOPS) plus the already-computed path sum of its ancestor
    at depth d0-1 (its "anchor")."""
    key = (node_parent.data_ptr(), int(S), node_parent.device.index, int(block_nodes))
    hit = _HOPS_CACHE.get(key)
    if hit is not None:
        return hit
    parent = node_parent.detach().to("cpu", torch.int64).tolist()[:S]
    depth = [-1] * S
    for s in range(S):
        cur, chain = s, []
        while 0 <= cur < S and depth[cur] < 0:
            chain.append(cur)
            cur = parent[cur]
        base = depth[cur] + 1 if 0 <= cur < S else 0
        for i, node in enumerate(reversed(chain)):
            depth[node] = base + i
    n_levels = max(depth) + 1
    by_level: list = [[] for _ in range(n_levels)]
    for s in range(S):
        by_level[depth[s]].append(s)

    rounds = []  # list of (d0, [levels d0..d1])
    d = 0
    while d < n_levels:
        d0, cum = d, 0
        levels = []
        while d < n_levels and (not levels or (cum + len(by_level[d]) <= block_nodes
                                               and d - d0 < _MAX_HOPS)):
            levels.append(d)
            cum += len(by_level[d])
            d += 1
        rounds.append((d0, levels))

    rnd_nodes, rnd_anchor, rnd_ptr = [], [], [0]
    hop_lists = []
    for d0, levels in rounds:
        for dl in levels:
            for s in by_level[dl]:
                hops = []
                cur = s
                for _ in range(dl - d0 + 1):
                    hops.append(cur)
                    cur = parent[cur] if 0 <= parent[cur] < S else -1
                anchor = cur  # ancestor at depth d0-1, or -1 if d0 == 0
                rnd_nodes.append(s)
                rnd_anchor.append(anchor if anchor is not None else -1)
                hop_lists.append(hops + [-1] * (_MAX_HOPS - len(hops)))
        rnd_ptr.append(len(rnd_nodes))
    n_total = len(rnd_nodes)
    # hops transposed [MAX_HOPS, N] so each unrolled hop loads contiguously
    hops_t = [[hop_lists[i][h] for i in range(n_total)] for h in range(_MAX_HOPS)]
    device = node_parent.device
    entry = (
        torch.tensor(rnd_nodes, dtype=torch.int32, device=device),
        torch.tensor(rnd_anchor, dtype=torch.int32, device=device),
        torch.tensor(hops_t, dtype=torch.int32, device=device).contiguous(),
        torch.tensor(rnd_ptr, dtype=torch.int64, device=device),
        len(rounds),
        n_total,
    )
    _HOPS_CACHE[key] = entry
    return entry


_LEVELS_ALL_CACHE: dict = {}


def _get_levels_with_roots(node_parent: torch.Tensor, S: int):
    """Level schedule including depth-0 roots (parent = -1), for the fused walk."""
    key = (node_parent.data_ptr(), int(S), node_parent.device.index)
    hit = _LEVELS_ALL_CACHE.get(key)
    if hit is not None:
        return hit
    parent = node_parent.detach().to("cpu", torch.int64).tolist()[:S]
    depth = [-1] * S
    for s in range(S):
        cur, chain = s, []
        while 0 <= cur < S and depth[cur] < 0:
            chain.append(cur)
            cur = parent[cur]
        base = depth[cur] + 1 if 0 <= cur < S else 0
        for i, node in enumerate(reversed(chain)):
            depth[node] = base + i
    n_levels = max(depth) + 1
    by_level: list = [[] for _ in range(n_levels)]
    for s in range(S):
        by_level[depth[s]].append(s)
    lvl_nodes, lvl_parent, lvl_ptr = [], [], [0]
    for d in range(n_levels):
        for s in by_level[d]:
            lvl_nodes.append(s)
            lvl_parent.append(parent[s] if 0 <= parent[s] < S else -1)
        lvl_ptr.append(len(lvl_nodes))
    device = node_parent.device
    max_width = max(len(lv) for lv in by_level)
    entry = (
        torch.tensor(lvl_nodes, dtype=torch.int32, device=device),
        torch.tensor(lvl_parent, dtype=torch.int32, device=device),
        torch.tensor(lvl_ptr, dtype=torch.int64, device=device),
        n_levels,
        int(min(256, triton.next_power_of_2(max(max_width, 1)))),
    )
    _LEVELS_ALL_CACHE[key] = entry
    return entry


def _get_pathsum_scratch(W: int, S: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, int(S), dtype)
    bufs = _PATHSUM_SCRATCH.get(key)
    if bufs is None or bufs[0].shape[0] < W:
        bufs = (
            torch.empty((int(W), int(S)), device=device, dtype=dtype),
            torch.empty((int(W), int(S)), device=device, dtype=dtype),
        )
        _PATHSUM_SCRATCH[key] = bufs
    return bufs


@triton.jit
def _row_logsumexp(
    Pi_ptr,
    col_log_probs_ptr,
    base,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    row_max = tl.full([1], value=NEG_INF, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + base + s_offs, mask=mask, other=NEG_INF)
        if USE_COL_WEIGHTS:
            col_logp = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG_INF)
            weighted_pi = col_logp + pi_val
        else:
            weighted_pi = pi_val
        new_max = tl.maximum(row_max, tl.max(weighted_pi, axis=0))
        new_max_safe = tl.where(new_max != NEG_INF, new_max, tl.zeros_like(new_max))
        previous = tl.where(
            row_max != NEG_INF,
            row_sum * tl.exp2(row_max - new_max_safe),
            tl.zeros_like(row_sum),
        )
        current = tl.sum(tl.exp2(weighted_pi - new_max_safe), axis=0)
        row_sum = previous + current
        row_max = new_max
    return row_max, row_sum


@triton.jit
def _pibar_tile(
    Pi_ptr,
    col_log_probs_ptr,
    base,
    s_offs,
    mask,
    row_max,
    row_sum,
    max_coupling,
    node_parent_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
    row_max_safe = tl.where(row_max != NEG_INF, row_max, tl.zeros_like(row_max))
    cur = s_offs.to(tl.int64)
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        cur_valid = mask & (cur >= 0) & (cur < S)
        pi_anc = tl.load(Pi_ptr + base + cur, mask=cur_valid, other=NEG_INF)
        if USE_COL_WEIGHTS:
            col_logp_anc = tl.load(col_log_probs_ptr + cur, mask=cur_valid, other=NEG_INF)
            ancestor_sum += tl.where(
                cur_valid,
                tl.exp2(col_logp_anc + pi_anc - row_max_safe),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
        else:
            ancestor_sum += tl.where(
                cur_valid,
                tl.exp2(pi_anc - row_max_safe),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
        cur = tl.load(node_parent_ptr + cur, mask=cur_valid, other=-1).to(tl.int64)
    denom = row_sum - ancestor_sum
    return tl.where(denom > 0.0, tl.log2(denom) + row_max + max_coupling, NEG_INF)


@triton.jit
def _store_expw(
    Pi_ptr,
    col_log_probs_ptr,
    base,
    scratch_ptr,
    scratch_base,
    row_max,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    row_max_safe = tl.where(row_max != NEG_INF, row_max, tl.zeros_like(row_max))
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + base + s_offs, mask=mask, other=NEG_INF)
        if USE_COL_WEIGHTS:
            col_logp = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG_INF)
            weighted = col_logp + pi_val
        else:
            weighted = pi_val
        expw = tl.exp2(weighted - row_max_safe)
        tl.store(scratch_ptr + scratch_base + s_offs, expw, mask=mask)


@triton.jit
def _row_max_only(
    Pi_ptr,
    col_log_probs_ptr,
    base,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    row_max = tl.full([1], value=NEG_INF, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + base + s_offs, mask=mask, other=NEG_INF)
        if USE_COL_WEIGHTS:
            col_logp = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG_INF)
            weighted_pi = col_logp + pi_val
        else:
            weighted_pi = pi_val
        row_max = tl.maximum(row_max, tl.max(weighted_pi, axis=0))
    return row_max


@triton.jit
def _pathsum_walk_fused(
    Pi_ptr,
    col_log_probs_ptr,
    base,
    scratch_ptr,
    scratch_base,
    row_max,
    lvl_nodes_ptr,
    lvl_parent_ptr,
    lvl_ptr_ptr,
    N_LEVELS: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Level walk that computes expw on the fly (no separate expw sweep) and returns
    # row_sum as a by-product. The schedule includes depth-0 roots (parent = -1).
    NEG_INF: tl.constexpr = -float("inf")
    row_max_safe = tl.where(row_max != NEG_INF, row_max, tl.zeros_like(row_max))
    row_sum = tl.zeros([1], dtype=DTYPE)
    for level in range(0, N_LEVELS):
        start = tl.load(lvl_ptr_ptr + level)
        end = tl.load(lvl_ptr_ptr + level + 1)
        pos = start
        while pos < end:
            offs = pos + tl.arange(0, BLOCK_NODES)
            m = offs < end
            node = tl.load(lvl_nodes_ptr + offs, mask=m, other=0)
            par = tl.load(lvl_parent_ptr + offs, mask=m, other=-1)
            pi_n = tl.load(Pi_ptr + base + node, mask=m, other=NEG_INF)
            if USE_COL_WEIGHTS:
                col_n = tl.load(col_log_probs_ptr + node, mask=m, other=NEG_INF)
                weighted = col_n + pi_n
            else:
                weighted = pi_n
            expw = tl.exp2(weighted - row_max_safe)
            pvalid = m & (par >= 0)
            ps_par = tl.load(scratch_ptr + scratch_base + par, mask=pvalid, other=0.0).to(DTYPE)
            tl.store(scratch_ptr + scratch_base + node, expw + ps_par, mask=m)
            row_sum += tl.sum(tl.where(m, expw, tl.zeros([BLOCK_NODES], dtype=DTYPE)), axis=0)
            pos += BLOCK_NODES
        tl.debug_barrier()
    return row_sum


@triton.jit
def _pathsum_walk(
    scratch_ptr,
    scratch_base,
    lvl_nodes_ptr,
    lvl_parent_ptr,
    lvl_ptr_ptr,
    N_LEVELS: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Turns per-node expw in scratch into ancestor-or-self path sums, one level at a
    # time top-down: scratch[s] += scratch[parent[s]]. Scratch rows are program-private;
    # barriers only order the program's own warps.
    tl.debug_barrier()
    for level in range(0, N_LEVELS):
        start = tl.load(lvl_ptr_ptr + level)
        end = tl.load(lvl_ptr_ptr + level + 1)
        pos = start
        while pos < end:
            offs = pos + tl.arange(0, BLOCK_NODES)
            m = offs < end
            node = tl.load(lvl_nodes_ptr + offs, mask=m, other=0)
            par = tl.load(lvl_parent_ptr + offs, mask=m, other=0)
            ps_par = tl.load(scratch_ptr + scratch_base + par, mask=m, other=0.0).to(DTYPE)
            ps_node = tl.load(scratch_ptr + scratch_base + node, mask=m, other=0.0).to(DTYPE)
            tl.store(scratch_ptr + scratch_base + node, ps_node + ps_par, mask=m)
            pos += BLOCK_NODES
        tl.debug_barrier()


@triton.jit
def _pathsum_hops(
    expw_ptr,
    psum_ptr,
    scratch_base,
    rnd_nodes_ptr,
    rnd_anchor_ptr,
    rnd_hops_ptr,
    rnd_ptr_ptr,
    N_TOTAL: tl.constexpr,
    N_ROUNDS: tl.constexpr,
    MAX_HOPS: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Hop walk: expw (read-only, in expw_ptr) was stored by _store_expw; path sums are
    # written to psum_ptr. Each round handles a group of consecutive tree levels: a node
    # gathers its <=MAX_HOPS in-group ancestors' expw (independent loads) plus one
    # anchor path sum from a previous round. Adding masked zeros is exact in fp, so the
    # fixed unroll changes nothing numerically.
    tl.debug_barrier()
    for r in range(0, N_ROUNDS):
        start = tl.load(rnd_ptr_ptr + r)
        end = tl.load(rnd_ptr_ptr + r + 1)
        pos = start
        while pos < end:
            offs = pos + tl.arange(0, BLOCK_NODES)
            m = offs < end
            node = tl.load(rnd_nodes_ptr + offs, mask=m, other=0)
            anchor = tl.load(rnd_anchor_ptr + offs, mask=m, other=-1)
            avalid = m & (anchor >= 0)
            acc = tl.load(psum_ptr + scratch_base + anchor, mask=avalid, other=0.0).to(DTYPE)
            for h_rev in tl.static_range(MAX_HOPS):
                h = MAX_HOPS - 1 - h_rev
                hid = tl.load(rnd_hops_ptr + h * N_TOTAL + offs, mask=m, other=-1)
                hvalid = m & (hid >= 0)
                ew = tl.load(expw_ptr + scratch_base + hid, mask=hvalid, other=0.0).to(DTYPE)
                acc += ew
            tl.store(psum_ptr + scratch_base + node, acc, mask=m)
            pos += BLOCK_NODES
        tl.debug_barrier()


@triton.jit
def _pathsum_doubling(
    buf_a_ptr,
    buf_b_ptr,
    base,
    jump_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    K_ROUNDS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Turns per-node expw in buf_a into ancestor-or-self path sums by binary lifting:
    # round k adds the running sum of the 2^k-step ancestor. Full-width tiles, one
    # barrier per round. The scratch rows are program-private; barriers only order the
    # program's own warps. Result lands in buf_a if K_ROUNDS is even, else buf_b.
    tl.debug_barrier()
    for k in tl.static_range(K_ROUNDS):
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            jmp = tl.load(jump_ptr + k * S + s_offs, mask=mask, other=-1)
            jvalid = mask & (jmp >= 0)
            if k % 2 == 0:
                self_val = tl.load(buf_a_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
                anc_val = tl.load(buf_a_ptr + base + jmp, mask=jvalid, other=0.0).to(DTYPE)
                tl.store(buf_b_ptr + base + s_offs, self_val + anc_val, mask=mask)
            else:
                self_val = tl.load(buf_b_ptr + base + s_offs, mask=mask, other=0.0).to(DTYPE)
                anc_val = tl.load(buf_b_ptr + base + jmp, mask=jvalid, other=0.0).to(DTYPE)
                tl.store(buf_a_ptr + base + s_offs, self_val + anc_val, mask=mask)
        tl.debug_barrier()


@triton.jit
def _wave_step_kernel(
    Pi_ptr,
    ws,
    pi_ws,
    max_coupling_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    col_log_probs_ptr,
    node_child1_ptr, node_child2_ptr,
    node_parent_ptr,
    leaf_state_ptr,
    leaf_logp_ptr,
    item_idx_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    Pibar_out_ptr,
    pibar_row_max_ptr,
    pi_residual_out_ptr,
    psum_a_ptr,
    psum_b_ptr,
    psum_out_ptr,
    jump_ptr,
    lvl_nodes_ptr,
    lvl_parent_ptr,
    lvl_ptr_ptr,
    rnd_nodes_ptr,
    rnd_anchor_ptr,
    rnd_hops_ptr,
    rnd_ptr_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    K_ROUNDS: tl.constexpr,
    N_LEVELS: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    HOP_N_TOTAL: tl.constexpr,
    HOP_N_ROUNDS: tl.constexpr,
    MAX_HOPS: tl.constexpr,
    WALK_MODE: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_FINAL_PIBAR: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_LARGE = -float("inf")

    w = tl.program_id(0)
    pi_base = (pi_ws + w) * stride
    global_base = (ws + w) * stride
    out_base = w * stride
    scratch_base = w * S
    item_const = tl.load(item_idx_ptr + ws + w)
    const_base = item_const * CONST_ROW_STRIDE

    if WALK_MODE == 2:
        row_max = _row_max_only(
            Pi_ptr, col_log_probs_ptr, pi_base, S, BLOCK_S, USE_COL_WEIGHTS, DTYPE
        )
        row_sum = _pathsum_walk_fused(
            Pi_ptr, col_log_probs_ptr, pi_base, psum_a_ptr, scratch_base, row_max,
            lvl_nodes_ptr, lvl_parent_ptr, lvl_ptr_ptr,
            N_LEVELS, BLOCK_NODES, USE_COL_WEIGHTS, DTYPE,
        )
    else:
        row_max, row_sum = _row_logsumexp(
            Pi_ptr, col_log_probs_ptr, pi_base, S, BLOCK_S, USE_COL_WEIGHTS, DTYPE
        )
        _store_expw(
            Pi_ptr, col_log_probs_ptr, pi_base, psum_a_ptr, scratch_base, row_max,
            S, BLOCK_S, USE_COL_WEIGHTS, DTYPE,
        )
        if WALK_MODE == 1:
            _pathsum_doubling(
                psum_a_ptr, psum_b_ptr, scratch_base, jump_ptr, S, BLOCK_S, K_ROUNDS, DTYPE,
            )
        elif WALK_MODE == 3:
            _pathsum_hops(
                psum_a_ptr, psum_b_ptr, scratch_base,
                rnd_nodes_ptr, rnd_anchor_ptr, rnd_hops_ptr, rnd_ptr_ptr,
                HOP_N_TOTAL, HOP_N_ROUNDS, MAX_HOPS, BLOCK_NODES, DTYPE,
            )
        else:
            _pathsum_walk(
                psum_a_ptr, scratch_base, lvl_nodes_ptr, lvl_parent_ptr, lvl_ptr_ptr,
                N_LEVELS, BLOCK_NODES, DTYPE,
            )

    if COMPUTE_DIFF:
        row_max_diff = tl.zeros([1], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        const_offsets = const_base + s_offs
        max_coupling = tl.load(max_coupling_ptr + const_offsets, mask=mask, other=0.0)
        ancestor_sum = tl.load(psum_out_ptr + scratch_base + s_offs, mask=mask, other=0.0).to(DTYPE)
        denom = row_sum - ancestor_sum
        pibar_w = tl.where(denom > 0.0, tl.log2(denom) + row_max + max_coupling, NEG_LARGE)

        dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

        c1 = tl.load(node_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(node_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_ptr + pi_base + c1, mask=mask & c1_valid, other=NEG_LARGE)
        pi_s2 = tl.load(Pi_ptr + pi_base + c2, mask=mask & c2_valid, other=NEG_LARGE)

        t0 = dl_const + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_const + pi_s1
        t4 = sl2_const + pi_s2
        if USE_LEAF_INDEX:
            leaf_state = tl.load(leaf_state_ptr + ws + w)
            leaf_hit = mask & (leaf_state == s_offs)
            leaf_logp = tl.load(leaf_logp_ptr + item_const * S + s_offs, mask=mask, other=NEG_LARGE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            t5 = tl.full([BLOCK_S], value=NEG_LARGE, dtype=DTYPE)

        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)
        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
            m = tl.maximum(m, dts_r)

        m_safe = tl.where(m != NEG_LARGE, m, tl.zeros_like(m))
        s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
        s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
        if has_splits:
            s += tl.exp2(dts_r - m_safe)

        result = tl.log2(s) + m
        tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

        if COMPUTE_DIFF:

            finite = mask & (result != NEG_LARGE) & (pi_w != NEG_LARGE)
            diff = tl.where(finite, tl.abs(result - pi_w), tl.zeros_like(result))
            tile_max = tl.max(diff, axis=0).to(tl.float32)
            row_max_diff = tl.maximum(row_max_diff, tile_max)

    if COMPUTE_DIFF:
        tl.store(pi_residual_out_ptr + ws + w, tl.max(row_max_diff, axis=0))

    if STORE_FINAL_PIBAR:
        # Order all warps' Pi_new stores and their reads of the first pathsum before
        # rereading Pi_new and overwriting the scratch.
        tl.debug_barrier()
        if WALK_MODE == 2:
            final_row_max = _row_max_only(
                Pi_new_ptr, col_log_probs_ptr, out_base, S, BLOCK_S, USE_COL_WEIGHTS, DTYPE
            )
            final_row_sum = _pathsum_walk_fused(
                Pi_new_ptr, col_log_probs_ptr, out_base, psum_a_ptr, scratch_base, final_row_max,
                lvl_nodes_ptr, lvl_parent_ptr, lvl_ptr_ptr,
                N_LEVELS, BLOCK_NODES, USE_COL_WEIGHTS, DTYPE,
            )
        else:
            final_row_max, final_row_sum = _row_logsumexp(
                Pi_new_ptr, col_log_probs_ptr, out_base, S, BLOCK_S, USE_COL_WEIGHTS, DTYPE
            )
            _store_expw(
                Pi_new_ptr, col_log_probs_ptr, out_base, psum_a_ptr, scratch_base, final_row_max,
                S, BLOCK_S, USE_COL_WEIGHTS, DTYPE,
            )
            if WALK_MODE == 1:
                _pathsum_doubling(
                    psum_a_ptr, psum_b_ptr, scratch_base, jump_ptr, S, BLOCK_S, K_ROUNDS, DTYPE,
                )
            elif WALK_MODE == 3:
                _pathsum_hops(
                    psum_a_ptr, psum_b_ptr, scratch_base,
                    rnd_nodes_ptr, rnd_anchor_ptr, rnd_hops_ptr, rnd_ptr_ptr,
                    HOP_N_TOTAL, HOP_N_ROUNDS, MAX_HOPS, BLOCK_NODES, DTYPE,
                )
            else:
                _pathsum_walk(
                    psum_a_ptr, scratch_base, lvl_nodes_ptr, lvl_parent_ptr, lvl_ptr_ptr,
                    N_LEVELS, BLOCK_NODES, DTYPE,
                )
        tl.store(pibar_row_max_ptr + ws + w, tl.max(final_row_max, axis=0))

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            const_offsets = const_base + s_offs
            max_coupling = tl.load(max_coupling_ptr + const_offsets, mask=mask, other=0.0)
            ancestor_sum = tl.load(psum_out_ptr + scratch_base + s_offs, mask=mask, other=0.0).to(DTYPE)
            denom = final_row_sum - ancestor_sum
            pibar_w = tl.where(denom > 0.0, tl.log2(denom) + final_row_max + max_coupling, NEG_LARGE)
            tl.store(Pibar_out_ptr + global_base + s_offs, pibar_w, mask=mask)


@triton.jit
def _wave_step_kernel_classic(
    Pi_ptr,
    ws,
    pi_ws,
    max_coupling_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    col_log_probs_ptr,
    node_child1_ptr, node_child2_ptr,
    node_parent_ptr,
    leaf_state_ptr,
    leaf_logp_ptr,
    item_idx_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    Pibar_out_ptr,
    pibar_row_max_ptr,
    pi_residual_out_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_FINAL_PIBAR: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_LARGE = -float("inf")

    w = tl.program_id(0)
    pi_base = (pi_ws + w) * stride
    global_base = (ws + w) * stride
    out_base = w * stride
    item_const = tl.load(item_idx_ptr + ws + w)
    const_base = item_const * CONST_ROW_STRIDE

    row_max, row_sum = _row_logsumexp(
        Pi_ptr, col_log_probs_ptr, pi_base, S, BLOCK_S, USE_COL_WEIGHTS, DTYPE
    )

    if COMPUTE_DIFF:
        row_max_diff = tl.zeros([1], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        const_offsets = const_base + s_offs
        max_coupling = tl.load(max_coupling_ptr + const_offsets, mask=mask, other=0.0)
        pibar_w = _pibar_tile(
            Pi_ptr, col_log_probs_ptr, pi_base, s_offs, mask, row_max, row_sum,
            max_coupling, node_parent_ptr, S, BLOCK_S, MAX_ANCESTOR_DEPTH, USE_COL_WEIGHTS, DTYPE,
        )

        dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

        c1 = tl.load(node_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(node_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_ptr + pi_base + c1, mask=mask & c1_valid, other=NEG_LARGE)
        pi_s2 = tl.load(Pi_ptr + pi_base + c2, mask=mask & c2_valid, other=NEG_LARGE)

        t0 = dl_const + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_const + pi_s1
        t4 = sl2_const + pi_s2
        if USE_LEAF_INDEX:
            leaf_state = tl.load(leaf_state_ptr + ws + w)
            leaf_hit = mask & (leaf_state == s_offs)
            leaf_logp = tl.load(leaf_logp_ptr + item_const * S + s_offs, mask=mask, other=NEG_LARGE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            t5 = tl.full([BLOCK_S], value=NEG_LARGE, dtype=DTYPE)

        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)
        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
            m = tl.maximum(m, dts_r)

        m_safe = tl.where(m != NEG_LARGE, m, tl.zeros_like(m))
        s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
        s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
        if has_splits:
            s += tl.exp2(dts_r - m_safe)

        result = tl.log2(s) + m
        tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

        if COMPUTE_DIFF:

            finite = mask & (result != NEG_LARGE) & (pi_w != NEG_LARGE)
            diff = tl.where(finite, tl.abs(result - pi_w), tl.zeros_like(result))
            tile_max = tl.max(diff, axis=0).to(tl.float32)
            row_max_diff = tl.maximum(row_max_diff, tile_max)

    if COMPUTE_DIFF:
        tl.store(pi_residual_out_ptr + ws + w, tl.max(row_max_diff, axis=0))

    if STORE_FINAL_PIBAR:
        final_row_max, final_row_sum = _row_logsumexp(
            Pi_new_ptr, col_log_probs_ptr, out_base, S, BLOCK_S, USE_COL_WEIGHTS, DTYPE
        )
        tl.store(pibar_row_max_ptr + ws + w, tl.max(final_row_max, axis=0))

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            const_offsets = const_base + s_offs
            max_coupling = tl.load(max_coupling_ptr + const_offsets, mask=mask, other=0.0)
            pibar_w = _pibar_tile(
                Pi_new_ptr, col_log_probs_ptr, out_base, s_offs, mask, final_row_max, final_row_sum,
                max_coupling, node_parent_ptr, S, BLOCK_S, MAX_ANCESTOR_DEPTH, USE_COL_WEIGHTS, DTYPE,
            )
            tl.store(Pibar_out_ptr + global_base + s_offs, pibar_w, mask=mask)


@triton.jit
def _wave_step_kernel_reg(
    Pi_ptr,
    ws,
    pi_ws,
    max_coupling_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    col_log_probs_ptr,
    node_child1_ptr, node_child2_ptr,
    leaf_state_ptr,
    leaf_logp_ptr,
    item_idx_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    Pibar_out_ptr,
    pibar_row_max_ptr,
    pi_residual_out_ptr,
    jump_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    K_ROUNDS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_FINAL_PIBAR: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Register-resident wave step: the whole row lives in one BLOCK_S tile; the
    ancestor path-sum runs as binary lifting with tl.gather (no scratch, no barriers)."""
    NEG_LARGE = -float("inf")

    w = tl.program_id(0)
    pi_base = (pi_ws + w) * stride
    global_base = (ws + w) * stride
    out_base = w * stride
    item_const = tl.load(item_idx_ptr + ws + w)
    const_base = item_const * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
    if USE_COL_WEIGHTS:
        colw = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG_LARGE)
        weighted = colw + pi_w
    else:
        weighted = pi_w
    row_max = tl.max(weighted, axis=0)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, 0.0)
    expw = tl.exp2(weighted - row_max_safe)
    row_sum = tl.sum(expw, axis=0)

    val = expw
    for k in tl.static_range(K_ROUNDS):
        jmp = tl.load(jump_ptr + k * S + s_offs, mask=mask, other=-1)
        jv = jmp >= 0
        g = tl.gather(val, tl.where(jv, jmp, 0), axis=0)
        val += tl.where(jv, g, 0.0)
    ancestor_sum = val

    const_offsets = const_base + s_offs
    max_coupling = tl.load(max_coupling_ptr + const_offsets, mask=mask, other=0.0)
    denom = row_sum - ancestor_sum
    pibar_w = tl.where(denom > 0.0, tl.log2(denom) + row_max + max_coupling, NEG_LARGE)

    dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

    c1 = tl.load(node_child1_ptr + s_offs, mask=mask, other=0)
    c2 = tl.load(node_child2_ptr + s_offs, mask=mask, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.where(mask & c1_valid, tl.gather(pi_w, tl.where(c1_valid, c1, 0), axis=0), NEG_LARGE)
    pi_s2 = tl.where(mask & c2_valid, tl.gather(pi_w, tl.where(c2_valid, c2, 0), axis=0), NEG_LARGE)

    t0 = dl_const + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_const + pi_s1
    t4 = sl2_const + pi_s2
    if USE_LEAF_INDEX:
        leaf_state = tl.load(leaf_state_ptr + ws + w)
        leaf_hit = mask & (leaf_state == s_offs)
        leaf_logp = tl.load(leaf_logp_ptr + item_const * S + s_offs, mask=mask, other=NEG_LARGE)
        t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
    else:
        t5 = tl.full([BLOCK_S], value=NEG_LARGE, dtype=DTYPE)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    if has_splits:
        dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
        m = tl.maximum(m, dts_r)

    m_safe = tl.where(m != NEG_LARGE, m, tl.zeros_like(m))
    s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
    s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
    if has_splits:
        s += tl.exp2(dts_r - m_safe)

    result = tl.log2(s) + m
    tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

    if COMPUTE_DIFF:
        finite = mask & (result != NEG_LARGE) & (pi_w != NEG_LARGE)
        diff = tl.where(finite, tl.abs(result - pi_w), tl.zeros_like(result))
        tl.store(pi_residual_out_ptr + ws + w, tl.max(diff, axis=0).to(tl.float32))

    if STORE_FINAL_PIBAR:
        # Pi_new row is still in registers as `result`; no reload, no barrier.
        if USE_COL_WEIGHTS:
            fweighted = colw + result
        else:
            fweighted = result
        fweighted = tl.where(mask, fweighted, NEG_LARGE)
        final_row_max = tl.max(fweighted, axis=0)
        final_row_max_safe = tl.where(final_row_max != NEG_LARGE, final_row_max, 0.0)
        fexpw = tl.exp2(fweighted - final_row_max_safe)
        final_row_sum = tl.sum(fexpw, axis=0)
        tl.store(pibar_row_max_ptr + ws + w, final_row_max)

        fval = fexpw
        for k in tl.static_range(K_ROUNDS):
            jmp = tl.load(jump_ptr + k * S + s_offs, mask=mask, other=-1)
            jv = jmp >= 0
            g = tl.gather(fval, tl.where(jv, jmp, 0), axis=0)
            fval += tl.where(jv, g, 0.0)
        fdenom = final_row_sum - fval
        fpibar = tl.where(fdenom > 0.0, tl.log2(fdenom) + final_row_max + max_coupling, NEG_LARGE)
        tl.store(Pibar_out_ptr + global_base + s_offs, fpibar, mask=mask)


@triton.jit
def _leaf_initial_wave_step_kernel(
    Pi_new_ptr,
    ws,
    max_coupling_ptr,
    DL_const_ptr,
    Ebar_ptr,
    E_ptr,
    SL1_const_ptr,
    SL2_const_ptr,
    col_log_probs_ptr,
    node_child1_ptr,
    node_child2_ptr,
    node_subtree_start_ptr,
    node_subtree_end_ptr,
    leaf_state_ptr,
    leaf_logp_ptr,
    item_idx_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_LARGE = -float("inf")

    w = tl.program_id(0)
    s_start = tl.program_id(1) * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    out_base = w * stride

    item = tl.load(item_idx_ptr + ws + w)
    const_base = item * CONST_ROW_STRIDE

    leaf_state = tl.load(leaf_state_ptr + ws + w)
    leaf_start = tl.load(node_subtree_start_ptr + leaf_state)
    leaf_end = tl.load(node_subtree_end_ptr + leaf_state)
    state_start = tl.load(node_subtree_start_ptr + s_offs, mask=mask, other=-1)
    descendant = (state_start >= leaf_start) & (state_start < leaf_end)
    leaf_hit = mask & (s_offs == leaf_state)

    const_offsets = const_base + s_offs
    max_coupling = tl.load(max_coupling_ptr + const_offsets, mask=mask, other=0.0)
    if USE_COL_WEIGHTS:
        leaf_col_logp = tl.load(col_log_probs_ptr + leaf_state).to(DTYPE)
    else:
        leaf_col_logp = tl.zeros((), dtype=DTYPE)
    dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

    leaf_obs_logp = tl.load(leaf_logp_ptr + item * S + leaf_state).to(DTYPE)
    pi_w = tl.where(leaf_hit, leaf_obs_logp, NEG_LARGE)
    pibar_w = tl.where(~descendant, max_coupling + leaf_col_logp + leaf_obs_logp, NEG_LARGE)

    c1 = tl.load(node_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(node_child2_ptr + s_offs, mask=mask, other=S)
    pi_s1 = tl.where(mask & (c1 == leaf_state), leaf_obs_logp, NEG_LARGE)
    pi_s2 = tl.where(mask & (c2 == leaf_state), leaf_obs_logp, NEG_LARGE)

    t0 = dl_const + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_const + pi_s1
    t4 = sl2_const + pi_s2
    leaf_logp = tl.load(leaf_logp_ptr + item * S + s_offs, mask=mask, other=NEG_LARGE)
    t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m != NEG_LARGE, m, tl.zeros_like(m))
    total = (
        tl.exp2(t0 - m_safe)
        + tl.exp2(t1 - m_safe)
        + tl.exp2(t2 - m_safe)
        + tl.exp2(t3 - m_safe)
        + tl.exp2(t4 - m_safe)
        + tl.exp2(t5 - m_safe)
    )
    result = tl.log2(total) + m
    tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)


def compute_leaf_initial_wave_step(
    Pi_out,
    ws,
    W,
    S,
    max_coupling_mat,
    DL_const,
    Ebar,
    E,
    SL1_const,
    SL2_const,
    col_log_probs,
    node_child1,
    node_child2,
    node_subtree_start,
    node_subtree_end,
    leaf_state_idx,
    leaf_logp,
    item_idx,
    use_col_weights=True,
):
    block_s, const_row_stride = _prepare_wave_launch(S, DL_const)
    grid = (W, triton.cdiv(S, block_s))
    Pi_out_rows = Pi_out.narrow(0, int(ws), int(W))
    _leaf_initial_wave_step_kernel[grid](
        Pi_out_rows,
        ws,
        max_coupling_mat,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        col_log_probs,
        node_child1,
        node_child2,
        node_subtree_start,
        node_subtree_end,
        leaf_state_idx,
        leaf_logp,
        item_idx,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        USE_COL_WEIGHTS=bool(use_col_weights),
        DTYPE=_tl_float_dtype(Pi_out.dtype),
        num_warps=8,
    )

def compute_wave_step(Pi_in, Pi_out, Pibar, ws, W, S,
                     max_coupling_mat, DL_const, Ebar, E, SL1_const, SL2_const,
                     col_log_probs,
                     node_child1, node_child2, node_parent, max_ancestor_depth,
                     DTS_reduced=None,
                     *,
                     leaf_state_idx, leaf_logp,
                     item_idx,
                     pibar_row_max,
                     store_final_pibar=False,
                     has_leaf_term=True,
                     input_ws=None,
                     use_col_weights=True,
                     pi_residual_out=None):
    has_splits = DTS_reduced is not None
    _, const_row_stride = _prepare_wave_launch(S, DL_const)
    block_s = int(min(_WS_BLOCK_S, triton.next_power_of_2(S)))
    use_leaf_index = bool(has_leaf_term)
    compute_diff = pi_residual_out is not None

    jump_table, k_rounds = _get_jumps(node_parent, S)
    if _WALK_MODE == 2:
        lvl_nodes, lvl_parent, lvl_ptr, n_levels, block_nodes = _get_levels_with_roots(node_parent, S)
    else:
        lvl_nodes, lvl_parent, lvl_ptr, n_levels, block_nodes = _get_levels(node_parent, S)
    rnd_nodes, rnd_anchor, rnd_hops, rnd_ptr, hop_rounds, hop_total = _get_hop_schedule(node_parent, S)
    psum_a, psum_b = _get_pathsum_scratch(W, S, Pi_in.device, Pi_in.dtype)
    if _WALK_MODE == 3:
        psum_out = psum_b
    elif _WALK_MODE == 1 and k_rounds % 2 == 1:
        psum_out = psum_b
    else:
        psum_out = psum_a

    grid = (W,)
    Pi_out_rows = Pi_out.narrow(0, int(ws), int(W))

    if _WALK_MODE == 4 and S < _WS_REG_MIN_S:
        # Small state rows: the pristine per-tile ancestor chase is cheaper than the
        # register-gather machinery (depth and S are both small).
        _wave_step_kernel_classic[grid](
            Pi_in, ws, ws if input_ws is None else int(input_ws),
            max_coupling_mat,
            DL_const, Ebar, E, SL1_const, SL2_const,
            col_log_probs,
            node_child1, node_child2,
            node_parent,
            leaf_state_idx,
            leaf_logp,
            item_idx,
            DTS_reduced if has_splits else Pi_in,
            has_splits,
            Pi_out_rows, Pibar, pibar_row_max,
            pi_residual_out if compute_diff else pibar_row_max,
            S,
            stride=S,
            CONST_ROW_STRIDE=const_row_stride,
            BLOCK_S=int(min(256, triton.next_power_of_2(S))),
            MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
            USE_LEAF_INDEX=use_leaf_index,
            STORE_FINAL_PIBAR=bool(store_final_pibar),
            COMPUTE_DIFF=compute_diff,
            USE_COL_WEIGHTS=bool(use_col_weights),
            DTYPE=_tl_float_dtype(Pi_in.dtype),
            num_warps=8,
        )
        return

    if _WALK_MODE == 4:
        _wave_step_kernel_reg[grid](
            Pi_in, ws, ws if input_ws is None else int(input_ws),
            max_coupling_mat,
            DL_const, Ebar, E, SL1_const, SL2_const,
            col_log_probs,
            node_child1, node_child2,
            leaf_state_idx,
            leaf_logp,
            item_idx,
            DTS_reduced if has_splits else Pi_in,
            has_splits,
            Pi_out_rows, Pibar, pibar_row_max,
            pi_residual_out if compute_diff else pibar_row_max,
            jump_table,
            S,
            stride=S,
            CONST_ROW_STRIDE=const_row_stride,
            BLOCK_S=int(triton.next_power_of_2(S)),
            K_ROUNDS=k_rounds,
            USE_LEAF_INDEX=use_leaf_index,
            STORE_FINAL_PIBAR=bool(store_final_pibar),
            COMPUTE_DIFF=compute_diff,
            USE_COL_WEIGHTS=bool(use_col_weights),
            DTYPE=_tl_float_dtype(Pi_in.dtype),
            num_warps=_WS_WARPS,
            **({"maxnreg": _WS_MAXNREG} if _WS_MAXNREG else {}),
        )
        return

    _wave_step_kernel[grid](
        Pi_in, ws, ws if input_ws is None else int(input_ws),
        max_coupling_mat,
        DL_const, Ebar, E, SL1_const, SL2_const,
        col_log_probs,
        node_child1, node_child2,
        node_parent,
        leaf_state_idx,
        leaf_logp,
        item_idx,
        DTS_reduced if has_splits else Pi_in,
        has_splits,
        Pi_out_rows, Pibar, pibar_row_max,
        pi_residual_out if compute_diff else pibar_row_max,
        psum_a,
        psum_b,
        psum_out,
        jump_table,
        lvl_nodes,
        lvl_parent,
        lvl_ptr,
        rnd_nodes,
        rnd_anchor,
        rnd_hops,
        rnd_ptr,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        K_ROUNDS=k_rounds,
        N_LEVELS=n_levels,
        BLOCK_NODES=256 if _WALK_MODE == 3 else block_nodes,
        HOP_N_TOTAL=hop_total,
        HOP_N_ROUNDS=hop_rounds,
        MAX_HOPS=_MAX_HOPS,
        WALK_MODE=_WALK_MODE,
        USE_LEAF_INDEX=use_leaf_index,
        STORE_FINAL_PIBAR=bool(store_final_pibar),
        COMPUTE_DIFF=compute_diff,
        USE_COL_WEIGHTS=bool(use_col_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=_WS_WARPS,
    )
