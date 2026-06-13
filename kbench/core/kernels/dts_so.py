"""Second-order contraction of the dts cross-wave backward (for the analytic exact-Hessian HVP).

Per split (l, r, parent w) the dts backward forms (wave_backward.py:1868-2285, at fixed adjoint v)
``vd_k = v[w,s] * w_k`` with ``w_k = 2^{lsp + d_k - Pi_parent[w,s]}`` and the five terms
d0=log_pD+Pi_l+Pi_r, d1=Pi_l+Pibar_r, d2=Pi_r+Pibar_l, d3=log_pS+Pi_l[c1]+Pi_r[c2],
d4=log_pS+Pi_r[c1]+Pi_l[c2]; it scatters Pi cotangents, stages pibar cotangents
``ud_l = vd2 * 2^{rm_l+mt_l-Pibar_l}`` / ``ud_r = vd1 * 2^{rm_r+mt_r-Pibar_r}``, and accumulates
parameter cotangents. The pibar-tree VJP then routes ``contrib[j] = p'_c[j] * (A - sub[j])`` into
the child rows (p'_c = 2^{(col+)Pi_child - rm_child}, A = sum ud, sub = subtree-or-self sums).

This module computes the tangent of all of that AT FIXED v along (dPi, dPibar, dlog_pD, dlog_pS,
dmc): kernel 1 (split-parallel) emits the d(vd) scatters into a global d_rhs buffer, stages
(ud, d_ud) per side, and accumulates d(param) terms; kernel 2 (side-parallel) does the tree part
``d(contrib) = dp'(A - sub) + p'(dA - dsub)`` with the same atomic path walks. The parent-Pi
normalizer is NOT frozen (w_k genuinely depends on it: dw_k = ln2 w_k (dd_k - dPi_parent));
the saved row maxes ARE frozen (normalizer invariance).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kbench.core.kernels.dts_fused import _load_rate, _tl_float_dtype


@triton.jit
def _dts_split_so_kernel(
    Pi, dPi, Pibar, dPibar, v_ptr,
    lefts, rights, node_child1, node_child2,
    log_pD, log_pS, dlog_pD, dlog_pS, mt_ptr, dmt_ptr,
    log_split_probs, reduce_idx, item_idx, item_offset, ws,
    pibar_row_max_ptr,
    d_rhs_ptr,
    ud_l_ptr, ud_r_ptr, dud_l_ptr, dud_r_ptr,
    d_grad_pD_ptr, d_grad_pS_ptr, d_grad_mt_ptr,
    S: tl.constexpr, BLOCK_S: tl.constexpr, ROW_STRIDE: tl.constexpr,
    BY_STATE: tl.constexpr, MT_ROW_STRIDE: tl.constexpr, DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    NEG = -float("inf")
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)
    parent_w = tl.load(reduce_idx + n).to(tl.int64)
    item = tl.load(item_idx + item_offset + parent_w).to(tl.int64)
    left = tl.load(lefts + n).to(tl.int64)
    right = tl.load(rights + n).to(tl.int64)
    base_l = left * S
    base_r = right * S
    base_p = (ws + parent_w) * S

    pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG)
    pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG)
    dpi_l = tl.load(dPi + base_l + s_offs, mask=mask, other=0.0)
    dpi_r = tl.load(dPi + base_r + s_offs, mask=mask, other=0.0)
    pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG)
    pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG)
    dpibar_l = tl.load(dPibar + base_l + s_offs, mask=mask, other=0.0)
    dpibar_r = tl.load(dPibar + base_r + s_offs, mask=mask, other=0.0)
    pi_p = tl.load(Pi + base_p + s_offs, mask=mask, other=NEG)
    dpi_p = tl.load(dPi + base_p + s_offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + parent_w * S + s_offs, mask=mask, other=0.0)

    log_d = _load_rate(log_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    log_s = _load_rate(log_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_d = _load_rate(dlog_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_s = _load_rate(dlog_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    mt = tl.load(mt_ptr + item * MT_ROW_STRIDE + s_offs, mask=mask, other=0.0)
    dmt = tl.load(dmt_ptr + item * MT_ROW_STRIDE + s_offs, mask=mask, other=0.0)

    c1 = tl.load(node_child1 + s_offs, mask=mask, other=S)
    c2 = tl.load(node_child2 + s_offs, mask=mask, other=S)
    c1v = mask & (c1 < S)
    c2v = mask & (c2 < S)
    pi_l_c1 = tl.load(Pi + base_l + c1, mask=c1v, other=NEG)
    pi_r_c2 = tl.load(Pi + base_r + c2, mask=c2v, other=NEG)
    pi_r_c1 = tl.load(Pi + base_r + c1, mask=c1v, other=NEG)
    pi_l_c2 = tl.load(Pi + base_l + c2, mask=c2v, other=NEG)
    dpi_l_c1 = tl.load(dPi + base_l + c1, mask=c1v, other=0.0)
    dpi_r_c2 = tl.load(dPi + base_r + c2, mask=c2v, other=0.0)
    dpi_r_c1 = tl.load(dPi + base_r + c1, mask=c1v, other=0.0)
    dpi_l_c2 = tl.load(dPi + base_l + c2, mask=c2v, other=0.0)
    lsp = tl.load(log_split_probs + n)

    d0 = lsp + log_d + pi_l + pi_r
    d1 = lsp + pi_l + pibar_r
    d2 = lsp + pi_r + pibar_l
    d3 = lsp + log_s + pi_l_c1 + pi_r_c2
    d4 = lsp + log_s + pi_r_c1 + pi_l_c2
    dd0 = dlog_d + dpi_l + dpi_r
    dd1 = dpi_l + dpibar_r
    dd2 = dpi_r + dpibar_l
    dd3 = dlog_s + dpi_l_c1 + dpi_r_c2
    dd4 = dlog_s + dpi_r_c1 + dpi_l_c2

    fin = mask & (pi_p != NEG)
    w0 = tl.where(fin, tl.exp2(d0 - pi_p), zero)
    w1 = tl.where(fin, tl.exp2(d1 - pi_p), zero)
    w2 = tl.where(fin, tl.exp2(d2 - pi_p), zero)
    w3 = tl.where(fin, tl.exp2(d3 - pi_p), zero)
    w4 = tl.where(fin, tl.exp2(d4 - pi_p), zero)
    vd1 = v * w1
    vd2 = v * w2
    dvd0 = v * LN2 * w0 * (dd0 - dpi_p)
    dvd1 = v * LN2 * w1 * (dd1 - dpi_p)
    dvd2 = v * LN2 * w2 * (dd2 - dpi_p)
    dvd3 = v * LN2 * w3 * (dd3 - dpi_p)
    dvd4 = v * LN2 * w4 * (dd4 - dpi_p)

    # tangent of the rhs scatters (same targets as the primal)
    tl.atomic_add(d_rhs_ptr + base_l + s_offs, dvd0 + dvd1, sem="relaxed", mask=mask)
    tl.atomic_add(d_rhs_ptr + base_r + s_offs, dvd0 + dvd2, sem="relaxed", mask=mask)
    tl.atomic_add(d_rhs_ptr + base_l + c1, dvd3, sem="relaxed", mask=c1v)
    tl.atomic_add(d_rhs_ptr + base_r + c1, dvd4, sem="relaxed", mask=c1v)
    tl.atomic_add(d_rhs_ptr + base_r + c2, dvd3, sem="relaxed", mask=c2v)
    tl.atomic_add(d_rhs_ptr + base_l + c2, dvd4, sem="relaxed", mask=c2v)

    # pibar staging: ud = vd * 2^{rm + mt - Pibar} (rm frozen), d(ud) = dvd*f + vd*ln2*f*(dmt - dPibar)
    rm_l = tl.load(pibar_row_max_ptr + left).to(DTYPE)
    rm_r = tl.load(pibar_row_max_ptr + right).to(DTYPE)
    fl_ok = mask & (pibar_l != NEG)
    fr_ok = mask & (pibar_r != NEG)
    f_l = tl.where(fl_ok, tl.exp2(rm_l + mt - pibar_l), zero)
    f_r = tl.where(fr_ok, tl.exp2(rm_r + mt - pibar_r), zero)
    ud_l = vd2 * f_l
    ud_r = vd1 * f_r
    dud_l = dvd2 * f_l + vd2 * LN2 * f_l * (dmt - dpibar_l)
    dud_r = dvd1 * f_r + vd1 * LN2 * f_r * (dmt - dpibar_r)
    tl.store(ud_l_ptr + n * S + s_offs, ud_l, mask=mask)
    tl.store(ud_r_ptr + n * S + s_offs, ud_r, mask=mask)
    tl.store(dud_l_ptr + n * S + s_offs, dud_l, mask=mask)
    tl.store(dud_r_ptr + n * S + s_offs, dud_r, mask=mask)

    # parameter tangents (same buckets as the primal accumulations)
    tl.atomic_add(d_grad_pD_ptr + item * S + s_offs, dvd0, sem="relaxed", mask=mask)
    tl.atomic_add(d_grad_pS_ptr + item * S + s_offs, dvd3 + dvd4, sem="relaxed", mask=mask)
    tl.atomic_add(d_grad_mt_ptr + item * S + s_offs, dvd1 + dvd2, sem="relaxed", mask=mask)


@triton.jit
def _dts_tree_so_kernel(
    Pi_ptr, dPi_ptr, col_log_probs_ptr,
    ud_ptr, dud_ptr, A_ptr, dA_ptr,
    sl_ptr, sr_ptr,
    pibar_row_max_ptr,
    compact_level_ptr, compact_level_parent_ptr,
    compact_level_child1_ptr, compact_level_child2_ptr,
    d_rhs_ptr, d_grad_col_ptr,
    n_ws: tl.constexpr, S: tl.constexpr, stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr, N_LEVELS: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr, DTYPE: tl.constexpr,
):
    """Second-order tangent of the pibar-tree contrib, replacing the host parent-chain walk.

    Bottom-up subtree-or-self accumulation (compact per-level, like the primal
    ``_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel``) on BOTH ``ud`` -> ``sub`` and
    ``dud`` -> ``dsub`` in place, then scatters
    ``d_contrib = dp'(A - sub) + p'(dA - dsub)`` (dp' = ln2 p' dPi, p' frozen row-max) into
    d_rhs at the child rows and into d_grad_col.
    """
    LN2 = 0.6931471805599453
    NEG = -float("inf")
    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws
    child_l = tl.load(sl_ptr + split_i).to(tl.int64)
    child_r = tl.load(sr_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)

    pi_base = child * stride_C
    row_base = row * S
    rm = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    rm_safe = tl.where(rm != NEG, rm, tl.zeros_like(rm))
    A = tl.load(A_ptr + row).to(DTYPE)
    dA = tl.load(dA_ptr + row).to(DTYPE)

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
            pv = node_mask & (parent >= 0) & (parent < S)
            c1v = node_mask & (c1 >= 0) & (c1 < S)
            c2v = node_mask & (c2 >= 0) & (c2 < S)
            pval = tl.load(ud_ptr + row_base + parent, mask=pv, other=0.0)
            c1val = tl.load(ud_ptr + row_base + c1, mask=c1v, other=0.0)
            c2val = tl.load(ud_ptr + row_base + c2, mask=c2v, other=0.0)
            tl.store(ud_ptr + row_base + parent, pval + c1val + c2val, mask=pv)
            dpval = tl.load(dud_ptr + row_base + parent, mask=pv, other=0.0)
            dc1 = tl.load(dud_ptr + row_base + c1, mask=c1v, other=0.0)
            dc2 = tl.load(dud_ptr + row_base + c2, mask=c2v, other=0.0)
            tl.store(dud_ptr + row_base + parent, dpval + dc1 + dc2, mask=pv)
            p_start += BLOCK_S
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
        dpi_val = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
        if USE_COL_WEIGHTS:
            col_logp = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG)
            p_prime = tl.exp2(col_logp + pi_val - rm_safe)
        else:
            p_prime = tl.exp2(pi_val - rm_safe)
        p_prime = tl.where(pi_val != NEG, p_prime, tl.zeros_like(p_prime))
        dp_prime = LN2 * p_prime * dpi_val
        sub = tl.load(ud_ptr + row_base + s_offs, mask=mask, other=0.0)
        dsub = tl.load(dud_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = dp_prime * (A - sub) + p_prime * (dA - dsub)
        tl.atomic_add(d_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)
        tl.atomic_add(d_grad_col_ptr + s_offs, contrib, sem="relaxed", mask=mask)


def dts_backward_so(
    Pi, dPi, Pibar, dPibar, v, ws, meta, S,
    log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param, mc_item, dmc_item,
    col_log_probs, node_child1, node_child2, node_parent, max_ancestor_depth,
    pibar_row_max, item_idx,
    d_rhs, d_grad_pD, d_grad_pS, d_grad_mt, d_grad_col,
    *, compact_level_ptr=None, compact_level_parents=None,
    compact_level_child1=None, compact_level_child2=None,
    use_col_weights=False,
):
    """Second-order contraction of the dts backward + pibar tree at fixed adjoint v.

    Accumulates into d_rhs [C,S] (tangent of the cross-wave rhs scatters and tree contribs) and
    the d_grad_* buffers in-place. ``meta`` is the wave's layout dict (sl, sr, reduce_idx, ...).
    """
    sl, sr = meta["sl"], meta["sr"]
    N = int(sl.numel())
    if N == 0:
        return
    lsp = meta.get("log_split_probs")
    if lsp is None:
        lsp = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        lsp = lsp.reshape(N).contiguous()
    by_state = log_pD_param.ndim == 2 and int(log_pD_param.shape[1]) != 1
    row_stride = 0 if int(log_pD_param.shape[0]) == 1 else int(log_pD_param.stride(0))
    mt_row_stride = 0 if int(mc_item.shape[0]) == 1 else int(mc_item.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    dev, dt = Pi.device, Pi.dtype

    # stacked staging: rows [0:N) = left side, [N:2N) = right side (contiguous views, so the
    # split kernel writes them via the same n*S offsets); the tree kernel walks all 2N rows.
    ud = torch.empty((2 * N, S), device=dev, dtype=dt)
    dud = torch.empty((2 * N, S), device=dev, dtype=dt)
    ud_l, ud_r = ud[:N], ud[N:]
    dud_l, dud_r = dud[:N], dud[N:]

    _dts_split_so_kernel[(N, triton.cdiv(S, block_s))](
        Pi, dPi, Pibar, dPibar, v,
        sl, sr, node_child1, node_child2,
        log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param, mc_item, dmc_item,
        lsp, meta["reduce_idx"], item_idx, int(meta["start"]), int(ws),
        pibar_row_max,
        d_rhs, ud_l, ud_r, dud_l, dud_r,
        d_grad_pD, d_grad_pS, d_grad_mt,
        S, BLOCK_S=block_s, ROW_STRIDE=row_stride, BY_STATE=bool(by_state),
        MT_ROW_STRIDE=mt_row_stride, DTYPE=_tl_float_dtype(Pi.dtype),
    )

    # tree part: bottom-up subtree-or-self sums for both ud->sub and dud->dsub in one fused
    # kernel (mirrors the primal compact level-walk), replacing the host parent-chain index_add
    # loop (which dominated the HVP at ~30% via per-level launches + .any() host syncs).
    if compact_level_ptr is None:
        raise ValueError("dts_backward_so requires compact_level_* tables for the tree kernel")
    A = ud.sum(dim=1).contiguous()    # totals BEFORE the in-place subtree accumulation
    dA = dud.sum(dim=1).contiguous()
    n_levels = int(compact_level_ptr.numel()) - 1
    _dts_tree_so_kernel[(2 * N,)](
        Pi, dPi, col_log_probs,
        ud, dud, A, dA, sl, sr,
        pibar_row_max,
        compact_level_ptr.contiguous(), compact_level_parents.contiguous(),
        compact_level_child1.contiguous(), compact_level_child2.contiguous(),
        d_rhs, d_grad_col,
        n_ws=N, S=S, stride_C=int(Pi.stride(0)),
        BLOCK_S=block_s, N_LEVELS=n_levels,
        USE_COL_WEIGHTS=bool(use_col_weights), DTYPE=_tl_float_dtype(Pi.dtype),
        num_warps=4,
    )
