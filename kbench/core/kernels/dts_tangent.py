"""Forward-mode tangent (Jvp) of the cross-wave dts reduction.

Linearization of ``compute_dts_forward`` (dts_fused.py). For each split (left row ``l``, right row
``r``) feeding parent wave row ``p = reduce_idx[n]``, the 5 terms are
``t0=lsp+log_d+pi_l+pi_r``, ``t1=lsp+pi_l+pibar_r``, ``t2=lsp+pi_r+pibar_l``,
``t3=lsp+log_s+Pi[l,c1]+Pi[r,c2]``, ``t4=lsp+log_s+Pi[r,c1]+Pi[l,c2]``, and
``dts_r[p,s] = logsumexp2`` over *all* of p's splits. The tangent is
``d(dts_r[p,s]) = Σ_{splits of p} Σ_k w_k dt_k`` with ``w_k = exp2(t_k - dts_r[p,s])`` (the
precomputed ``dts_r`` is the normalizer, so no online-softmax pass is needed).

Split-parallel with an atomic scatter into the parent row, so the eq1 (1 split / parent) and ge2
(many splits / parent) cases are handled uniformly via the full ``reduce_idx`` map.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kbench.core.kernels.dts_fused import _load_rate, _tl_float_dtype


@triton.jit
def _dts_tangent_kernel(
    Pi, Pibar, dPi, dPibar, lefts, rights, node_child1, node_child2,
    log_pD, log_pS, dlog_pD, dlog_pS, log_split_probs, reduce_idx,
    dts_r_ptr, d_out_ptr, item_idx, item_offset,
    S: tl.constexpr, BLOCK_S: tl.constexpr, ROW_STRIDE: tl.constexpr,
    BY_STATE: tl.constexpr, DTYPE: tl.constexpr,
):
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

    pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG)
    pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG)
    pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG)
    pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG)
    dpi_l = tl.load(dPi + base_l + s_offs, mask=mask, other=0.0)
    dpi_r = tl.load(dPi + base_r + s_offs, mask=mask, other=0.0)
    dpibar_l = tl.load(dPibar + base_l + s_offs, mask=mask, other=0.0)
    dpibar_r = tl.load(dPibar + base_r + s_offs, mask=mask, other=0.0)

    log_d = _load_rate(log_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    log_s = _load_rate(log_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_d = _load_rate(dlog_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_s = _load_rate(dlog_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)

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

    t0 = lsp + log_d + pi_l + pi_r
    t1 = lsp + pi_l + pibar_r
    t2 = lsp + pi_r + pibar_l
    t3 = lsp + log_s + pi_l_c1 + pi_r_c2
    t4 = lsp + log_s + pi_r_c1 + pi_l_c2
    dt0 = dlog_d + dpi_l + dpi_r
    dt1 = dpi_l + dpibar_r
    dt2 = dpi_r + dpibar_l
    dt3 = dlog_s + dpi_l_c1 + dpi_r_c2
    dt4 = dlog_s + dpi_r_c1 + dpi_l_c2

    dts_out = tl.load(dts_r_ptr + parent_w * S + s_offs, mask=mask, other=NEG)
    active = mask & (dts_out != NEG)
    w0 = tl.exp2(t0 - dts_out)
    w1 = tl.exp2(t1 - dts_out)
    w2 = tl.exp2(t2 - dts_out)
    w3 = tl.exp2(t3 - dts_out)
    w4 = tl.exp2(t4 - dts_out)
    contrib = w0 * dt0 + w1 * dt1 + w2 * dt2 + w3 * dt3 + w4 * dt4
    contrib = tl.where(active, contrib, zero)
    tl.atomic_add(d_out_ptr + parent_w * S + s_offs, contrib, sem="relaxed", mask=mask)


def compute_dts_tangent(
    Pi, Pibar, dPi, dPibar, lefts, rights, node_child1, node_child2, W, reduce_idx,
    log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec, dts_r, item_idx,
    *, log_split_probs=None, n_eq1=None, eq1_reduce_idx=None,
    ge2_ptr=None, ge2_parent_ids=None, ge2_max_fanout=None, item_offset=0,
):
    """Tangent of compute_dts_forward (eq1 + ge2 via the full reduce_idx map). Returns d_dts [W, S]."""
    N = int(lefts.shape[0])
    S = int(Pi.shape[1])
    d_out = torch.zeros((W, S), device=Pi.device, dtype=Pi.dtype)
    if N == 0:
        return d_out
    if log_split_probs is None:
        log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        log_split_probs = log_split_probs.reshape(N).contiguous()
    by_state = log_pD_vec.ndim == 2 and int(log_pD_vec.shape[1]) != 1
    row_stride = 0 if int(log_pD_vec.shape[0]) == 1 else int(log_pD_vec.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    _dts_tangent_kernel[(N, triton.cdiv(S, block_s))](
        Pi, Pibar, dPi, dPibar, lefts, rights, node_child1, node_child2,
        log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec, log_split_probs, reduce_idx,
        dts_r, d_out, item_idx, int(item_offset),
        S, BLOCK_S=block_s, ROW_STRIDE=row_stride, BY_STATE=bool(by_state),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    return d_out
