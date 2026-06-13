"""Forward-mode tangent (Jvp) of the Pi-wave self-loop step.

Linearization of one ``compute_wave_step`` application (wave_step.py): per row ``w``, state ``s``,
``Pi_new = logsumexp2(t0..t5[,dts])`` with
``t0=dl+Pi[s]``, ``t1=Pi[s]+ebar``, ``t2=pibar[s]+e``, ``t3=sl1+Pi[c1]``, ``t4=sl2+Pi[c2]``,
``t5=leaf``. The tangent is ``dPi_new = Σ_k w_k dt_k`` with ``w_k = exp2(t_k - Pi_new)`` and the
pibar differential ``dpibar[s] = (dRS - dAS[s]) / denom[s] + dmc[s]`` (same ancestor-path structure
as the E-step tangent). ``dt`` forcing comes from the E-step tangent (de, debar, de_s1, de_s2),
the parameter tangents (dlog_pD, dlog_pS, dmax_coupling), and the cross-wave ``dts`` tangent.

Single-tile (whole row in one ``BLOCK_S`` tile, so ``S <= BLOCK_S``); ancestor sums use the parent
chain (``MAX_ANCESTOR_DEPTH``), matching ``_wave_step_kernel_classic``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from kbench.core.kernels.wave_step import _prepare_wave_launch, _tl_float_dtype

# Launch tuning: num_warps for _wave_step_tangent_kernel. Tuned on the representative 666x80 fixture
# (S=1331, BLOCK_S=2048): num_warps=4 is ~7.4% faster on the total HVP than the old default of 8
# (sweep {2,4,8,16,32}; 2 spills catastrophically, 8/16/32 slower). The win is NOT from occupancy —
# both 4 and 8 pin to 8 active warps/SM (16.67%), register+shared-limited — but from more elements
# per thread (16 vs 8 → better ILP) and 2 resident blocks/SM (vs 1) hiding the cold-DRAM latency.
# Bit-identical (hvp FD gate unchanged); neutral on small (S=119). Override per run via env.
_WST_NUM_WARPS = int(os.environ.get("NEWTON_WST_NUM_WARPS", "4"))


@triton.jit
def _wave_step_tangent_kernel(
    Pi_ptr, dPi_ptr,
    ws, pi_ws,
    max_coupling_ptr, dMC_ptr,
    DL_ptr, dDL_ptr,
    Ebar_ptr, dEbar_ptr,
    E_ptr, dE_ptr,
    SL1_ptr, dSL1_ptr,
    SL2_ptr, dSL2_ptr,
    col_log_probs_ptr,
    node_child1_ptr, node_child2_ptr, node_parent_ptr,
    leaf_state_ptr, leaf_logp_ptr, dleaf_logp_ptr,
    item_idx_ptr,
    DTS_ptr, dDTS_ptr,
    has_splits: tl.constexpr,
    dPi_new_ptr,
    dPibar_out_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_PIBAR: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG = -float("inf")
    w = tl.program_id(0)
    pi_base = (pi_ws + w) * stride
    out_base = w * stride
    global_base = (ws + w) * stride
    item_const = tl.load(item_idx_ptr + ws + w)
    const_base = item_const * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
    dpi_w = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
    if USE_COL_WEIGHTS:
        colw = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG)
        weighted = colw + pi_w
    else:
        weighted = pi_w
    row_max = tl.max(weighted, axis=0)
    row_max_safe = tl.where(row_max != NEG, row_max, tl.zeros([1], dtype=DTYPE))
    r = tl.where(mask, tl.exp2(weighted - row_max_safe), zero)
    row_sum = tl.sum(r, axis=0)
    dRS = tl.sum(tl.where(mask, r * dpi_w, zero), axis=0)

    cur = s_offs
    ancestor_sum = zero
    dAS = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        cv = mask & (cur >= 0) & (cur < S)
        pi_anc = tl.load(Pi_ptr + pi_base + cur, mask=cv, other=NEG)
        dpi_anc = tl.load(dPi_ptr + pi_base + cur, mask=cv, other=0.0)
        if USE_COL_WEIGHTS:
            col_anc = tl.load(col_log_probs_ptr + cur, mask=cv, other=NEG)
            r_anc = tl.where(cv, tl.exp2(col_anc + pi_anc - row_max_safe), zero)
        else:
            r_anc = tl.where(cv, tl.exp2(pi_anc - row_max_safe), zero)
        ancestor_sum += r_anc
        dAS += r_anc * dpi_anc
        cur = tl.load(node_parent_ptr + cur, mask=cv, other=-1).to(tl.int32)

    const_offsets = const_base + s_offs
    mc = tl.load(max_coupling_ptr + const_offsets, mask=mask, other=0.0)
    dmc = tl.load(dMC_ptr + const_offsets, mask=mask, other=0.0)
    denom = row_sum - ancestor_sum
    pos = denom > 0.0
    safe_denom = tl.where(pos, denom, tl.full([BLOCK_S], 1.0, DTYPE))
    pibar = tl.where(pos, tl.log2(safe_denom) + row_max + mc, NEG)
    dpibar = tl.where(pos, (dRS - dAS) / safe_denom + dmc, zero)

    dl = tl.load(DL_ptr + const_offsets, mask=mask, other=NEG)
    ddl = tl.load(dDL_ptr + const_offsets, mask=mask, other=0.0)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG)
    debar = tl.load(dEbar_ptr + const_offsets, mask=mask, other=0.0)
    e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG)
    de = tl.load(dE_ptr + const_offsets, mask=mask, other=0.0)
    sl1 = tl.load(SL1_ptr + const_offsets, mask=mask, other=NEG)
    dsl1 = tl.load(dSL1_ptr + const_offsets, mask=mask, other=0.0)
    sl2 = tl.load(SL2_ptr + const_offsets, mask=mask, other=NEG)
    dsl2 = tl.load(dSL2_ptr + const_offsets, mask=mask, other=0.0)

    c1 = tl.load(node_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(node_child2_ptr + s_offs, mask=mask, other=S)
    c1_valid = mask & (c1 < S)
    c2_valid = mask & (c2 < S)
    pi_s1 = tl.where(c1_valid, tl.gather(pi_w, tl.where(c1_valid, c1, 0), axis=0), NEG)
    pi_s2 = tl.where(c2_valid, tl.gather(pi_w, tl.where(c2_valid, c2, 0), axis=0), NEG)
    dpi_s1 = tl.where(c1_valid, tl.gather(dpi_w, tl.where(c1_valid, c1, 0), axis=0), zero)
    dpi_s2 = tl.where(c2_valid, tl.gather(dpi_w, tl.where(c2_valid, c2, 0), axis=0), zero)

    t0 = dl + pi_w
    dt0 = ddl + dpi_w
    t1 = pi_w + ebar
    dt1 = dpi_w + debar
    t2 = pibar + e_val
    dt2 = dpibar + de
    t3 = sl1 + pi_s1
    dt3 = dsl1 + dpi_s1
    t4 = sl2 + pi_s2
    dt4 = dsl2 + dpi_s2
    if USE_LEAF_INDEX:
        leaf_state = tl.load(leaf_state_ptr + ws + w)
        leaf_hit = mask & (leaf_state == s_offs)
        leaf_logp = tl.load(leaf_logp_ptr + item_const * S + s_offs, mask=mask, other=NEG)
        dleaf = tl.load(dleaf_logp_ptr + item_const * S + s_offs, mask=mask, other=0.0)
        t5 = tl.where(leaf_hit, leaf_logp, NEG)
        dt5 = tl.where(leaf_hit, dleaf, zero)
    else:
        t5 = tl.full([BLOCK_S], NEG, dtype=DTYPE)
        dt5 = zero

    m = tl.maximum(tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3)), tl.maximum(t4, t5))
    if has_splits:
        dts_r = tl.load(DTS_ptr + out_base + s_offs, mask=mask, other=NEG)
        ddts = tl.load(dDTS_ptr + out_base + s_offs, mask=mask, other=0.0)
        m = tl.maximum(m, dts_r)
    m_safe = tl.where(m != NEG, m, zero)
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    total = e0 + e1 + e2 + e3 + e4 + e5
    num = e0 * dt0 + e1 * dt1 + e2 * dt2 + e3 * dt3 + e4 * dt4 + e5 * dt5
    if has_splits:
        edts = tl.exp2(dts_r - m_safe)
        total += edts
        num += edts * ddts
    result = tl.log2(total) + m
    inv = tl.where(total > 0.0, 1.0 / total, zero)
    dPi_new = tl.where(mask & (result != NEG), num * inv, zero)
    tl.store(dPi_new_ptr + out_base + s_offs, dPi_new, mask=mask)

    if STORE_PIBAR:
        tl.store(dPibar_out_ptr + global_base + s_offs, dpibar, mask=mask)


def compute_wave_step_tangent(
    Pi_in, dPi_in, dPi_out, ws, W, S,
    max_coupling_mat, dMC, DL, dDL, Ebar, dEbar, E, dE, SL1, dSL1, SL2, dSL2,
    col_log_probs, node_child1, node_child2, node_parent, max_ancestor_depth,
    DTS_reduced=None, dDTS=None,
    *, leaf_state_idx, leaf_logp, dleaf_logp, item_idx,
    dPibar_out=None, has_leaf_term=True, input_ws=None, use_col_weights=True,
):
    """One tangent application of the wave step. Writes dPi_out[ws:ws+W]; optionally dPibar_out."""
    has_splits = DTS_reduced is not None
    _, const_row_stride = _prepare_wave_launch(S, DL)
    block_s = int(triton.next_power_of_2(S))
    store_pibar = dPibar_out is not None
    dPi_out_rows = dPi_out.narrow(0, int(ws), int(W))
    dummy = Pi_in  # unused placeholder for None pointers
    _wave_step_tangent_kernel[(int(W),)](
        Pi_in, dPi_in,
        ws, ws if input_ws is None else int(input_ws),
        max_coupling_mat, dMC,
        DL, dDL, Ebar, dEbar, E, dE, SL1, dSL1, SL2, dSL2,
        col_log_probs,
        node_child1, node_child2, node_parent,
        leaf_state_idx, leaf_logp, dleaf_logp,
        item_idx,
        DTS_reduced if has_splits else dummy,
        dDTS if has_splits else dummy,
        has_splits,
        dPi_out_rows,
        dPibar_out if store_pibar else dummy,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=bool(has_leaf_term),
        STORE_PIBAR=bool(store_pibar),
        USE_COL_WEIGHTS=bool(use_col_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=_WST_NUM_WARPS,
    )
