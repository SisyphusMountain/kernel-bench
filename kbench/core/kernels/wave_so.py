"""Second-order contraction of the wave-step adjoint (for the analytic exact-Hessian HVP).

Given a wave's primals (Pi*, Pibar*, dts_r, consts, saved pibar_row_max), the converged cached
adjoint ``v`` and tangent states (dPi, dPibar, d_dts, dconsts), computes at FIXED v:

  * ``d(A^T v)``  — tangent of the self-loop Jacobian-transpose application (seed correction for
    the reused solve:  dv = (I-A^T)^{-1} [d_rhs + d(A^T)v]);
  * ``d(B^T v)``  — tangents of the parameter-cotangent buckets (d_aw0..d_aw4, d_aw345).

Derivation (see newton/_wave_so_notes.md): with terms t0..t5 (+ dts_r), weights
``w~_k = 2^{t_k - lse}``, left-branch weight ``w_L = 2^{lse - Pi_new}``, and the pibar routing
``A^T``-block ``pibar_u_coeff_s * p_prime_j`` (j not in path(s)):

  dw~_k = ln2 * w~_k * (dt_k - dlse),     dlse = sum_j w~_j dt_j
  dw_L  = ln2 * w_L * (1 - w_L) * (dlse - d_dts_r)
  dp'   = ln2 * p' * dPi                  (pibar_row_max FROZEN: outputs normalizer-invariant)
  ddenom_s = drow_sum - dancsum_s,        d(inv_denom) = -inv_denom^2 * ddenom
  d(A^T v)[j] = v_j d(diag_wt_j) + dp'_j (A - sub_j) + p'_j (dA - dsub_j)
              + sum_{s: c1[s]=j} v_s d(sl1_wt_s) + sum_{s: c2[s]=j} v_s d(sl2_wt_s)
  d_aw_k = v * (dw_L * w~_k + w_L * dw~_k)

with u_d = v * pibar_u_coeff, A = sum u_d, sub_j = subtree-or-self sum of u_d (transpose of the
ancestor walk -> atomic path scatter), and du_d/dA/dsub their tangents. All log2-space: the lse
derivative carries no ln2; the pibar block's ln2 factors cancel exactly (d2^x vs dlog2).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kbench.core.kernels.wave_step import _prepare_wave_launch, _tl_float_dtype


@triton.jit
def _wave_so_kernel(
    Pi_ptr, dPi_ptr, Pibar_ptr, dPibar_ptr,
    v_ptr,
    ws, S: tl.constexpr, stride: tl.constexpr,
    pibar_row_max_ptr,
    mc_ptr, DL_ptr, dDL_ptr, Ebar_ptr, dEbar_ptr, E_ptr, dE_ptr,
    SL1_ptr, dSL1_ptr, SL2_ptr, dSL2_ptr,
    col_log_probs_ptr,
    node_child1_ptr, node_child2_ptr, node_parent_ptr,
    leaf_state_ptr, leaf_logp_ptr, dleaf_logp_ptr,
    item_idx_ptr,
    DTS_ptr, dDTS_ptr, has_splits: tl.constexpr,
    d_out_ptr,
    d_aw0_ptr, d_aw1_ptr, d_aw2_ptr, d_aw345_ptr, d_aw3_ptr, d_aw4_ptr,
    sub_ptr, dsub_ptr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    NEG = -float("inf")
    w = tl.program_id(0)
    pi_base = (ws + w) * stride
    out_base = w * stride
    item_const = tl.load(item_idx_ptr + ws + w)
    const_base = item_const * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
    dpi_w = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
    pibar_w = tl.load(Pibar_ptr + pi_base + s_offs, mask=mask, other=NEG)
    dpibar_w = tl.load(dPibar_ptr + pi_base + s_offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + out_base + s_offs, mask=mask, other=0.0)
    rm = tl.load(pibar_row_max_ptr + ws + w).to(DTYPE)
    rm_safe = tl.where(rm != NEG, rm, tl.zeros((), dtype=DTYPE))

    if USE_COL_WEIGHTS:
        colw = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG)
        p_prime = tl.where(mask, tl.exp2(colw + pi_w - rm_safe), zero)
    else:
        p_prime = tl.where(mask, tl.exp2(pi_w - rm_safe), zero)
    dp_prime = LN2 * p_prime * dpi_w  # rm frozen
    row_sum = tl.sum(p_prime, axis=0)
    drow_sum = tl.sum(dp_prime, axis=0)

    # ancestor-or-self path sums of p_prime (and tangent) along the parent chain
    cur = s_offs
    anc = zero
    danc = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        cv = mask & (cur >= 0) & (cur < S)
        pi_a = tl.load(Pi_ptr + pi_base + cur, mask=cv, other=NEG)
        dpi_a = tl.load(dPi_ptr + pi_base + cur, mask=cv, other=0.0)
        if USE_COL_WEIGHTS:
            col_a = tl.load(col_log_probs_ptr + cur, mask=cv, other=NEG)
            pa = tl.where(cv, tl.exp2(col_a + pi_a - rm_safe), zero)
        else:
            pa = tl.where(cv, tl.exp2(pi_a - rm_safe), zero)
        anc += pa
        danc += LN2 * pa * dpi_a
        cur = tl.load(node_parent_ptr + cur, mask=cv, other=-1).to(tl.int32)

    denom = row_sum - anc
    pos = mask & (denom > 0.0)
    inv_denom = tl.where(pos, 1.0 / tl.where(pos, denom, tl.full([BLOCK_S], 1.0, DTYPE)), zero)
    ddenom = drow_sum - danc

    # terms and weights at the saved primals (mirror the precompute kernel)
    const_offsets = const_base + s_offs
    mc = tl.load(mc_ptr + const_offsets, mask=mask, other=0.0)
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
    c1v = mask & (c1 < S)
    c2v = mask & (c2 < S)
    pi_s1 = tl.where(c1v, tl.gather(pi_w, tl.where(c1v, c1, 0), axis=0), NEG)
    pi_s2 = tl.where(c2v, tl.gather(pi_w, tl.where(c2v, c2, 0), axis=0), NEG)
    dpi_s1 = tl.where(c1v, tl.gather(dpi_w, tl.where(c1v, c1, 0), axis=0), zero)
    dpi_s2 = tl.where(c2v, tl.gather(dpi_w, tl.where(c2v, c2, 0), axis=0), zero)

    t0 = dl + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1 + pi_s1
    t4 = sl2 + pi_s2
    dt0 = ddl + dpi_w
    dt1 = dpi_w + debar
    dt2 = dpibar_w + de
    dt3 = dsl1 + dpi_s1
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
    m_safe = tl.where(m != NEG, m, zero)
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    lsum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(lsum > 0.0, 1.0 / lsum, zero)
    w0 = e0 * inv_sum
    w1 = e1 * inv_sum
    w2 = e2 * inv_sum
    w3 = e3 * inv_sum
    w4 = e4 * inv_sum
    w5 = e5 * inv_sum

    dlse = w0 * dt0 + w1 * dt1 + w2 * dt2 + w3 * dt3 + w4 * dt4 + w5 * dt5
    dw0 = LN2 * w0 * (dt0 - dlse)
    dw1 = LN2 * w1 * (dt1 - dlse)
    dw2 = LN2 * w2 * (dt2 - dlse)
    dw3 = LN2 * w3 * (dt3 - dlse)
    dw4 = LN2 * w4 * (dt4 - dlse)
    dw5 = LN2 * w5 * (dt5 - dlse)

    if has_splits:
        dts_r = tl.load(DTS_ptr + out_base + s_offs, mask=mask, other=NEG)
        d_dts = tl.load(dDTS_ptr + out_base + s_offs, mask=mask, other=0.0)
        dts_l = tl.where(mask & (lsum > 0.0), tl.log2(tl.where(lsum > 0.0, lsum, tl.full([BLOCK_S], 1.0, DTYPE))) + m, tl.full([BLOCK_S], NEG, DTYPE))
        pm = tl.maximum(dts_l, dts_r)
        pm_safe = tl.where(pm != NEG, pm, zero)
        pi_new = tl.log2(tl.exp2(dts_l - pm_safe) + tl.exp2(dts_r - pm_safe)) + pm
        w_L = tl.where(dts_l != NEG, tl.exp2(dts_l - pi_new), zero)
        dw_L = tl.where(mask & (dts_r != NEG) & (dts_l != NEG),
                        LN2 * w_L * (1.0 - w_L) * (dlse - d_dts), zero)
    else:
        w_L = tl.where(mask, tl.full([BLOCK_S], 1.0, DTYPE), zero)
        dw_L = zero

    # parameter-bucket tangents: d_aw_k = v * (dw_L * w~_k + w_L * dw~_k)
    da0 = v * (dw_L * w0 + w_L * dw0)
    da1 = v * (dw_L * w1 + w_L * dw1)
    da2 = v * (dw_L * w2 + w_L * dw2)
    da3 = v * (dw_L * w3 + w_L * dw3)
    da4 = v * (dw_L * w4 + w_L * dw4)
    da5 = v * (dw_L * w5 + w_L * dw5)
    tl.store(d_aw0_ptr + out_base + s_offs, da0, mask=mask)
    tl.store(d_aw1_ptr + out_base + s_offs, da1, mask=mask)
    tl.store(d_aw2_ptr + out_base + s_offs, da2, mask=mask)
    tl.store(d_aw345_ptr + out_base + s_offs, da3 + da4 + da5, mask=mask)
    tl.store(d_aw3_ptr + out_base + s_offs, da3, mask=mask)
    tl.store(d_aw4_ptr + out_base + s_offs, da4, mask=mask)

    # pibar routing: u_d, du_d and their subtree-or-self sums (atomic path scatter)
    pibar_u_coeff = w_L * w2 * inv_denom
    d_pibar_u_coeff = (dw_L * w2 + w_L * dw2) * inv_denom - pibar_u_coeff * inv_denom * ddenom
    u_d = v * pibar_u_coeff
    du_d = v * d_pibar_u_coeff
    A = tl.sum(u_d, axis=0)
    dA = tl.sum(du_d, axis=0)

    tl.store(sub_ptr + out_base + s_offs, zero, mask=mask)
    tl.store(dsub_ptr + out_base + s_offs, zero, mask=mask)
    tl.debug_barrier()
    cur = s_offs
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        cv = mask & (cur >= 0) & (cur < S)
        tl.atomic_add(sub_ptr + out_base + cur, u_d, sem="relaxed", mask=cv)
        tl.atomic_add(dsub_ptr + out_base + cur, du_d, sem="relaxed", mask=cv)
        cur = tl.load(node_parent_ptr + cur, mask=cv, other=-1).to(tl.int32)
    tl.debug_barrier()
    sub = tl.load(sub_ptr + out_base + s_offs, mask=mask, other=0.0)
    dsub = tl.load(dsub_ptr + out_base + s_offs, mask=mask, other=0.0)

    d_diag = dw_L * (w0 + w1) + w_L * (dw0 + dw1)
    d_self = v * d_diag + dp_prime * (A - sub) + p_prime * (dA - dsub)
    tl.store(d_out_ptr + out_base + s_offs, d_self, mask=mask)
    tl.debug_barrier()

    d_sl1 = dw_L * w3 + w_L * dw3
    d_sl2 = dw_L * w4 + w_L * dw4
    tl.atomic_add(d_out_ptr + out_base + c1, v * d_sl1, sem="relaxed", mask=c1v)
    tl.atomic_add(d_out_ptr + out_base + c2, v * d_sl2, sem="relaxed", mask=c2v)


def wave_backward_so(
    Pi_star, dPi, Pibar_star, dPibar, v, ws, W, S,
    pibar_row_max, mc, DL, dDL, Ebar, dEbar, E, dE, SL1, dSL1, SL2, dSL2,
    col_log_probs, node_child1, node_child2, node_parent, max_ancestor_depth,
    dts_r=None, d_dts=None,
    *, leaf_state_idx, leaf_logp, dleaf_logp, item_idx, has_leaf_term=True,
    use_col_weights=False,
):
    """Second-order contraction at fixed adjoint v. Returns
    (d_Av [W,S], d_aw0, d_aw1, d_aw2, d_aw345, d_aw3, d_aw4)."""
    has_splits = dts_r is not None
    _, const_row_stride = _prepare_wave_launch(S, DL)
    block_s = int(triton.next_power_of_2(S))
    dev, dt = Pi_star.device, Pi_star.dtype
    d_out = torch.empty((W, S), device=dev, dtype=dt)
    d_aws = tuple(torch.empty((W, S), device=dev, dtype=dt) for _ in range(6))
    sub = torch.empty((W, S), device=dev, dtype=dt)
    dsub = torch.empty((W, S), device=dev, dtype=dt)
    dummy = Pi_star
    _wave_so_kernel[(int(W),)](
        Pi_star, dPi, Pibar_star, dPibar, v,
        ws, S, S,
        pibar_row_max,
        mc, DL, dDL, Ebar, dEbar, E, dE, SL1, dSL1, SL2, dSL2,
        col_log_probs,
        node_child1, node_child2, node_parent,
        leaf_state_idx, leaf_logp, dleaf_logp,
        item_idx,
        dts_r if has_splits else dummy,
        d_dts if has_splits else dummy,
        has_splits,
        d_out, *d_aws, sub, dsub,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=bool(has_leaf_term),
        USE_COL_WEIGHTS=bool(use_col_weights),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=8,
    )
    return (d_out, *d_aws)
