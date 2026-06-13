"""Second-order contraction of the E-step backward (for the analytic exact-Hessian HVP).

The E-step backward (``_e_step_backward_prepare_2d_kernel`` + finalize, e_step.py) maps primals
``x = (E, E_new, E_s1, E_s2, Ebar, log_pS, log_pD, log_pL[, col])`` and cotangents
``g = (g_new, g_s1, g_s2, g_ebar)`` to ``(grad_E, grad_pS, grad_pD, grad_pL, grad_mc, grad_col)``.
The outputs are LINEAR in ``g`` and nonlinear in ``x``, so their full tangent splits as

    d[bwd(x; g)] = bwd(x; dg)          # existing kernel, applied to tangent cotangents
                 + SO(x; g; dx)        # THIS kernel: (d/dx bwd)(x; g) . dx  at fixed g

Term tangents mirror the primal kernel exactly: ``q_k = g_new * 2^{t_k - E_new}`` gives
``dq_k = ln2 * q_k * (dt_k - dE_new)``; the pibar-tree part (``u = wbar/denom``,
``pibar_vjp = r * (total_u - excluded_u)``) differentiates with the row-max tangent FROZEN —
legitimate because every output is invariant to the row normalizer (the same invariance the
forward tangent kernels rely on, FD-verified at ~1e-10).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kbench.core.kernels.e_step import _tl_float_dtype

_LN2 = 0.6931471805599453


@triton.jit
def _e_step_so_prepare_kernel(
    E_ptr, dE_ptr, E_new_ptr, dE_new_ptr, E_s1_ptr, dE_s1_ptr, E_s2_ptr, dE_s2_ptr,
    Ebar_ptr, dEbar_ptr,
    log_pS_ptr, dlog_pS_ptr, log_pD_ptr, dlog_pD_ptr, log_pL_ptr, dlog_pL_ptr,
    col_log_probs_ptr, dcol_ptr,
    node_parent_ptr, node_child1_ptr, node_child2_ptr,
    g_new_ptr, g_ebar_ptr,
    d_grad_E_ptr, d_grad_pS_ptr, d_grad_pD_ptr, d_grad_pL_ptr, d_grad_mc_ptr,
    r_ptr, dr_ptr, excl_ptr, dexcl_ptr, tot_ptr, dtot_ptr,
    S: tl.constexpr, BLOCK_S: tl.constexpr, MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr, DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    neg_inf = -float("inf")
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    E = tl.load(E_ptr + base + offs, mask=mask, other=neg_inf)
    dE = tl.load(dE_ptr + base + offs, mask=mask, other=0.0)
    E_new = tl.load(E_new_ptr + base + offs, mask=mask, other=neg_inf)
    dE_new = tl.load(dE_new_ptr + base + offs, mask=mask, other=0.0)
    E_s1 = tl.load(E_s1_ptr + base + offs, mask=mask, other=neg_inf)
    dE_s1 = tl.load(dE_s1_ptr + base + offs, mask=mask, other=0.0)
    E_s2 = tl.load(E_s2_ptr + base + offs, mask=mask, other=neg_inf)
    dE_s2 = tl.load(dE_s2_ptr + base + offs, mask=mask, other=0.0)
    Ebar = tl.load(Ebar_ptr + base + offs, mask=mask, other=neg_inf)
    dEbar = tl.load(dEbar_ptr + base + offs, mask=mask, other=0.0)
    pS = tl.load(log_pS_ptr + base + offs, mask=mask, other=neg_inf)
    dpS = tl.load(dlog_pS_ptr + base + offs, mask=mask, other=0.0)
    pD = tl.load(log_pD_ptr + base + offs, mask=mask, other=neg_inf)
    dpD = tl.load(dlog_pD_ptr + base + offs, mask=mask, other=0.0)
    pL = tl.load(log_pL_ptr + base + offs, mask=mask, other=neg_inf)
    dpL = tl.load(dlog_pL_ptr + base + offs, mask=mask, other=0.0)
    g_new = tl.load(g_new_ptr + base + offs, mask=mask, other=0.0)
    g_ebar = tl.load(g_ebar_ptr + base + offs, mask=mask, other=0.0)

    if USE_COL_WEIGHTS:
        col = tl.load(col_log_probs_ptr + offs, mask=mask, other=neg_inf)
        dcol = tl.load(dcol_ptr + offs, mask=mask, other=0.0)
        wE = col + E
        dwE = dcol + dE
    else:
        wE = E
        dwE = dE
    row_max = tl.max(wE, axis=0)
    row_max_safe = tl.where(row_max != neg_inf, row_max, tl.zeros([1], dtype=DTYPE))

    # term tangents (q_k linear in g_new, nonlinear in primals)
    t0 = pS + E_s1 + E_s2
    t1 = pD + 2.0 * E
    t2 = E + Ebar
    t3 = pL
    q0 = tl.where(mask, g_new * tl.exp2(t0 - E_new), zero)
    q1 = tl.where(mask, g_new * tl.exp2(t1 - E_new), zero)
    q2 = tl.where(mask, g_new * tl.exp2(t2 - E_new), zero)
    dt0 = dpS + dE_s1 + dE_s2
    dt1 = dpD + 2.0 * dE
    dt2 = dE + dEbar
    dt3 = dpL
    dq0 = LN2 * q0 * (dt0 - dE_new)
    dq1 = LN2 * q1 * (dt1 - dE_new)
    dq2 = LN2 * q2 * (dt2 - dE_new)
    dq3 = LN2 * tl.where(mask, g_new * tl.exp2(t3 - E_new), zero) * (dt3 - dE_new)

    dwbar = dq2  # g_ebar is a fixed cotangent here
    tl.store(d_grad_pS_ptr + base + offs, dq0, mask=mask)
    tl.store(d_grad_pD_ptr + base + offs, dq1, mask=mask)
    tl.store(d_grad_pL_ptr + base + offs, dq3, mask=mask)
    tl.store(d_grad_mc_ptr + base + offs, dwbar, mask=mask)
    tl.store(d_grad_E_ptr + base + offs, 2.0 * dq1 + dq2, mask=mask)

    r = tl.where(mask, tl.exp2(wE - row_max_safe), zero)
    dr = LN2 * r * dwE  # row_max frozen: outputs are normalizer-invariant
    tl.store(r_ptr + base + offs, r, mask=mask)
    tl.store(dr_ptr + base + offs, dr, mask=mask)
    tl.store(excl_ptr + base + offs, zero, mask=mask)
    tl.store(dexcl_ptr + base + offs, zero, mask=mask)

    # order plain stores before overlapping atomics (same discipline as the primal backward)
    tl.debug_barrier()

    c1 = tl.load(node_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(node_child2_ptr + offs, mask=mask, other=-1)
    c1v = mask & (c1 >= 0) & (c1 < S)
    c2v = mask & (c2 >= 0) & (c2 < S)
    tl.atomic_add(d_grad_E_ptr + base + c1, dq0, sem="relaxed", mask=c1v)
    tl.atomic_add(d_grad_E_ptr + base + c2, dq0, sem="relaxed", mask=c2v)

    row_sum = tl.sum(r, axis=0)
    drow_sum = tl.sum(dr, axis=0)
    cur = offs
    anc = zero
    danc = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (cur >= 0) & (cur < S)
        E_a = tl.load(E_ptr + base + cur, mask=valid, other=neg_inf)
        dE_a = tl.load(dE_ptr + base + cur, mask=valid, other=0.0)
        if USE_COL_WEIGHTS:
            col_a = tl.load(col_log_probs_ptr + cur, mask=valid, other=neg_inf)
            dcol_a = tl.load(dcol_ptr + cur, mask=valid, other=0.0)
            r_a = tl.where(valid, tl.exp2(col_a + E_a - row_max_safe), zero)
            dr_a = LN2 * r_a * (dcol_a + dE_a)
        else:
            r_a = tl.where(valid, tl.exp2(E_a - row_max_safe), zero)
            dr_a = LN2 * r_a * dE_a
        anc += r_a
        danc += dr_a
        cur = tl.load(node_parent_ptr + cur, mask=valid, other=-1).to(tl.int32)

    denom = row_sum - anc
    ddenom = drow_sum - danc
    pos = mask & (denom > 0.0)
    safe = tl.where(pos, denom, tl.full([BLOCK_S], 1.0, DTYPE))
    wbar = q2 + g_ebar
    u = tl.where(pos, wbar / safe, zero)
    du = tl.where(pos, (dwbar - u * ddenom) / safe, zero)
    tl.store(tot_ptr + g, tl.sum(u, axis=0))
    tl.store(dtot_ptr + g, tl.sum(du, axis=0))

    cur = offs
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (cur >= 0) & (cur < S)
        tl.atomic_add(excl_ptr + base + cur, u, sem="relaxed", mask=valid)
        tl.atomic_add(dexcl_ptr + base + cur, du, sem="relaxed", mask=valid)
        cur = tl.load(node_parent_ptr + cur, mask=valid, other=-1).to(tl.int32)


@triton.jit
def _e_step_so_finalize_kernel(
    d_grad_E_ptr, d_grad_col_ptr,
    r_ptr, dr_ptr, excl_ptr, dexcl_ptr, tot_ptr, dtot_ptr,
    S: tl.constexpr, BLOCK_S: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0)
    dr = tl.load(dr_ptr + base + offs, mask=mask, other=0.0)
    excl = tl.load(excl_ptr + base + offs, mask=mask, other=0.0)
    dexcl = tl.load(dexcl_ptr + base + offs, mask=mask, other=0.0)
    tot = tl.load(tot_ptr + g)
    dtot = tl.load(dtot_ptr + g)
    cur = tl.load(d_grad_E_ptr + base + offs, mask=mask, other=0.0)
    d_pv = dr * (tot - excl) + r * (dtot - dexcl)
    tl.store(d_grad_E_ptr + base + offs, cur + d_pv, mask=mask)
    tl.atomic_add(d_grad_col_ptr + offs, d_pv, sem="relaxed", mask=mask)


def e_step_backward_so(
    E, E_new, E_s1, E_s2, Ebar, log_pS, log_pD, log_pL, col_log_probs,
    node_parent, node_child1, node_child2, max_ancestor_depth,
    g_new, g_s1, g_s2, g_ebar,
    dE, dE_new, dE_s1, dE_s2, dEbar, dlog_pS, dlog_pD, dlog_pL, dcol,
    *, use_col_weights=False,
):
    """(d/dx of the e-step backward)(x; g) . dx at fixed cotangents g.

    Returns (d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_mc, d_grad_col). Note g_s1/g_s2
    contribute only linearly (child scatters of fixed cotangents) so they do not appear in the
    contraction; they are accepted for interface symmetry.
    """
    G, S = int(E.shape[0]), int(E.shape[1])
    block_s = int(triton.next_power_of_2(S))
    d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_mc, r, dr, excl, dexcl = (
        torch.empty_like(E) for _ in range(9)
    )
    d_grad_col = torch.zeros_like(col_log_probs)
    tot = torch.empty((G,), dtype=E.dtype, device=E.device)
    dtot = torch.empty((G,), dtype=E.dtype, device=E.device)
    dcol_arg = dcol if dcol is not None else torch.zeros_like(col_log_probs)
    _e_step_so_prepare_kernel[(G,)](
        E, dE, E_new, dE_new, E_s1, dE_s1, E_s2, dE_s2, Ebar, dEbar,
        log_pS, dlog_pS, log_pD, dlog_pD, log_pL, dlog_pL,
        col_log_probs, dcol_arg,
        node_parent, node_child1, node_child2,
        g_new, g_ebar,
        d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_mc,
        r, dr, excl, dexcl, tot, dtot,
        S, BLOCK_S=block_s, MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_COL_WEIGHTS=bool(use_col_weights), DTYPE=_tl_float_dtype(E.dtype),
        num_warps=8,
    )
    _e_step_so_finalize_kernel[(G,)](
        d_grad_E, d_grad_col, r, dr, excl, dexcl, tot, dtot,
        S, BLOCK_S=block_s, num_warps=8,
    )
    return d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_mc, d_grad_col
