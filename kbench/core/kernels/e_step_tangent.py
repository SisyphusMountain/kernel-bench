"""Forward-mode tangent (Jvp) of the E-step fixed point.

Linearization of ``_e_step_forward_2d_kernel`` (e_step.py): given a base fixed point ``E*`` and a
direction in (E, params), produce the directional derivative ``dE_new``. The per-state Jacobian of
the ``E_new = logsumexp2(t0..t3)`` update is the softmax-weighted contraction
``dE_new = Σ_k w_k dt_k`` with ``w_k = exp2(t_k - E_new)`` (the same weights the backward's
``q0..q3`` use, e_step.py:173-176). The ``Ebar`` branch differentiates the stabilized
``log2(row_sum - ancestor_sum)`` to ``(dRS - dAS[s]) / denom + dmc``, where
``dRS = Σ_s r[s] dE[s]`` and ``dAS[s] = Σ_{a in path(s)} r[a] dE[a]`` with ``r = exp2(E - row_max)``.

``e_tangent_fixed_point`` solves the tangent fixed point ``(I - J_E^{EE}) dE* = J_E^{E,p} dp`` by
Richardson iteration (the E-step is a contraction, so its tangent converges at the same rate),
evaluating the Jacobian at the frozen ``E*``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kbench.core.parameters.extract_parameters import as_item_state
from kbench.core.kernels.e_step import _tl_float_dtype


@triton.jit
def _e_step_tangent_2d_kernel(
    E_ptr,
    dE_ptr,
    dE_new_ptr,
    dE_s1_out_ptr,
    dE_s2_out_ptr,
    dEbar_out_ptr,
    max_diff_out_ptr,
    log_pS_ptr,
    log_pD_ptr,
    log_pL_ptr,
    max_coupling_ptr,
    col_log_probs_ptr,
    dlog_pS_ptr,
    dlog_pD_ptr,
    dlog_pL_ptr,
    dmax_coupling_ptr,
    node_parent_ptr,
    node_child1_ptr,
    node_child2_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    neg_inf = -float("inf")
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    E = tl.load(E_ptr + base + offs, mask=mask, other=neg_inf)
    dE = tl.load(dE_ptr + base + offs, mask=mask, other=0.0)
    if USE_COL_WEIGHTS:
        col_logp = tl.load(col_log_probs_ptr + offs, mask=mask, other=neg_inf)
        weighted_E = col_logp + E
    else:
        weighted_E = E
    row_max = tl.max(weighted_E, axis=0)
    row_max_safe = tl.where(row_max != neg_inf, row_max, tl.zeros([1], dtype=DTYPE))
    r = tl.where(mask, tl.exp2(weighted_E - row_max_safe), zero)
    row_sum = tl.sum(r, axis=0)
    dRS = tl.sum(tl.where(mask, r * dE, zero), axis=0)

    cur = offs
    ancestor_sum = zero
    dAS = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        cur_valid = mask & (cur >= 0) & (cur < S)
        E_anc = tl.load(E_ptr + base + cur, mask=cur_valid, other=neg_inf)
        dE_anc = tl.load(dE_ptr + base + cur, mask=cur_valid, other=0.0)
        if USE_COL_WEIGHTS:
            col_anc = tl.load(col_log_probs_ptr + cur, mask=cur_valid, other=neg_inf)
            r_anc = tl.where(cur_valid, tl.exp2(col_anc + E_anc - row_max_safe), zero)
        else:
            r_anc = tl.where(cur_valid, tl.exp2(E_anc - row_max_safe), zero)
        ancestor_sum += r_anc
        dAS += r_anc * dE_anc
        cur = tl.load(node_parent_ptr + cur, mask=cur_valid, other=-1).to(tl.int32)

    c1 = tl.load(node_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(node_child2_ptr + offs, mask=mask, other=-1)
    c1_valid = mask & (c1 >= 0) & (c1 < S)
    c2_valid = mask & (c2 >= 0) & (c2 < S)
    E_s1 = tl.load(E_ptr + base + c1, mask=c1_valid, other=neg_inf)
    E_s2 = tl.load(E_ptr + base + c2, mask=c2_valid, other=neg_inf)
    dE_s1 = tl.load(dE_ptr + base + c1, mask=c1_valid, other=0.0)
    dE_s2 = tl.load(dE_ptr + base + c2, mask=c2_valid, other=0.0)

    mc = tl.load(max_coupling_ptr + base + offs, mask=mask, other=0.0)
    dmc = tl.load(dmax_coupling_ptr + base + offs, mask=mask, other=0.0)
    denom = row_sum - ancestor_sum
    pos = denom > 0.0
    Ebar = tl.where(pos, tl.log2(tl.where(pos, denom, tl.full([BLOCK_S], 1.0, DTYPE))) + row_max + mc, neg_inf)
    dEbar = tl.where(pos, (dRS - dAS) / tl.where(pos, denom, tl.full([BLOCK_S], 1.0, DTYPE)) + dmc, zero)

    pS = tl.load(log_pS_ptr + base + offs, mask=mask, other=neg_inf)
    pD = tl.load(log_pD_ptr + base + offs, mask=mask, other=neg_inf)
    pL = tl.load(log_pL_ptr + base + offs, mask=mask, other=neg_inf)
    dpS = tl.load(dlog_pS_ptr + base + offs, mask=mask, other=0.0)
    dpD = tl.load(dlog_pD_ptr + base + offs, mask=mask, other=0.0)
    dpL = tl.load(dlog_pL_ptr + base + offs, mask=mask, other=0.0)

    t0 = pS + E_s1 + E_s2
    t1 = pD + 2.0 * E
    t2 = E + Ebar
    t3 = pL
    m = tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3))
    m_safe = tl.where(m == neg_inf, zero, m)
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    total = e0 + e1 + e2 + e3
    E_new = tl.log2(total) + m
    inv_total = tl.where(total > 0.0, 1.0 / total, zero)
    w0 = e0 * inv_total
    w1 = e1 * inv_total
    w2 = e2 * inv_total
    w3 = e3 * inv_total

    dt0 = dpS + dE_s1 + dE_s2
    dt1 = dpD + 2.0 * dE
    dt2 = dE + dEbar
    dt3 = dpL
    dE_new = w0 * dt0 + w1 * dt1 + w2 * dt2 + w3 * dt3
    dE_new = tl.where(mask & (E_new != neg_inf), dE_new, zero)

    tl.store(dE_new_ptr + base + offs, dE_new, mask=mask)
    tl.store(dE_s1_out_ptr + base + offs, dE_s1, mask=mask)
    tl.store(dE_s2_out_ptr + base + offs, dE_s2, mask=mask)
    tl.store(dEbar_out_ptr + base + offs, dEbar, mask=mask)
    if COMPUTE_DIFF:
        diff = tl.where(mask, tl.abs(dE_new - dE), zero)
        tl.store(max_diff_out_ptr + g, tl.max(diff, axis=0))


def _launch_e_step_tangent_2d(
    E, dE, log_pS_mat, log_pD_mat, log_pL_mat, max_coupling_mat, col_log_probs,
    dlog_pS_mat, dlog_pD_mat, dlog_pL_mat, dmax_coupling_mat,
    node_parent, node_child1, node_child2, max_ancestor_depth,
    *, out=None, max_diff_out=None, use_col_weights=True,
):
    G = int(E.shape[0])
    S = int(E.shape[1])
    block_s = int(triton.next_power_of_2(S))
    dE_new, dE_s1, dE_s2, dEbar = (torch.empty_like(E) for _ in range(4)) if out is None else out
    _e_step_tangent_2d_kernel[(G,)](
        E, dE, dE_new, dE_s1, dE_s2, dEbar,
        dE_new if max_diff_out is None else max_diff_out,
        log_pS_mat, log_pD_mat, log_pL_mat, max_coupling_mat, col_log_probs,
        dlog_pS_mat, dlog_pD_mat, dlog_pL_mat, dmax_coupling_mat,
        node_parent, node_child1, node_child2,
        S,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        COMPUTE_DIFF=max_diff_out is not None,
        USE_COL_WEIGHTS=bool(use_col_weights),
        DTYPE=_tl_float_dtype(E.dtype),
        num_warps=8,
    )
    return dE_new, dE_s1, dE_s2, dEbar


def e_tangent_fixed_point(
    E_star,
    dlog_pS, dlog_pD, dlog_pL, dmax_coupling,
    log_pS, log_pD, log_pL, max_coupling, col_log_probs,
    node_parent, node_child1, node_child2, max_ancestor_depth,
    *, max_iter=2000, tol=1e-9, use_col_weights=True, dE0=None,
):
    """Solve (I - J_E^EE) dE* = J_E^Ep dp at the frozen E*; return (dE*, dE_s1, dE_s2, dEbar)."""
    E_a = E_star.contiguous()
    S = int(E_a.shape[1])
    item_rows = int(E_a.shape[0])
    mats = tuple(as_item_state(p, S, item_rows) for p in (log_pS, log_pD, log_pL, max_coupling))
    dmats = tuple(as_item_state(p, S, item_rows) for p in (dlog_pS, dlog_pD, dlog_pL, dmax_coupling))
    col = col_log_probs.contiguous()
    args = (*mats, col, *dmats, node_parent, node_child1, node_child2, int(max_ancestor_depth))

    dE_a = torch.zeros_like(E_a) if dE0 is None else dE0.contiguous().clone()
    dE_b, dE_s1, dE_s2, dEbar = (torch.empty_like(E_a) for _ in range(4))
    max_diff_out = torch.empty((item_rows,), dtype=E_a.dtype, device=E_a.device)

    for _ in range(int(max_iter)):
        _launch_e_step_tangent_2d(
            E_a, dE_a, *args, out=(dE_b, dE_s1, dE_s2, dEbar),
            max_diff_out=max_diff_out, use_col_weights=bool(use_col_weights),
        )
        dE_a, dE_b = dE_b, dE_a
        max_diff = float(max_diff_out.max().item())
        scale = float(dE_a.abs().max().item())
        if max_diff <= tol * max(1.0, scale):
            break

    _, dE_s1, dE_s2, dEbar = _launch_e_step_tangent_2d(
        E_a, dE_a, *args, use_col_weights=bool(use_col_weights),
    )
    return dE_a, dE_s1, dE_s2, dEbar
