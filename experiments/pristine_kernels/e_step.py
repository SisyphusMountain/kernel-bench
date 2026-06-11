import torch
import triton
import triton.language as tl

from kbench.core.parameters.extract_parameters import as_item_state


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


@triton.jit
def _e_step_forward_2d_kernel(
    E_ptr,
    E_new_ptr,
    E_s1_out_ptr,
    E_s2_out_ptr,
    Ebar_out_ptr,
    max_diff_out_ptr,
    log_pS_ptr,
    log_pD_ptr,
    log_pL_ptr,
    max_coupling_ptr,
    col_log_probs_ptr,
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
    if USE_COL_WEIGHTS:
        col_logp = tl.load(col_log_probs_ptr + offs, mask=mask, other=neg_inf)
        weighted_E = col_logp + E
    else:
        weighted_E = E
    row_max = tl.max(weighted_E, axis=0)
    row_max_safe = tl.where(row_max != neg_inf, row_max, tl.zeros([1], dtype=DTYPE))
    row_sum = tl.sum(tl.exp2(weighted_E - row_max_safe), axis=0)

    cur = offs
    ancestor_sum = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        cur_valid = mask & (cur >= 0) & (cur < S)
        E_anc = tl.load(E_ptr + base + cur, mask=cur_valid, other=neg_inf)
        if USE_COL_WEIGHTS:
            col_logp_anc = tl.load(col_log_probs_ptr + cur, mask=cur_valid, other=neg_inf)
            ancestor_sum += tl.where(cur_valid, tl.exp2(col_logp_anc + E_anc - row_max_safe), zero)
        else:
            ancestor_sum += tl.where(cur_valid, tl.exp2(E_anc - row_max_safe), zero)
        cur = tl.load(node_parent_ptr + cur, mask=cur_valid, other=-1).to(tl.int32)

    c1 = tl.load(node_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(node_child2_ptr + offs, mask=mask, other=-1)
    c1_valid = mask & (c1 >= 0) & (c1 < S)
    c2_valid = mask & (c2 >= 0) & (c2 < S)
    E_s1 = tl.load(E_ptr + base + c1, mask=c1_valid, other=neg_inf)
    E_s2 = tl.load(E_ptr + base + c2, mask=c2_valid, other=neg_inf)

    max_coupling_val = tl.load(max_coupling_ptr + base + offs, mask=mask, other=0.0)
    denom = row_sum - ancestor_sum
    Ebar = tl.where(denom > 0.0, tl.log2(denom) + row_max + max_coupling_val, neg_inf)

    pS = tl.load(log_pS_ptr + base + offs, mask=mask, other=neg_inf)
    pD = tl.load(log_pD_ptr + base + offs, mask=mask, other=neg_inf)
    pL = tl.load(log_pL_ptr + base + offs, mask=mask, other=neg_inf)

    t0 = pS + E_s1 + E_s2
    t1 = pD + 2.0 * E
    t2 = E + Ebar
    t3 = pL
    m = tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3))
    m_safe = tl.where(m == neg_inf, zero, m)
    total = (
        tl.exp2(t0 - m_safe)
        + tl.exp2(t1 - m_safe)
        + tl.exp2(t2 - m_safe)
        + tl.exp2(t3 - m_safe)
    )
    E_new = tl.log2(total) + m

    tl.store(E_new_ptr + base + offs, E_new, mask=mask)
    tl.store(E_s1_out_ptr + base + offs, E_s1, mask=mask)
    tl.store(E_s2_out_ptr + base + offs, E_s2, mask=mask)
    tl.store(Ebar_out_ptr + base + offs, Ebar, mask=mask)
    if COMPUTE_DIFF:
        diff = tl.where(mask, tl.abs(E_new - E), zero)
        tl.store(max_diff_out_ptr + g, tl.max(diff, axis=0))


@triton.jit
def _e_step_backward_prepare_2d_kernel(
    E_ptr,
    E_new_ptr,
    E_s1_ptr,
    E_s2_ptr,
    Ebar_ptr,
    log_pS_ptr,
    log_pD_ptr,
    log_pL_ptr,
    col_log_probs_ptr,
    node_parent_ptr,
    node_child1_ptr,
    node_child2_ptr,
    grad_E_new_ptr,
    grad_E_s1_out_ptr,
    grad_E_s2_out_ptr,
    grad_Ebar_out_ptr,
    grad_E_ptr,
    grad_log_pS_ptr,
    grad_log_pD_ptr,
    grad_log_pL_ptr,
    grad_max_coupling_ptr,
    grad_col_log_probs_ptr,
    r_ptr,
    excluded_u_ptr,
    total_u_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
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
    if USE_COL_WEIGHTS:
        col_logp = tl.load(col_log_probs_ptr + offs, mask=mask, other=neg_inf)
        weighted_E = col_logp + E
    else:
        weighted_E = E
    E_new = tl.load(E_new_ptr + base + offs, mask=mask, other=neg_inf)
    E_s1 = tl.load(E_s1_ptr + base + offs, mask=mask, other=neg_inf)
    E_s2 = tl.load(E_s2_ptr + base + offs, mask=mask, other=neg_inf)
    Ebar = tl.load(Ebar_ptr + base + offs, mask=mask, other=neg_inf)
    pS = tl.load(log_pS_ptr + base + offs, mask=mask, other=neg_inf)
    pD = tl.load(log_pD_ptr + base + offs, mask=mask, other=neg_inf)
    pL = tl.load(log_pL_ptr + base + offs, mask=mask, other=neg_inf)

    row_max = tl.max(weighted_E, axis=0)
    row_max_safe = tl.where(row_max != neg_inf, row_max, tl.zeros([1], dtype=DTYPE))
    r = tl.exp2(weighted_E - row_max_safe)
    r = tl.where(mask, r, zero)
    row_sum = tl.sum(r, axis=0)
    tl.store(r_ptr + base + offs, r, mask=mask)
    tl.store(excluded_u_ptr + base + offs, zero, mask=mask)

    g_new = tl.load(grad_E_new_ptr + base + offs, mask=mask, other=0.0)
    g_s1_out = tl.load(grad_E_s1_out_ptr + base + offs, mask=mask, other=0.0)
    g_s2_out = tl.load(grad_E_s2_out_ptr + base + offs, mask=mask, other=0.0)
    g_ebar_out = tl.load(grad_Ebar_out_ptr + base + offs, mask=mask, other=0.0)

    t0 = pS + E_s1 + E_s2
    t1 = pD + 2.0 * E
    t2 = E + Ebar
    t3 = pL
    q0 = tl.where(mask, g_new * tl.exp2(t0 - E_new), zero)
    q1 = tl.where(mask, g_new * tl.exp2(t1 - E_new), zero)
    q2 = tl.where(mask, g_new * tl.exp2(t2 - E_new), zero)
    q3 = tl.where(mask, g_new * tl.exp2(t3 - E_new), zero)

    wbar = q2 + g_ebar_out
    tl.store(grad_log_pS_ptr + base + offs, q0, mask=mask)
    tl.store(grad_log_pD_ptr + base + offs, q1, mask=mask)
    tl.store(grad_log_pL_ptr + base + offs, q3, mask=mask)
    tl.store(grad_max_coupling_ptr + base + offs, wbar, mask=mask)

    tl.store(grad_E_ptr + base + offs, 2.0 * q1 + q2, mask=mask)

    # All warps must finish the plain grad_E / excluded_u initializing stores
    # above before any warp issues the overlapping atomic accumulations below;
    # otherwise a late store overwrites (loses) another warp's atomic add.
    tl.debug_barrier()

    c1 = tl.load(node_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(node_child2_ptr + offs, mask=mask, other=-1)
    c1_valid = mask & (c1 >= 0) & (c1 < S)
    c2_valid = mask & (c2 >= 0) & (c2 < S)
    tl.atomic_add(grad_E_ptr + base + c1, q0 + g_s1_out, sem="relaxed", mask=c1_valid)
    tl.atomic_add(grad_E_ptr + base + c2, q0 + g_s2_out, sem="relaxed", mask=c2_valid)

    cur = offs
    ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (cur >= 0) & (cur < S)
        E_anc = tl.load(E_ptr + base + cur, mask=valid, other=neg_inf)
        if USE_COL_WEIGHTS:
            col_logp_anc = tl.load(col_log_probs_ptr + cur, mask=valid, other=neg_inf)
            ancestor_sum += tl.where(valid, tl.exp2(col_logp_anc + E_anc - row_max_safe), zero)
        else:
            ancestor_sum += tl.where(valid, tl.exp2(E_anc - row_max_safe), zero)
        cur = tl.load(node_parent_ptr + cur, mask=valid, other=-1).to(tl.int32)

    denom = row_sum - ancestor_sum
    u = tl.where(mask & (denom > 0.0), wbar / denom, zero)
    tl.store(total_u_ptr + g, tl.sum(u, axis=0))

    cur = offs
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (cur >= 0) & (cur < S)
        tl.atomic_add(excluded_u_ptr + base + cur, u, sem="relaxed", mask=valid)
        cur = tl.load(node_parent_ptr + cur, mask=valid, other=-1).to(tl.int32)


@triton.jit
def _e_step_backward_finalize_2d_kernel(
    grad_E_ptr,
    grad_col_log_probs_ptr,
    r_ptr,
    excluded_u_ptr,
    total_u_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0)
    excluded = tl.load(excluded_u_ptr + base + offs, mask=mask, other=0.0)
    total = tl.load(total_u_ptr + g)
    current = tl.load(grad_E_ptr + base + offs, mask=mask, other=0.0)
    pibar_vjp = r * (total - excluded)
    tl.store(grad_E_ptr + base + offs, current + pibar_vjp, mask=mask)
    tl.atomic_add(grad_col_log_probs_ptr + offs, pibar_vjp, sem="relaxed", mask=mask)


def _launch_e_step_forward_2d(
    E: torch.Tensor,
    log_pS_mat: torch.Tensor,
    log_pD_mat: torch.Tensor,
    log_pL_mat: torch.Tensor,
    max_coupling_mat: torch.Tensor,
    col_log_probs: torch.Tensor,
    node_parent: torch.Tensor,
    node_child1: torch.Tensor,
    node_child2: torch.Tensor,
    max_ancestor_depth: int,
    *,
    max_diff_out: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    use_col_weights: bool = True,
):
    G = int(E.shape[0])
    S = int(E.shape[1])
    block_s = int(triton.next_power_of_2(S))
    E_new, E_s1, E_s2, Ebar = (torch.empty_like(E) for _ in range(4)) if out is None else out
    _e_step_forward_2d_kernel[(G,)](
        E,
        E_new,
        E_s1,
        E_s2,
        Ebar,
        E_new if max_diff_out is None else max_diff_out,
        log_pS_mat,
        log_pD_mat,
        log_pL_mat,
        max_coupling_mat,
        col_log_probs,
        node_parent,
        node_child1,
        node_child2,
        S,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        COMPUTE_DIFF=max_diff_out is not None,
        USE_COL_WEIGHTS=bool(use_col_weights),
        DTYPE=_tl_float_dtype(E.dtype),
        num_warps=8,
    )
    return E_new, E_s1, E_s2, Ebar


class _TritonEStep2D(torch.autograd.Function):
    @staticmethod
    def forward(
        E,
        log_pS_mat,
        log_pD_mat,
        log_pL_mat,
        max_coupling_mat,
        col_log_probs,
        node_parent,
        node_child1,
        node_child2,
        max_ancestor_depth: int,
        use_col_weights: bool,
    ):
        return _launch_e_step_forward_2d(
            E,
            log_pS_mat,
            log_pD_mat,
            log_pL_mat,
            max_coupling_mat,
            col_log_probs,
            node_parent,
            node_child1,
            node_child2,
            int(max_ancestor_depth),
            use_col_weights=bool(use_col_weights),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], *output, *inputs[1:4], inputs[5], *inputs[6:9])
        ctx.max_ancestor_depth = int(inputs[9])
        ctx.use_col_weights = bool(inputs[10])

    @staticmethod
    def backward(ctx, grad_E_new, grad_E_s1_out, grad_E_s2_out, grad_Ebar_out):
        (
            E,
            E_new,
            E_s1,
            E_s2,
            Ebar,
            log_pS_mat,
            log_pD_mat,
            log_pL_mat,
            col_log_probs,
            node_parent,
            node_child1,
            node_child2,
        ) = ctx.saved_tensors
        G = int(E.shape[0])
        S = int(E.shape[1])
        block_s = int(triton.next_power_of_2(S))
        grad_E_new, grad_E_s1_out, grad_E_s2_out, grad_Ebar_out = (
            torch.zeros_like(E) if grad is None else grad.contiguous()
            for grad in (grad_E_new, grad_E_s1_out, grad_E_s2_out, grad_Ebar_out)
        )
        grad_E, grad_log_pS, grad_log_pD, grad_log_pL, grad_max_coupling, r, excluded_u = (
            torch.empty_like(E) for _ in range(7)
        )
        grad_col_log_probs = torch.zeros_like(col_log_probs)
        total_u = torch.empty((G,), dtype=E.dtype, device=E.device)

        _e_step_backward_prepare_2d_kernel[(G,)](
            E,
            E_new,
            E_s1,
            E_s2,
            Ebar,
            log_pS_mat,
            log_pD_mat,
            log_pL_mat,
            col_log_probs,
            node_parent,
            node_child1,
            node_child2,
            grad_E_new,
            grad_E_s1_out,
            grad_E_s2_out,
            grad_Ebar_out,
            grad_E,
            grad_log_pS,
            grad_log_pD,
            grad_log_pL,
            grad_max_coupling,
            grad_col_log_probs,
            r,
            excluded_u,
            total_u,
            S,
            BLOCK_S=block_s,
            MAX_ANCESTOR_DEPTH=int(ctx.max_ancestor_depth),
            USE_COL_WEIGHTS=bool(ctx.use_col_weights),
            DTYPE=_tl_float_dtype(E.dtype),
            num_warps=8,
        )
        _e_step_backward_finalize_2d_kernel[(G,)](
            grad_E,
            grad_col_log_probs,
            r,
            excluded_u,
            total_u,
            S,
            BLOCK_S=block_s,
            num_warps=8,
        )
        return (
            grad_E,
            grad_log_pS,
            grad_log_pD,
            grad_log_pL,
            grad_max_coupling,
            grad_col_log_probs,
            None,
            None,
            None,
            None,
            None,
        )


def e_step_triton_autograd(
    E: torch.Tensor,
    log_pS: torch.Tensor,
    log_pD: torch.Tensor,
    log_pL: torch.Tensor,
    max_coupling: torch.Tensor,
    col_log_probs: torch.Tensor,
    node_parent: torch.Tensor,
    node_child1: torch.Tensor,
    node_child2: torch.Tensor,
    max_ancestor_depth: int,
    use_col_weights: bool = True,
):
    E_arg = E.contiguous()
    return _TritonEStep2D.apply(
        E_arg,
        *(as_item_state(param, int(E_arg.shape[1]), int(E_arg.shape[0])) for param in (log_pS, log_pD, log_pL, max_coupling)),
        col_log_probs.contiguous(),
        node_parent,
        node_child1,
        node_child2,
        int(max_ancestor_depth),
        bool(use_col_weights),
    )


def e_fixed_point_triton(
    E0: torch.Tensor,
    log_pS: torch.Tensor,
    log_pD: torch.Tensor,
    log_pL: torch.Tensor,
    max_coupling: torch.Tensor,
    col_log_probs: torch.Tensor,
    node_parent: torch.Tensor,
    node_child1: torch.Tensor,
    node_child2: torch.Tensor,
    max_ancestor_depth: int,
    *,
    max_iter: int = 2000,
    tol: float = 1e-8,
    use_col_weights: bool = True,
):
    max_iter = int(max_iter)
    tol = float(tol)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    E_a = E0.contiguous().clone()
    G = int(E_a.shape[0])
    log_pS_mat, log_pD_mat, log_pL_mat, max_coupling_mat = (
        as_item_state(param, int(E_a.shape[1]), int(E_a.shape[0]))
        for param in (log_pS, log_pD, log_pL, max_coupling)
    )
    forward_args = (
        log_pS_mat,
        log_pD_mat,
        log_pL_mat,
        max_coupling_mat,
        col_log_probs.contiguous(),
        node_parent,
        node_child1,
        node_child2,
        int(max_ancestor_depth),
    )

    E_b, E_s1, E_s2, Ebar = (torch.empty_like(E_a) for _ in range(4))
    max_diff_out = torch.empty((G,), dtype=E_a.dtype, device=E_a.device)

    for _ in range(max_iter):
        _launch_e_step_forward_2d(
            E_a,
            *forward_args,
            max_diff_out=max_diff_out,
            out=(E_b, E_s1, E_s2, Ebar),
            use_col_weights=bool(use_col_weights),
        )
        E_a, E_b = E_b, E_a
        max_diff = float(max_diff_out.max().item())
        if max_diff < tol:
            break

    _, E_s1, E_s2, Ebar = _launch_e_step_forward_2d(
        E_a,
        *forward_args,
        use_col_weights=bool(use_col_weights),
    )

    return E_a, E_s1, E_s2, Ebar
