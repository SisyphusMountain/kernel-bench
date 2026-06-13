"""Matrix-free conjugate gradient for the damped Gauss-Newton system (M + lambda I) p = b.

Vectors and inner products are fp64 (the HVP runs fp32 kernels internally and casts at the
boundary). ``M + lambda I`` is positive definite for ``lambda > 0`` and PSD ``M``, so CG is
well-posed; a negative-curvature guard is kept only as numerical safety.
"""

from __future__ import annotations

import torch


def lanczos_extremes(Av, p, *, m=40, seed=0, device="cuda", dtype=torch.float64):
    """Estimate (lambda_min, lambda_max) of the operator via m Lanczos iterations with full
    reorthogonalization. Matrix-free: only m HVPs. Note: the lambda_min Ritz estimate converges
    from ABOVE (optimistic) — on this problem's clustered bottom edge m≈40 is needed for an
    accurate value (m=10 can even miss the sign); lambda_max is accurate by m≈10.
    """
    import numpy as np
    from scipy.linalg import eigh_tridiagonal

    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(p, generator=gen, device=device, dtype=dtype)
    q /= q.norm()
    Q, alphas, betas = [], [], []
    beta, q_prev = 0.0, torch.zeros_like(q)
    for _ in range(int(m)):
        w = Av(q) - beta * q_prev
        a = float(torch.dot(w, q))
        w -= a * q
        for qq in Q:  # full reorthogonalization
            w -= torch.dot(w, qq) * qq
        Q.append(q.clone())
        alphas.append(a)
        b = float(w.norm())
        if b < 1e-12:
            break
        q_prev, q, beta = q, w / b, b
        betas.append(b)
    ev = eigh_tridiagonal(np.array(alphas), np.array(betas[: len(alphas) - 1]), eigvals_only=True)
    return float(ev[0]), float(ev[-1])


def steihaug_cg(Av, b, delta, *, tol, max_iter):
    """Steihaug-Toint truncated CG: approximately minimize the quadratic model
    ``m(p) = -b^T p + 1/2 p^T A p`` subject to ``||p|| <= delta`` (i.e. solve ``A p = b`` inside a
    trust region). Negative curvature is exploited, not avoided: on ``d^T A d <= 0`` the step
    follows ``d`` to the boundary.

    Returns ``(p, Ap, iters, status)`` with status in {'converged','boundary','neg_curv','max_iter'}.
    ``Ap`` is accumulated exactly from the iteration's ``A d`` products (no extra HVP), so the
    caller gets the model reduction ``-m(p) = b^T p - 1/2 p^T (Ap)`` for free.
    """
    p = torch.zeros_like(b)
    Ap = torch.zeros_like(b)
    r = b.clone()
    d = r.clone()
    rs = float(torch.dot(r, r))
    if rs ** 0.5 <= tol:
        return p, Ap, 0, "converged"

    def to_boundary(p, d):
        # positive tau with ||p + tau d|| = delta
        pd = float(torch.dot(p, d))
        dd = float(torch.dot(d, d))
        pp = float(torch.dot(p, p))
        tau = (-pd + (pd * pd + dd * (delta * delta - pp)) ** 0.5) / dd
        return tau

    for j in range(1, int(max_iter) + 1):
        Ad = Av(d)
        dAd = float(torch.dot(d, Ad))
        if dAd <= 0.0:
            tau = to_boundary(p, d)
            return p + tau * d, Ap + tau * Ad, j, "neg_curv"
        alpha = rs / dAd
        p_next = p + alpha * d
        if float(torch.linalg.vector_norm(p_next)) >= delta:
            tau = to_boundary(p, d)
            return p + tau * d, Ap + tau * Ad, j, "boundary"
        p = p_next
        Ap = Ap + alpha * Ad
        r = r - alpha * Ad
        rs_new = float(torch.dot(r, r))
        if rs_new ** 0.5 <= tol:
            return p, Ap, j, "converged"
        d = r + (rs_new / rs) * d
        rs = rs_new
    return p, Ap, int(max_iter), "max_iter"


def cg_witness(Av, b, *, tol, max_iter):
    """CG on the damped system ``Av(x) = b`` that reports negative curvature as a certificate.

    Returns ``(x, iters, status, cert)`` with status in {'converged','max_iter','neg_curv'}.
    ``cert`` is None except on 'neg_curv', where it is the damped Rayleigh quotient
    ``d^T A d / ||d||^2 <= 0`` of the offending search direction. With ``A = H + lam*I`` this
    certifies ``|d^T H d|/||d||^2 = lam - cert > lam``, i.e. lambda_min(H) <= cert - lam < -lam:
    the damping was provably too small along ``d``, and the caller should re-solve with
    ``lam_new = nu * (lam - cert)``.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    d = r.clone()
    rs = float(torch.dot(r, r))
    if rs ** 0.5 <= tol:
        return x, 0, "converged", None
    for j in range(1, int(max_iter) + 1):
        Ad = Av(d)
        dAd = float(torch.dot(d, Ad))
        dd = float(torch.dot(d, d))
        if dAd <= 0.0:
            return x, j, "neg_curv", dAd / dd
        alpha = rs / dAd
        x = x + alpha * d
        r = r - alpha * Ad
        rs_new = float(torch.dot(r, r))
        if rs_new ** 0.5 <= tol:
            return x, j, "converged", None
        d = r + (rs_new / rs) * d
        rs = rs_new
    return x, int(max_iter), "max_iter", None


def cg_solve(Av, b, *, tol, max_iter, x0=None):
    """Solve ``Av(x) = b`` by CG. Returns (x, iters, converged). ``tol`` is on the residual norm."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Av(x) if x0 is not None else b.clone()
    p = r.clone()
    rs = float(torch.dot(r, r))
    bnorm = float(torch.linalg.vector_norm(b))
    if bnorm == 0.0:
        return x, 0, True
    it = 0
    for it in range(1, int(max_iter) + 1):
        Ap = Av(p)
        pAp = float(torch.dot(p, Ap))
        if pAp <= 0.0:  # safety only; damped system is PD
            if it == 1:
                x = b / max(pAp / max(float(torch.dot(p, p)), 1e-30), 1e-12) if pAp != 0 else b
            break
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(torch.dot(r, r))
        if rs_new ** 0.5 <= tol:
            return x, it, True
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, it, False
