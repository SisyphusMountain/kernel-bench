"""Levenberg-Marquardt Newton-CG outer loop using the analytic Gauss-Newton/Fisher HVP.

Each step: forward+backward at theta -> (loss, g, sv); CG solves (M + lambda I) p = -g with the
matrix-free GGN HVP built from sv; an Armijo backtracking line search on the (deterministic)
forward loss accepts a step and adapts lambda. The forward is warm-started from the previous E
throughout (FD-free here: M is analytic).
"""

from __future__ import annotations

import torch

from newton.vg import make_value_and_grad, forward_solve
from newton.ggn import make_ggn_hvp
from newton.cg import cg_solve, cg_witness, lanczos_extremes, steihaug_cg


def _fd_hessian_hvp(vg, theta_vec, warm_E, *, eps=1e-5):
    """True-Hessian HVP via central FD of the gradient (reuses forward+backward, no new kernels).

    The direction is normalized so the FD perturbation magnitude is ``eps`` regardless of the CG
    vector's scale (CG search directions grow/shrink across iterations).
    """
    from newton.vg import free_cuda_cache_if_tight

    base = theta_vec.double()

    def hvp(v):
        v = v.double()
        nv = float(torch.linalg.vector_norm(v))
        if nv == 0.0:
            return torch.zeros_like(v)
        u = v / nv
        free_cuda_cache_if_tight()
        # avoid `_, gp, _, _ = vg(...)`: it would pin the GB-sized saved dict across calls
        out = vg((base + eps * u).to(theta_vec.dtype), warm_E=warm_E)
        gp = out[1].double()
        del out
        out = vg((base - eps * u).to(theta_vec.dtype), warm_E=warm_E)
        gm = out[1].double()
        del out
        return nv * (gp - gm) / (2 * eps)

    return hvp


def newton_lanczos(static, theta0, col_weights, *, sigma=0.01, sigma_floor=1e-4, lanczos_m=10,
                   nu=1.5, omega=1.5, max_bumps=3, eta_max=0.1, max_cg=40, c1=1e-4, ls_max=25,
                   gtol=1e-2, max_newton=40, fd_eps=1e-5, lam=0.0, theta_ref=None,
                   lanczos_refresh=0, ftol=1e-9, hvp_mode="fd", verbose=True):
    """Lanczos-initialized, witness-corrected damped Newton descent ("Newton-gradient descent").

    lam_damp interpolates between Newton (small) and scaled gradient descent (large). It is
    initialized by the cheap spectral rule ``lam_damp = sigma * lam_max`` (m~10 Lanczos: only
    lam_max is needed, which converges almost immediately; no lam_min estimation). At runtime the
    CG negative-curvature witness self-corrects: if CG on ``H_eff + lam_damp*I`` encounters
    ``d^T A d <= 0``, the direction certifies the damping needed and lam_damp is bumped to
    ``nu *`` that magnitude and the solve restarted. Steps are globalized by Armijo backtracking
    on the (deterministic) forward loss; lam_damp decays toward ``sigma_floor*lam_max`` on full
    steps and grows on backtracked/failed ones.

    ``lam``/``theta_ref`` optionally add the ridge/MAP objective term (as in ``newton_tr``):
    F = L + lam/2 ||x - theta_ref||^2 with H_eff = H + lam*I — required for a quadratic endgame
    on this problem's flat-at-the-optimum spectrum. Run in fp64 (pass an fp64 theta0).

    Returns (theta, history); history rows carry loss/F, ||gF||, lam_damp, cg iters/status,
    witness certificates, alpha, and cumulative gradient-eval count.
    """
    S = int(static.state_helpers["S"])
    theta_vec = theta0.reshape(-1).clone()
    p_dim = theta_vec.numel()
    vg = make_value_and_grad(static, col_weights)
    lam_obj = float(lam)
    x_ref = (theta_vec if theta_ref is None else theta_ref.reshape(-1).to(theta_vec)).double().clone()
    evals = [0]  # cumulative forward+backward evaluations, tracked via wrapper

    def vg_counted(x, **kw):
        evals[0] += 1
        return vg(x, **kw)

    def penalty(x):
        return 0.5 * lam_obj * float((x.double() - x_ref).norm() ** 2) if lam_obj > 0 else 0.0

    loss, g, sv, warm_E = vg_counted(theta_vec)
    sv = None

    def make_hvp_eff(x_vec, warm):
        if hvp_mode == "exact":
            # analytic exact HVP: one forward+backward builds the per-point adjoint cache,
            # then every CG iteration costs ~1 tangent-forward + 1 tangent-adjoint sweep
            from newton.hvp_exact import make_exact_hvp

            theta_m = x_vec.reshape(S, 3)
            _, sv_pt = forward_solve(static, theta_m, col_weights, warm_E=warm)
            evals[0] += 2  # cache build ~ 1 fwd + 1 bwd
            h = make_exact_hvp(static, theta_m, col_weights, sv_pt)
        else:
            h = _fd_hessian_hvp(vg_counted, x_vec, warm, eps=fd_eps)
        if lam_obj > 0:
            return lambda v: h(v).double() + lam_obj * v.double()
        return lambda v: h(v).double()

    hvp_eff = make_hvp_eff(theta_vec, warm_E)
    _, lam_max = lanczos_extremes(hvp_eff, p_dim, m=lanczos_m, device=str(theta_vec.device))
    lam_damp = sigma * lam_max
    lam_floor = sigma_floor * lam_max
    lam_ceil = 10.0 * lam_max
    if verbose:
        print(f"[lanczos-newton] m={lanczos_m}  lam_max~{lam_max:.2f}  "
              f"lam_damp0={lam_damp:.4f}  floor={lam_floor:.2e}  ceil={lam_ceil:.1f}")

    history = []
    accepted_steps = 0
    stalls = 0
    for k in range(int(max_newton)):
        gF = g.double() + (lam_obj * (theta_vec.double() - x_ref) if lam_obj > 0 else 0.0)
        F = loss + penalty(theta_vec)
        gnorm = float(torch.linalg.vector_norm(gF))
        rec = {"newton": k, "loss": loss, "F": F, "gnorm": gnorm, "lam_damp": lam_damp,
               "witness_certs": [], "evals": evals[0]}
        history.append(rec)
        if verbose:
            print(f"[ln {k:2d}] F={F:.6f}  ||gF||={gnorm:.4e}  lam={lam_damp:.3e}", end="")
        if gnorm < gtol:
            if verbose:
                print("  converged")
            break

        if lanczos_refresh and accepted_steps and accepted_steps % int(lanczos_refresh) == 0:
            hvp_eff = make_hvp_eff(theta_vec, warm_E)
            _, lam_max = lanczos_extremes(hvp_eff, p_dim, m=lanczos_m, device=str(theta_vec.device))
            lam_floor = sigma_floor * lam_max
            lam_ceil = 10.0 * lam_max
        else:
            hvp_eff = make_hvp_eff(theta_vec, warm_E)

        # damped solve with witness self-correction
        eta = min(eta_max, gnorm ** 0.5)
        p, cg_iters, status = None, 0, ""
        for bump in range(int(max_bumps) + 1):
            Av = lambda v: hvp_eff(v) + lam_damp * v
            p, cg_iters, status, cert = cg_witness(Av, -gF, tol=eta * gnorm, max_iter=max_cg)
            if status != "neg_curv":
                break
            new_lam = nu * (lam_damp - cert)  # cert <= 0 is the damped Rayleigh quotient
            rec["witness_certs"].append(lam_damp - cert)
            if verbose:
                print(f"\n      witness: d^T(H+lam)d/|d|^2={cert:.3e} -> lam {lam_damp:.3e} -> {new_lam:.3e}", end="")
            lam_damp = min(lam_ceil, new_lam)
        if status == "neg_curv":  # bumps exhausted
            p = -gF / lam_damp
            status = "fallback_gd"
        rec["cg"] = cg_iters
        rec["status"] = status

        gp = float(torch.dot(gF, p))
        if gp >= 0.0:
            p = -gF / lam_damp
            gp = -gnorm * gnorm / lam_damp
            status += "+gd"

        # Armijo backtracking on the deterministic forward loss
        alpha, accepted, sv_t = 1.0, False, None
        for _ in range(int(ls_max)):
            trial = (theta_vec.double() + alpha * p).to(theta_vec.dtype)
            lt, st = forward_solve(static, trial.reshape(S, 3), col_weights, warm_E=warm_E)
            Ft = float(lt) + penalty(trial)
            if Ft <= F + c1 * alpha * gp:
                accepted, sv_t = True, st
                break
            alpha *= 0.5
        rec["alpha"] = alpha if accepted else None

        if accepted:
            accepted_steps += 1
            theta_vec = trial
            warm_E = sv_t["E"]
            sv_t = None
            lam_damp = max(lam_floor, lam_damp / omega) if alpha == 1.0 else min(lam_ceil, 1.5 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  a={alpha:.2e}  dF={Ft - F:+.4e}")
            # accepted improvements at the forward solver's truncation floor are noise; two in a
            # row means further polishing cannot be validated -- stop instead of micro-stepping
            stalls = stalls + 1 if (F - Ft) <= ftol * max(1.0, abs(F)) else 0
            if stalls >= 2:
                if verbose:
                    print(f"[ln {k + 1:2d}] improvement below ftol floor twice -- stopping")
                break
            loss, g, sv, warm_E = vg_counted(theta_vec, warm_E=warm_E)
            sv = None
        else:
            lam_damp = min(lam_ceil, 4.0 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  line-search failed -> lam={lam_damp:.3e}")
            if lam_damp >= lam_ceil:
                if verbose:
                    print("  lam at ceiling with no accepted step -- stopping")
                break

    return theta_vec.reshape(S, 3), history


def newton_tr(static, theta0, col_weights, *, curvature="fd_hessian", max_newton=30, gtol=1e-2,
              delta0=1.0, delta_max=1e3, eta_accept=0.05, max_cg=40, fd_eps=1e-5,
              lam=0.0, theta_ref=None, verbose=True, hvp_kwargs=None):
    """Trust-region Newton-CG (Steihaug): negative curvature is followed to the trust boundary
    instead of being damped away, which is what an indefinite/saddle start needs.

    With ``lam > 0`` minimizes the ridge/MAP objective ``F = L + lam/2 ||x - theta_ref||^2``
    (theta_ref defaults to theta0): H+lam*I is PD for lam past the spectrum's negative edge, which
    restores the quadratic Newton endgame on this problem's flat/indefinite landscape. ``gtol``
    then applies to ||grad F||.

    Runs in the dtype of ``theta0`` (use fp64: the loss is then precise, so the acceptance ratio
    test is exact, and the backward's atomic noise is negligible). Returns (theta, history).
    """
    S = int(static.state_helpers["S"])
    theta_vec = theta0.reshape(-1).clone()
    vg = make_value_and_grad(static, col_weights)
    hvp_kwargs = hvp_kwargs or {}
    delta = float(delta0)
    lam = float(lam)
    x_ref = (theta_vec if theta_ref is None else theta_ref.reshape(-1).to(theta_vec)).double().clone()
    warm_E = None
    history = []
    rejects = 0

    def penalty(x):
        return 0.5 * lam * float((x.double() - x_ref).norm() ** 2) if lam > 0 else 0.0

    loss, g, sv, warm_E = vg(theta_vec, warm_E=warm_E)
    for k in range(int(max_newton)):
        if curvature != "ggn":
            sv = None  # only the GGN curvature needs the saved intermediates; they are ~GBs
        from newton.vg import free_cuda_cache_if_tight
        free_cuda_cache_if_tight()
        gF = g.double() + (lam * (theta_vec.double() - x_ref) if lam > 0 else 0.0)
        F = loss + penalty(theta_vec)
        gnorm = float(torch.linalg.vector_norm(gF))
        history.append({"newton": k, "loss": loss, "F": F, "gnorm": gnorm, "delta": delta})
        if verbose:
            print(f"[tr {k:2d}] F={F:.6f}  ||gF||={gnorm:.4e}  delta={delta:.2e}", end="")
        if gnorm < gtol:
            if verbose:
                print("  converged")
            break

        theta = theta_vec.reshape(S, 3)
        if curvature == "fd_hessian":
            hvp = _fd_hessian_hvp(vg, theta_vec, warm_E, eps=fd_eps)
        elif curvature == "ggn":
            hvp = make_ggn_hvp(static, theta, col_weights, sv, **hvp_kwargs)
        else:
            raise ValueError(f"unknown curvature {curvature!r}")

        eta = min(0.1, gnorm ** 0.5)
        Av = (lambda x: hvp(x).double() + lam * x) if lam > 0 else (lambda x: hvp(x).double())
        p, Hp, cg_iters, status = steihaug_cg(Av, -gF, delta, tol=eta * gnorm, max_iter=max_cg)
        pred = float(torch.dot(-gF, p)) - 0.5 * float(torch.dot(p, Hp))
        pnorm = float(torch.linalg.vector_norm(p))

        trial = (theta_vec.double() + p).to(theta_vec.dtype)
        loss_t, sv_t = forward_solve(static, trial.reshape(S, 3), col_weights, warm_E=warm_E)
        loss_t = float(loss_t)
        ared = F - (loss_t + penalty(trial))
        rho = ared / pred if pred > 0 else -1.0

        if rho < 0.25:
            delta = 0.25 * pnorm
        elif rho > 0.75 and pnorm >= 0.99 * delta:
            delta = min(2.0 * delta, delta_max)

        if rho > eta_accept and ared > 0:
            rejects = 0
            theta_vec = trial
            warm_E = sv_t["E"]
            sv_t = None  # release trial intermediates before the next full forward+backward
            if verbose:
                print(f"  cg={cg_iters}({status})  |p|={pnorm:.2e}  rho={rho:.2f}  dL={-ared:+.4e}")
            loss, g, sv, warm_E = vg(theta_vec, warm_E=warm_E)
        else:
            rejects += 1
            if verbose:
                print(f"  cg={cg_iters}({status})  |p|={pnorm:.2e}  rho={rho:.2f}  REJECT")
            # consecutive rejects with shrinking radius = the quadratic model can no longer be
            # validated against the forward solver's truncation noise -> converged to the floor
            if delta < 1e-10 or rejects >= 3:
                if verbose:
                    print("  stalled at solver precision floor" if rejects >= 3 else "  trust region collapsed")
                break

    return theta_vec.reshape(S, 3), history


def newton_cg(static, theta0, col_weights, *, curvature="ggn", max_newton=25, gtol=1e-2,
              lam0=1.0, lam_min=1e-7, lam_max=1e10, max_cg=30, eta_max=0.1, c1=1e-4, ls_max=25,
              fd_eps=1e-5, verbose=True, hvp_kwargs=None):
    """Minimize the NLL over theta with LM-damped Newton-CG. Returns (theta, history).

    ``curvature='ggn'`` uses the analytic Gauss-Newton/Fisher HVP (PSD); ``'fd_hessian'`` uses the
    true Hessian via central FD of the gradient (indefinite away from a minimum).
    """
    S = int(static.state_helpers["S"])
    theta_vec = theta0.reshape(-1).clone()
    vg = make_value_and_grad(static, col_weights)
    hvp_kwargs = hvp_kwargs or {}
    lam = float(lam0)
    warm_E = None
    history = []

    for k in range(int(max_newton)):
        loss, g, sv, warm_E = vg(theta_vec, warm_E=warm_E)
        g64 = g.double()
        gnorm = float(torch.linalg.vector_norm(g64))
        history.append({"newton": k, "loss": loss, "gnorm": gnorm, "lam": lam})
        if verbose:
            print(f"[newton {k:2d}] loss={loss:.6f}  ||g||={gnorm:.4e}  lam={lam:.2e}", end="")
        if gnorm < gtol:
            if verbose:
                print("  converged")
            break

        theta = theta_vec.reshape(S, 3)
        if curvature == "ggn":
            hvp = make_ggn_hvp(static, theta, col_weights, sv, **hvp_kwargs)
        elif curvature == "fd_hessian":
            hvp = _fd_hessian_hvp(vg, theta_vec, warm_E, eps=fd_eps)
        else:
            raise ValueError(f"unknown curvature {curvature!r}")
        eta = min(eta_max, gnorm ** 0.5)

        def Av(x):
            return hvp(x).double() + lam * x

        p, cg_iters, conv = cg_solve(Av, -g64, tol=eta * gnorm, max_iter=max_cg)
        gp = float(torch.dot(g64, p))
        if gp >= 0.0:  # not a descent direction (CG failure); fall back to steepest descent
            p = -g64
            gp = -gnorm * gnorm

        alpha, accepted, loss_trial, sv_trial = 1.0, False, loss, None
        for _ in range(int(ls_max)):
            tt = (theta_vec + alpha * p).to(theta_vec.dtype)
            lt, st = forward_solve(static, tt.reshape(S, 3), col_weights, warm_E=warm_E)
            loss_trial = float(lt)
            if loss_trial <= loss + c1 * alpha * gp:
                accepted, sv_trial = True, st
                break
            alpha *= 0.5

        if accepted:
            theta_vec = (theta_vec + alpha * p).to(theta_vec.dtype)
            warm_E = sv_trial["E"]
            lam = max(lam_min, lam * (0.4 if alpha == 1.0 else 1.5))
            if verbose:
                print(f"  cg={cg_iters}{'' if conv else '*'}  a={alpha:.2e}  d_loss={loss_trial - loss:+.4e}")
        else:
            lam = min(lam_max, lam * 8.0)
            if verbose:
                print(f"  cg={cg_iters}  line-search failed -> lam={lam:.2e}")
            if lam >= lam_max:
                break

    return theta_vec.reshape(S, 3), history
