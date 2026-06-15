"""Dataset-agnostic likelihood optimization: pluggable first-order optimizer + LR schedule,
then an optional exact-fp32 Newton polish.

The trick that makes any ``torch.optim`` optimizer + any LR schedule usable on this model: the
gradient is computed by the hand-written Triton backward (``make_value_and_grad``), not autograd,
so we make ``theta`` a leaf and assign that gradient to ``theta.grad`` before ``opt.step()``.

Usage:
    python newton/optimize.py --size 666x80                 # default recipe (adam + adaptive -> Newton)
    python newton/optimize.py --size 666x80 --bench          # sweep optimizers x schedules + baselines
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from newton.vg import (DATA, forward_solve, free_cuda_cache_if_tight, load_problem,
                       make_value_and_grad)

# ----------------------------------------------------------------------------------------------
# optimizer factory
# ----------------------------------------------------------------------------------------------
_OPTIMIZERS = {
    "adam": lambda p, lr: torch.optim.Adam([p], lr=lr),
    "adamw": lambda p, lr: torch.optim.AdamW([p], lr=lr),
    "nadam": lambda p, lr: torch.optim.NAdam([p], lr=lr),
    "adagrad": lambda p, lr: torch.optim.Adagrad([p], lr=lr),
    "rmsprop": lambda p, lr: torch.optim.RMSprop([p], lr=lr),
    "sgd": lambda p, lr: torch.optim.SGD([p], lr=lr, momentum=0.9),
}


class Schedule:
    """Unified LR schedule. ``update(loss, g)`` is called each step (after the grad is known,
    before ``opt.step()``) and returns the LR to use for this step.

    - ``constant``: fixed lr0.
    - ``cosine``  : lr0 * 0.5(1+cos(pi t/T)) over ``t_max`` steps.
    - ``plateau`` : multiply by ``factor`` after ``patience`` steps without loss improvement.
    - ``adaptive``: loss-reactive bold-driver. Shrink hard on a loss increase (overshoot), shrink
      on gradient-direction reversal (oscillation: g.gprev < 0), grow while progress is steady.
      Robust to the ~1e-5 relative atomic loss noise via ``noise_rtol``.
    """

    def __init__(self, kind, lr0, *, t_max=200, patience=15, factor=0.5,
                 grow=1.1, shrink=0.5, osc_shrink=0.7, lr_min=None, lr_max=None, noise_rtol=3e-5):
        self.kind = kind
        self.lr0 = float(lr0)
        self.lr = float(lr0)
        self.t = 0
        self.t_max = max(1, int(t_max))
        self.patience = int(patience)
        self.factor = float(factor)
        self.grow, self.shrink, self.osc_shrink = float(grow), float(shrink), float(osc_shrink)
        self.lr_min = lr_min if lr_min is not None else lr0 * 1e-4
        self.lr_max = lr_max if lr_max is not None else lr0 * 4.0
        self.noise_rtol = float(noise_rtol)
        self.best = math.inf
        self.bad = 0
        self.prev_loss = None
        self.prev_g = None

    def _clamp(self):
        self.lr = float(min(self.lr_max, max(self.lr_min, self.lr)))

    def update(self, loss, g):
        if self.kind == "constant":
            pass
        elif self.kind == "cosine":
            self.lr = self.lr0 * 0.5 * (1.0 + math.cos(math.pi * min(self.t, self.t_max) / self.t_max))
        elif self.kind == "plateau":
            tol = self.noise_rtol * max(1.0, abs(loss))
            if loss < self.best - tol:
                self.best, self.bad = loss, 0
            else:
                self.bad += 1
                if self.bad >= self.patience:
                    self.lr *= self.factor
                    self.bad = 0
        elif self.kind == "adaptive":
            tol = self.noise_rtol * max(1.0, abs(loss))
            cos = None
            if self.prev_g is not None:
                denom = float(g.norm() * self.prev_g.norm())
                cos = float(torch.dot(g, self.prev_g)) / denom if denom > 0 else 0.0
            if self.prev_loss is not None:
                if loss > self.prev_loss + tol:          # overshoot
                    self.lr *= self.shrink
                elif cos is not None and cos < 0.0:       # oscillation
                    self.lr *= self.osc_shrink
                elif loss < self.prev_loss - tol and (cos is None or cos > 0.5):
                    self.lr *= self.grow                  # steady progress -> accelerate
            self.prev_loss = loss
            self.prev_g = g.detach().clone()
        self._clamp()
        self.t += 1
        return self.lr


# ----------------------------------------------------------------------------------------------
# stage 1 : first-order
# ----------------------------------------------------------------------------------------------
def first_order(static, theta0, col_weights, *, optimizer="adam", lr0=1.0, schedule="adaptive",
                max_steps=300, rtol=0.05, window=20, loss_rtol=1e-5, lr_floor_frac=1e-3,
                verbose=True, t0_wall=None, return_best=False, early_stop=True):
    """Run a torch.optim optimizer with a pluggable LR schedule. Returns (theta[S,3], hist, warm).

    ``return_best=True`` appends the LOWEST-loss theta visited (a 4th return value): on this fixture
    the low-loss valley floor is *visited mid-trajectory* by an oscillating (e.g. constant-LR) run but
    is not the final iterate, and that valley floor is the right hand-off point for the L-BFGS stage.
    ``early_stop=False`` disables the relative flat/grad stop (let an oscillating dive run its course).
    """
    S = int(static.state_helpers["S"])
    dev = theta0.device
    theta = theta0.detach().reshape(S, 3).float().clone().requires_grad_(True)
    f = make_value_and_grad(static, col_weights)
    opt = _OPTIMIZERS[optimizer](theta, lr0)
    sched = Schedule(schedule, lr0, t_max=max_steps,
                     lr_min=lr0 * lr_floor_frac, patience=max(5, window // 2))
    lr_floor = lr0 * lr_floor_frac

    hist, warm, g0 = [], None, None
    best_loss, best_theta = float("inf"), theta.detach().reshape(S, 3).clone()
    t_start = time.perf_counter() if t0_wall is None else t0_wall
    for step in range(int(max_steps)):
        loss, g, _sv, warm = f(theta.detach().reshape(-1), warm_E=warm)
        gn = float(g.norm())
        if g0 is None:
            g0 = max(gn, 1e-30)
        if loss < best_loss:                       # the current theta attains `loss` (pre-step)
            best_loss, best_theta = loss, theta.detach().reshape(S, 3).clone()
        lr = sched.update(loss, g)
        opt.param_groups[0]["lr"] = lr
        theta.grad = g.reshape(S, 3)
        opt.step()
        wall = time.perf_counter() - t_start
        hist.append({"stage": "first", "step": step, "loss": loss, "gnorm": gn,
                     "lr": lr, "wall_s": wall})
        if verbose and (step < 3 or step % 25 == 0):
            print(f"  [{optimizer}/{schedule} {step:3d}] loss={loss:.4f} ||g||={gn:.3e} "
                  f"lr={lr:.3e} t={wall:.1f}s")
        # relative, dataset-agnostic stopping
        if early_stop and step >= window:
            recent = [h["loss"] for h in hist[-window:]]
            flat = (max(recent) - min(recent)) <= loss_rtol * max(1.0, abs(loss))
            if gn <= rtol * g0 or flat or lr <= lr_floor:
                why = ("grad" if gn <= rtol * g0 else "flat" if flat else "lr-floor")
                if verbose:
                    print(f"  [{optimizer}/{schedule}] stop@{step} ({why}) loss={loss:.4f} "
                          f"||g||={gn:.3e}")
                break
    if return_best:
        return theta.detach().reshape(S, 3), hist, warm, best_theta
    return theta.detach().reshape(S, 3), hist, warm


# ----------------------------------------------------------------------------------------------
# stage 2 : exact-fp32 Newton polish
# ----------------------------------------------------------------------------------------------
def newton_polish(static, theta_stage1, col_weights, *, ridge=False, max_newton=12, gtol=1e-2,
                  lanczos_m=10, sigma=0.01, verbose=True, t0_wall=None):
    """Ridge/witness-corrected Newton on the exact fp32 HVP from the first-order endpoint.

    ``ridge=False``: rely on newton_lanczos's internal sigma*lam_max damping + CG witness
    (cheap: ~lanczos_m HVPs). ``ridge=True``: add the MAP term lam/2||theta-ref||^2 with lam from
    a short exact-HVP Lanczos (convexifies the flat optimum for a quadratic endgame)."""
    from newton.newton_cg import newton_lanczos

    S = int(static.state_helpers["S"])
    theta_f = theta_stage1.detach().reshape(S, 3).float().contiguous()
    lam = 0.0
    if ridge:
        lam = _exact_ridge_lambda(static, theta_f, col_weights, m=max(20, lanczos_m), sigma=sigma,
                                  verbose=verbose)
    t_start = time.perf_counter() if t0_wall is None else t0_wall
    theta_hat, h_newton = newton_lanczos(
        static, theta_f, col_weights, hvp_mode="exact", lanczos_m=lanczos_m, sigma=sigma,
        max_newton=max_newton, gtol=gtol, lam=lam,
        theta_ref=(theta_f if ridge else None), verbose=verbose,
    )
    hist = []
    for r in h_newton:
        hist.append({"stage": "newton", "step": int(r.get("newton", 0)), "loss": float(r["loss"]),
                     "gnorm": float(r["gnorm"]), "lam_damp": float(r.get("lam_damp", 0.0)),
                     "wall_s": time.perf_counter() - t_start})
    return theta_hat.detach().reshape(S, 3), hist, lam


def _exact_ridge_lambda(static, theta, col_weights, *, m=20, sigma=0.01, verbose=True):
    """lam = -min(lam_min,0) + sigma*lam_max via a short EXACT-fp32 HVP Lanczos (cheaper than the
    FD-fp64 auto_lambda in pipeline.py)."""
    from newton.cg import lanczos_extremes
    from newton.hvp_exact import make_exact_hvp

    _, sv = forward_solve(static, theta, col_weights)
    hvp = make_exact_hvp(static, theta, col_weights, sv)
    p = 3 * int(static.state_helpers["S"])
    lo, hi = lanczos_extremes(lambda q: hvp(q.float()).double(), p, m=int(m),
                              device=str(theta.device))
    lam = -min(lo, 0.0) + sigma * hi
    if verbose:
        print(f"  [ridge] exact-HVP Lanczos m={m}: lam_min~{lo:+.3e} lam_max~{hi:.2f} -> lam={lam:.4f}")
    return float(lam)


# ----------------------------------------------------------------------------------------------
# stage 2b : ridge-annealing (lambda-continuation) Newton polish
# ----------------------------------------------------------------------------------------------
def ridge_anneal(static, theta0, col_weights, *, lam0=None, sigma=0.01, theta_ref_mode="moving",
                 inner_steps=3, max_levels=8, max_cg=30, nu=1.5, max_bumps=3, gtol=1e-2,
                 lam_floor_frac=1e-3, eta_max=0.1, c1=1e-4, ls_max=25, lanczos_m=20,
                 anneal_fast=0.3, anneal_slow=0.7, improve_rtol=1e-6, spectrum_m=0,
                 verbose=True, t0_wall=None):
    """Lambda-continuation Newton polish for the flat/indefinite optimum.

    A single ridge lambda conditions the system (CG converges, ||g|| monotone) but its MAP term
    ``lam/2||theta-theta_ref||^2`` biases the minimizer toward ``theta_ref`` (stalls ~0.1% above the
    valley floor). This anneals lambda down: start big (stable, fast CG), shrink it warm-started so
    the target slides toward the floor, using the CG negative-curvature witness as BOTH the per-step
    safety net and the stop signal for how low lambda can safely go.

    Two regularizers, distinct jobs:
      - outer ridge ``lam`` (annealed): adds ``lam*I`` to the Hessian AND moves the minimizer; the
        global conditioning schedule.
      - inner witness ``delta`` (``cg_witness`` bump ``delta <- nu*(delta-cert)``): adds ``delta*I``
        to the SOLVE only; per-subproblem safety when ``H+lam*I`` goes indefinite. While the witness
        stays quiet there is headroom -> keep shrinking; when it fires every step the bare problem is
        in the bounce regime -> stop.

    ``theta_ref_mode``: "moving" (proximal: re-center on the current iterate each level; default,
    slides toward the floor) or "fixed" (Tikhonov homotopy from theta0). Runs the exact fp32 HVP;
    CG vectors are fp64. Returns (theta[S,3], history, lam0)."""
    from newton.cg import cg_witness, lanczos_extremes
    from newton.hvp_exact import make_exact_hvp

    S = int(static.state_helpers["S"])
    p_dim = 3 * S
    theta = theta0.detach().reshape(S, 3).float().contiguous()
    f = make_value_and_grad(static, col_weights)
    t_start = time.perf_counter() if t0_wall is None else t0_wall

    # up-front spectrum on the exact HVP at the start point -> lam0 (auto_lambda rule) + lam_floor
    free_cuda_cache_if_tight()
    _, sv = forward_solve(static, theta, col_weights)
    warm = sv["E"]
    hvp0 = make_exact_hvp(static, theta, col_weights, sv)
    lo, hi = lanczos_extremes(lambda q: hvp0(q.float()).double(), p_dim, m=int(lanczos_m),
                              device=str(theta.device))
    lam = (-min(lo, 0.0) + sigma * hi) if lam0 is None else float(lam0)
    lam_start = lam
    lam_floor = lam_floor_frac * hi
    if verbose:
        print(f"[ridge-anneal] exact-HVP Lanczos m={lanczos_m}: lam_min~{lo:+.3e} lam_max~{hi:.2f}"
              f" -> lam0={lam:.4f}  lam_floor={lam_floor:.3e}  ref={theta_ref_mode}")

    # drop the up-front spectrum HVP closure + its pinned forward intermediates BEFORE the loop --
    # only ONE point's forward intermediates may be live at once or the backward's driver-free
    # scratch gate trips (same discipline as newton_lanczos). Keep only warm (E) across the drop.
    theta_ref = theta.clone()
    del hvp0, sv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    history, converged, hvp = [], False, None
    for level in range(int(max_levels)):
        # build the exact HVP ONCE per level at the (moved) current iterate; fixed for the inner
        # steps (a loose, warm-started subproblem solve). Drop the PREVIOUS level's closure first
        # (free its pinned sv before building the next point's cache) then defrag.
        hvp = None
        free_cuda_cache_if_tight(min_free_gib=8.0)
        _, sv = forward_solve(static, theta, col_weights, warm_E=warm)
        warm = sv["E"]
        hvp = make_exact_hvp(static, theta, col_weights, sv)
        sv = None  # the closure pins what it needs; drop the big dict container
        lam_cur = lam
        A = lambda v, lc=lam_cur: hvp(v.float()).double() + lc * v  # ridge-damped operator

        # DIAGNOSTIC: bare-Hessian spectrum at this level's iterate. lam_min crossing 0 as lam
        # anneals down is the smoking gun for the flat/indefinite valley (it explains a ||g|| bounce
        # and CG stalling). lam_min Ritz converges from ABOVE (optimistic), so it is a trend, not an
        # exact edge; lam_max is accurate. PD margin of the damped system = lam_min + lam_cur.
        lamH_min = lamH_max = None
        if spectrum_m:
            lamH_min, lamH_max = lanczos_extremes(lambda q: hvp(q.float()).double(), p_dim,
                                                  m=int(spectrum_m), device=str(theta.device))
            if verbose:
                print(f"  [lvl {level} spectrum] lam_min~{lamH_min:+.4e} lam_max~{lamH_max:.2f}  "
                      f"PD margin(H+lamI)~{lamH_min + lam_cur:+.4e}")
        level_hist = []
        for inner in range(int(inner_steps)):
            # the CG run + line searches re-fragment/grow the pool; the level HVP closure pins it, so
            # vg's default 4 GiB defrag is too low. Free at the 8 GiB threshold before the gradient
            # eval (mirrors newton_lanczos) so the backward's driver-free gate doesn't trip.
            free_cuda_cache_if_tight(min_free_gib=8.0)
            loss, g, _sv, warm = f(theta.reshape(-1), warm_E=warm)
            _sv = None  # drop the gradient eval's big saved dict (keep only warm=E) -- else two
            #             stale forward-intermediate sets stay live and the next f() trips the gate
            tv = theta.reshape(-1).double()
            tref = theta_ref.reshape(-1).double()
            gF = g.double() + lam_cur * (tv - tref)
            gLn, gFn = float(g.norm()), float(torch.linalg.vector_norm(gF))
            g = None
            F = loss + 0.5 * lam_cur * float((tv - tref).norm() ** 2)
            if gFn < gtol:
                history.append({"stage": "ridge_anneal", "level": level, "inner": inner,
                                "lam": lam_cur, "loss": loss, "F": F, "gLnorm": gLn,
                                "gnorm": gFn, "cg": 0, "status": "converged", "fired": False,
                                "delta": 0.0, "alpha": None, "gp": 0.0, "dF": 0.0, "step_norm": 0.0,
                                "ref_dist": float(torch.linalg.vector_norm(tv - tref)),
                                "lamH_min": lamH_min, "lamH_max": lamH_max,
                                "wall_s": time.perf_counter() - t_start})
                converged = True
                break

            # safe step: cg_witness with witness-driven delta self-correction (the proven pattern
            # from newton_lanczos: on neg_curv, delta <- nu*(delta - cert))
            eta = min(eta_max, gFn ** 0.5)
            delta, fired = 0.0, False
            p = it = status = cert = None
            for _bump in range(int(max_bumps) + 1):
                p, it, status, cert = cg_witness(lambda v, d=delta: A(v) + d * v, -gF,
                                                 tol=eta * gFn, max_iter=max_cg)
                if status != "neg_curv":
                    break
                fired = True
                delta = nu * (delta - cert)
            eff = lam_cur + delta
            if status == "neg_curv":  # bumps exhausted -> scaled gradient
                p, status = -gF / eff, "fallback_gd"
            gp = float(torch.dot(gF, p))
            if gp >= 0.0:  # not a descent direction -> scaled gradient
                p = -gF / eff
                gp = float(torch.dot(gF, p))
                status += "+gd"

            # Armijo backtracking on the deterministic forward loss F = L + penalty
            alpha, accepted, st = 1.0, False, None
            for _ in range(int(ls_max)):
                trial = (tv + alpha * p).to(theta.dtype)
                lt, st = forward_solve(static, trial.reshape(S, 3), col_weights, warm_E=warm)
                Ft = float(lt) + 0.5 * lam_cur * float((trial.double() - tref).norm() ** 2)
                if Ft <= F + c1 * alpha * gp:
                    accepted, warm = True, st["E"]
                    break
                alpha *= 0.5
            st = None  # drop the line search's saved dict (keep only warm=E)
            rec = {"stage": "ridge_anneal", "level": level, "inner": inner, "lam": lam_cur,
                   "loss": loss, "F": F, "gLnorm": gLn, "gnorm": gFn, "cg": it, "status": status,
                   "fired": fired, "delta": delta, "alpha": alpha if accepted else None,
                   "gp": gp, "dF": (Ft - F) if accepted else None,
                   "step_norm": (abs(alpha) * float(torch.linalg.vector_norm(p))) if accepted else 0.0,
                   "ref_dist": float(torch.linalg.vector_norm(tv - tref)),
                   "lamH_min": lamH_min, "lamH_max": lamH_max,
                   "wall_s": time.perf_counter() - t_start}
            level_hist.append(rec)
            history.append(rec)
            if verbose:
                marg = "" if lamH_min is None else f" margin={lamH_min + lam_cur:+.2e}"
                print(f"  [lvl {level} in {inner}] lam={lam_cur:.3e} L={loss:.4f} F={F:.4f} "
                      f"||gL||={gLn:.4e} ||gF||={gFn:.3e} cg={it}({status}) delta={delta:.2e} "
                      f"a={alpha if accepted else float('nan'):.2e} dF={(Ft - F) if accepted else float('nan'):+.2e}{marg}")
            if accepted:
                theta = trial.reshape(S, 3)
            else:  # line search failed at this lambda -> end this level's inner loop
                break
        if converged:
            break
        if theta_ref_mode == "moving":
            theta_ref = theta.clone()
        # adaptive anneal: clean & cheap -> shrink fast; near the edge -> ease off; stalled -> stop
        if not level_hist:
            break
        fired = any(h["fired"] for h in level_hist)
        cg_hard = level_hist[-1]["cg"] >= max_cg
        tol = improve_rtol * max(1.0, abs(level_hist[0]["loss"]))
        improved = level_hist[-1]["loss"] < level_hist[0]["loss"] - tol
        if not fired and not cg_hard and improved:
            lam_next = lam * anneal_fast
        elif fired or cg_hard:
            lam_next = lam * anneal_slow
        else:
            if verbose:
                print(f"[ridge-anneal] level {level}: no progress, witness quiet -> stop")
            break
        if lam_next < lam_floor:
            if verbose:
                print(f"[ridge-anneal] lam {lam_next:.3e} < floor {lam_floor:.3e} -> stop")
            break
        lam = lam_next
    return theta.detach().reshape(S, 3), history, lam_start


# ----------------------------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------------------------
def optimize(static, theta0, col_weights, *, optimizer="adam", lr0=1.0, schedule="adaptive",
             max_steps=300, polish_mode="ridge", max_newton=8, verbose=True,
             polish=None, ridge=None):
    """Full recipe: first-order stage -> optional exact-fp32 Newton polish. Returns (theta_hat, hist).

    Defaults reflect the 666x80 characterization: Adam(lr=1)+adaptive schedule for fast basin
    entry, then a RIDGE-regularized Newton polish. On this problem's flat/indefinite optimum the
    un-ridged (lam=0) Newton bounces ||g|| back up and CG stalls; the ridge/MAP term (auto_lambda)
    makes CG converge (8-11 iters) and gives a monotone endgame to the solver floor.

    ``polish_mode``:
      - ``"ridge"``        : single-lambda ridge Newton (default; monotone, CG-cheap, biased ~0.1%).
      - ``"ridge_anneal"`` : lambda-continuation (slides toward the valley floor; witness-driven stop).
      - ``"lanczos"``      : un-ridged (lam=0) Newton (NOT recommended here -- bounces).
      - ``"none"``         : Adam alone (competitive for pure NLL; 4x cheaper).
    ``polish=False`` / ``ridge=False`` are kept as back-compat aliases for "none" / "lanczos"."""
    if polish is False:
        polish_mode = "none"
    elif ridge is False and polish_mode == "ridge":
        polish_mode = "lanczos"
    t0 = time.perf_counter()
    theta1, h1, _warm = first_order(static, theta0, col_weights, optimizer=optimizer, lr0=lr0,
                                    schedule=schedule, max_steps=max_steps, verbose=verbose,
                                    t0_wall=t0)
    hist = list(h1)
    theta_hat = theta1
    if polish_mode != "none":
        free_cuda_cache_if_tight()
        if polish_mode == "ridge_anneal":
            theta_hat, h2, _lam = ridge_anneal(static, theta1, col_weights, verbose=verbose,
                                               t0_wall=t0)
        else:
            theta_hat, h2, _lam = newton_polish(static, theta1, col_weights,
                                                ridge=(polish_mode == "ridge"),
                                                max_newton=max_newton, verbose=verbose, t0_wall=t0)
        hist += h2
    return theta_hat, hist


def _final_eval(static, theta, col_weights):
    """Fair fp64 loss + ||g|| at theta (so fp32 arms aren't judged by their own noisy evaluator)."""
    S = int(static.state_helpers["S"])
    f = make_value_and_grad(static, col_weights)
    loss, g, _sv, _w = f(theta.detach().reshape(-1).double(), want_grad=True)
    return float(loss), float(g.norm())


# ----------------------------------------------------------------------------------------------
# comparison harness
# ----------------------------------------------------------------------------------------------
def bench(size="666x80", *, max_steps=150, polish=True, max_newton=10, lr0=1.0,
          optimizers=("adam", "adagrad", "rmsprop"),
          schedules=("constant", "cosine", "plateau", "adaptive"), out=None):
    cap, static, theta0, col_weights = load_problem(size)
    col_weights = col_weights.float().contiguous()
    print(f"=== bench {size}: S={int(static.state_helpers['S'])} "
          f"CCPs={int(static.wave_layout['root_row_ids'].numel())} ===")
    arms = []

    def run_arm(name, theta1, h1, t0):
        theta_hat, hist = theta1, list(h1)
        if polish:
            free_cuda_cache_if_tight()
            theta_hat, h2, lam = newton_polish(static, theta1, col_weights, max_newton=max_newton,
                                               verbose=False, t0_wall=t0)
            hist += h2
        nll, gn = _final_eval(static, theta_hat, col_weights)
        rec = {"arm": name, "final_nll_fp64": nll, "final_gnorm_fp64": gn,
               "first_steps": len(h1), "total_wall_s": hist[-1]["wall_s"], "hist": hist}
        arms.append(rec)
        print(f"  {name:22s} NLL={nll:.4f} ||g||={gn:.3e} wall={rec['total_wall_s']:.1f}s "
              f"(first={len(h1)} steps)")
        return rec

    for opt in optimizers:
        for sch in schedules:
            t0 = time.perf_counter()
            theta1, h1, _ = first_order(static, theta0, col_weights, optimizer=opt, lr0=lr0,
                                        schedule=sch, max_steps=max_steps, verbose=False, t0_wall=t0)
            run_arm(f"{opt}/{sch}", theta1, h1, t0)

    # baseline: scipy L-BFGS (fp64)
    try:
        from newton.baselines import lbfgs_scipy
        t0 = time.perf_counter()
        theta_lb, hl = lbfgs_scipy(static, theta0.double(), col_weights.double(),
                                   maxiter=max_steps, verbose=False)
        h1 = [{"stage": "first", "step": i, "loss": float(r["loss"]),
               "gnorm": float(r["gnorm"]), "lr": 0.0,
               "wall_s": time.perf_counter() - t0} for i, r in enumerate(hl)]
        run_arm("lbfgs(scipy)", theta_lb.float(), h1, t0)
    except Exception as e:  # noqa: BLE001
        print(f"  lbfgs(scipy): skipped ({type(e).__name__}: {str(e)[:60]})")

    arms.sort(key=lambda a: a["final_nll_fp64"] if math.isfinite(a["final_nll_fp64"]) else math.inf)
    best = arms[0]["final_nll_fp64"]
    tgt = best + 1e-4 * abs(best)  # relative target (0.01% of best); absolute on a ~1e5 loss is silly
    print(f"\n  best NLL={best:.4f}.  wall to reach NLL<={tgt:.2f} (within 0.01%):")
    for a in arms:
        hit = next((h["wall_s"] for h in a["hist"] if h["loss"] <= tgt), None)
        print(f"    {a['arm']:22s} NLL={a['final_nll_fp64']:.4f} ||g||={a['final_gnorm_fp64']:.2e} "
              f"wall={a['total_wall_s']:6.1f}s  ->target={'%.1fs' % hit if hit else 'never'}")

    if out is None:
        out = f"/tmp/claude-1000/optimize_bench_{size}.json"
    with open(out, "w") as fh:
        json.dump({"size": size, "best_nll": best, "arms": arms}, fh, indent=1)
    print(f"\n  saved {out}")
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="666x80")
    ap.add_argument("--optimizer", default="adam", choices=list(_OPTIMIZERS))
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--schedule", default="adaptive",
                    choices=["constant", "cosine", "plateau", "adaptive"])
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=300)
    ap.add_argument("--polish-mode", dest="polish_mode", default="ridge",
                    choices=["ridge", "ridge_anneal", "lanczos", "none"],
                    help="endgame polish: ridge (default), ridge_anneal (lambda-continuation), "
                         "lanczos (un-ridged), none")
    ap.add_argument("--no-polish", dest="polish", action="store_true",
                    help="alias for --polish-mode none")
    ap.add_argument("--no-ridge", dest="ridge", action="store_true",
                    help="alias for --polish-mode lanczos (un-ridged lam=0 Newton)")
    ap.add_argument("--max-newton", dest="max_newton", type=int, default=8)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    polish_mode = "none" if args.polish else ("lanczos" if args.ridge else args.polish_mode)
    if args.bench:
        bench(args.size, max_steps=args.max_steps, polish=(polish_mode != "none"),
              max_newton=args.max_newton, lr0=args.lr)
        return
    cap, static, theta0, col_weights = load_problem(args.size)
    col_weights = col_weights.float().contiguous()
    theta_hat, hist = optimize(static, theta0, col_weights, optimizer=args.optimizer, lr0=args.lr,
                               schedule=args.schedule, max_steps=args.max_steps,
                               polish_mode=polish_mode, max_newton=args.max_newton)
    nll, gn = _final_eval(static, theta_hat, col_weights)
    print(f"\nFINAL (fp64 eval): NLL={nll:.6f}  ||g||={gn:.4e}  wall={hist[-1]['wall_s']:.1f}s  "
          f"steps={len(hist)}")


if __name__ == "__main__":
    main()
