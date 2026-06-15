"""End-to-end specieswise fit: optimize the per-state/species rate tensor ``theta[S,3]`` (pS/pD/pL)
to a CERTIFIED positive-definite minimum.

Motivation (full analysis in ``newton/_optimize_findings.md``). On the representative ``666x80``
fixture the likelihood landscape is treacherous:
  * The gradient pipeline (Adam -> exact-Newton polish) converges to a SADDLE ~173 NLL ABOVE the true
    basin, because the descent direction is orthogonal to the gradient (Newton snaps onto the nearest
    critical point; Adam finds low loss but will not stationarize).
  * The true low-loss basin is reached only by a QUASI-NEWTON with line search (L-BFGS) launched from
    Adam's low-loss valley.
  * The bare-Hessian minimum is NOT strictly PD: it has a genuine non-identifiable (lambda ~ 0) mode
    plus an unresolvable near-zero residual. A small Gaussian/MAP prior of precision ``lambda`` shifts
    every eigenvalue up by ``lambda`` (``H_MAP = H + lambda*I``) and yields a *certified* PD minimum,
    regularizing exactly the flat non-identifiable directions.

Recipe (this script):
    Adam  ->  L-BFGS (descend the ravine)  ->  [optional] deflate deep negative directions
          ->  fixed-lambda MAP polish (lambda from the resolved spectrum)  ->  certify PD + spectrum.

Runs at the CONVERGED objective (``pi_iters=128``, ``neumann_terms=64``, ``NEWTON_TANGENT_SELF_ITERS>=64``
-- the 666x80 capture's pi=16/neumann=16 do NOT converge; see ``kernel-bench-truncation-convergence``).
fp32 throughout (dtype-independent for the geometry; fp64 HVP OOMs on 24 GB). The PD certificate is
rigorous: it lower-bounds ``lambda_min(H_MAP) >= (lanczos_ritz - residual) + lambda`` so the Lanczos
residual cannot hide a negative eigenvalue.

CLI:  python -m newton.specieswise_fit --size 666x80 [--deflate-rounds 1] [--certify-m 200]
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")

import torch

from newton.vg import load_problem, forward_solve, make_value_and_grad, free_cuda_cache_if_tight
from newton.hvp_exact import make_exact_hvp
from newton.cg import lanczos_min_eigpair
from newton.optimize import first_order, ridge_anneal
from newton.baselines import lbfgs_scipy


def spectrum_min(static, theta, col_weights, p, *, m=200, seed=0):
    """Resolved smallest eigenpair of the BARE Hessian at ``theta`` (fp32 HVP, fp64 Lanczos vectors).

    Returns ``(lam_min, v_min, residual)``. The bottom of this fixture's spectrum is near-degenerate,
    so the residual can be large -- it is propagated into the PD certificate rather than ignored.
    """
    S = int(static.state_helpers["S"])
    free_cuda_cache_if_tight()
    _, sv = forward_solve(static, theta, col_weights)
    hvp = make_exact_hvp(static, theta, col_weights, sv)
    Av = lambda v: hvp(v.float()).double()
    lam, vmin = lanczos_min_eigpair(Av, p, m=m, seed=seed)
    Hv = Av(vmin)
    resid = float((Hv - lam * vmin).norm())
    del hvp, sv, Hv
    free_cuda_cache_if_tight(min_free_gib=99.0)
    return lam, vmin, resid


def _deflate_step(static, theta, col_weights, S, p, vmin, g):
    """One negative-curvature deflation: line-min the true loss along (sign-corrected) ``v_min`` to
    flip its curvature positive (reduce the Morse index by 1). Returns ``(theta_new, t_star, dL)``."""
    tv = theta.reshape(-1).double()
    gv = float(torch.dot(g.double(), vmin))
    d = (-vmin if gv > 0 else vmin)
    d = d / d.norm()
    L0 = float(forward_solve(static, theta, col_weights)[0])
    best = (0.0, L0)
    for t in (0.5, 1, 2, 4, 8, 16, 24):
        free_cuda_cache_if_tight(min_free_gib=8.0)
        L = float(forward_solve(static, (tv + t * d).reshape(S, 3), col_weights)[0])
        if L < best[1]:
            best = (t, L)
        elif t > best[0] and L > best[1] + 1e-5:
            break
    tb, Lb = best
    theta2 = ((tv + tb * d).to(theta.dtype).reshape(S, 3).contiguous() if tb > 0 else theta)
    return theta2, tb, Lb - L0


def fit(size="666x80", *, pi_iters=128, neumann_terms=64, adam_steps=300, lbfgs_iters=300,
        deflate_rounds=0, lam_margin=1.3, lam_floor=1e-3, lanczos_m=200, certify_m=200,
        seed=0, verbose=True, out=None):
    """Run the end-to-end specieswise fit. Returns ``(theta_hat, report)`` where ``report`` carries the
    stage-by-stage loss/||g|| trace, the chosen prior ``lambda``, and the PD certificate."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap, static, theta0, col_weights = load_problem(size, dev)
    col_weights = col_weights.float().contiguous()
    S = int(static.state_helpers["S"])
    p = 3 * S
    so = static.solver_options
    so.pi_iters, so.neumann_terms = pi_iters, neumann_terms
    theta = theta0.to(dev).reshape(S, 3).float().contiguous()
    f = make_value_and_grad(static, col_weights, grad_avg_K=2)
    t0 = time.perf_counter()
    trace = []

    def record(stage, th):
        loss, g, _, _ = f(th.reshape(-1))
        rec = {"stage": stage, "loss": float(loss), "gnorm": float(g.norm()),
               "wall_s": time.perf_counter() - t0}
        trace.append(rec)
        if verbose:
            print(f"[{stage:14s}] L={rec['loss']:.4f}  ||g||={rec['gnorm']:.4e}  ({rec['wall_s']:.0f}s)",
                  flush=True)
        return rec, g

    if verbose:
        print(f"=== specieswise fit: {size}  S={S} p={p}  pi={pi_iters} neumann={neumann_terms} "
              f"tangent={os.environ['NEWTON_TANGENT_SELF_ITERS']} ===", flush=True)
    record("init", theta)

    # 1. Adam -- DIVE into the low-loss valley. A constant-LR (oscillating) run visits the deep valley
    #    floor mid-trajectory; we hand the LOWEST-loss point to L-BFGS. (The adaptive schedule stops on
    #    the valley WALL, ~100 NLL higher, from which L-BFGS falls into a worse, steeper-saddle basin.)
    theta, _, _, theta_best = first_order(static, theta, col_weights, optimizer="adam", lr0=1.0,
                                          schedule="constant", max_steps=adam_steps, verbose=False,
                                          t0_wall=t0, return_best=True, early_stop=False)
    record("adam.final", theta)
    theta = theta_best
    record("adam.best", theta)

    # 2. L-BFGS -- descend the ravine to the true basin (the step the gradient/Newton pipeline misses).
    free_cuda_cache_if_tight()
    theta, _ = lbfgs_scipy(static, theta.float(), col_weights, maxiter=lbfgs_iters,
                           dtype=torch.float32, verbose=False)
    theta = theta.reshape(S, 3).float().contiguous()
    record("lbfgs", theta)

    # 3. (optional) deflate the deepest negative-curvature directions -> lets a smaller prior certify.
    for r in range(int(deflate_rounds)):
        lam, vmin, resid = spectrum_min(static, theta, col_weights, p, m=lanczos_m, seed=seed)
        if lam > -lam_floor:
            if verbose:
                print(f"  [deflate {r}] lam_min={lam:+.4e} (>= -{lam_floor:.0e}) -> nothing deep to remove",
                      flush=True)
            break
        free_cuda_cache_if_tight(min_free_gib=8.0)
        _, g = record(f"deflate{r}.pre", theta)
        theta2, tb, dL = _deflate_step(static, theta, col_weights, S, p, vmin, g)
        if verbose:
            print(f"  [deflate {r}] lam_min={lam:+.4e} resid={resid:.2e}  nc-step t*={tb} dL={dL:+.4f}",
                  flush=True)
        if tb == 0.0 or dL >= -1e-4:
            if verbose:
                print(f"  [deflate {r}] no descent (near-degenerate floor) -> stop deflating", flush=True)
            break
        theta = theta2

    # 4. resolve the bare spectrum and pick an initial prior precision lambda > |lam_min| + residual.
    lam_min, _, resid = spectrum_min(static, theta, col_weights, p, m=lanczos_m, seed=seed)
    lam = max(lam_margin * (max(-lam_min, 0.0) + resid), lam_floor)
    theta_ref = theta.clone()
    if verbose:
        print(f"  [spectrum] bare lam_min={lam_min:+.4e} resid={resid:.2e}  -> lambda0={lam:.4f}",
              flush=True)

    # 5+6. fixed-lambda MAP polish + certify, ITERATING lambda until the POST-polish point certifies.
    #      minimize F = L + (lambda/2)||theta - theta_ref||^2 (theta_ref = ravine bottom, proximal).
    #      The polish can drift to a worse-curvature point; a larger lambda both certifies AND keeps the
    #      polish proximal (less drift). Certificate lower-bounds lam_min(H_MAP) = (ritz - resid) + lambda
    #      at the POLISHED point, so the Lanczos residual cannot hide a negative eigenvalue.
    map_rec, gF, cert_lam, cert_resid, hmap_lb, pd = trace[-1], None, lam_min, resid, None, False
    for _attempt in range(6):
        th, _, _ = ridge_anneal(static, theta_ref, col_weights, lam0=lam, theta_ref_mode="fixed",
                                max_levels=1, inner_steps=12, gtol=1e-3, verbose=False, t0_wall=t0)
        th = th.reshape(S, 3).float().contiguous()
        cert_lam, _, cert_resid = spectrum_min(static, th, col_weights, p, m=certify_m, seed=seed)
        hmap_lb = (cert_lam - cert_resid) + lam
        map_rec, g_map = record(f"map(l={lam:.3f})", th)
        gF = g_map.double() + lam * (th.reshape(-1).double() - theta_ref.reshape(-1).double())
        if verbose:
            print(f"  [certify] lambda={lam:.4f} bare lam_min(m{certify_m})={cert_lam:+.4e} "
                  f"resid={cert_resid:.2e} -> lam_min(H_MAP) >= {hmap_lb:+.4e}", flush=True)
        theta = th
        if hmap_lb > 0.0:
            pd = True
            break
        lam *= 2.0   # not certified -> stronger prior, re-polish from theta_ref
    report = {
        "size": size, "S": S, "p": p, "pi_iters": pi_iters, "neumann_terms": neumann_terms,
        "lambda": lam, "trace": trace,
        "final_loss": map_rec["loss"], "final_gnorm_bare": map_rec["gnorm"],
        "final_gnorm_map": float(gF.norm()),
        "bare_lam_min_ritz": cert_lam, "bare_lam_min_resid": cert_resid,
        "map_lam_min_lower_bound": hmap_lb, "certified_pd": pd,
        "wall_s": time.perf_counter() - t0,
        "theta_range": [float(theta.min()), float(theta.max())],
    }
    if verbose:
        print(f"\n=== RESULT  ({report['wall_s']:.0f}s) ===")
        print(f"  MAP minimum  L={report['final_loss']:.4f}  ||g_bare||={report['final_gnorm_bare']:.3e}  "
              f"||grad F||={report['final_gnorm_map']:.3e} (conditioning-floored)")
        print(f"  bare H lam_min(m{certify_m})={cert_lam:+.5e} resid={cert_resid:.2e}")
        print(f"  prior lambda={lam:.4f}  ->  lam_min(H_MAP) >= {hmap_lb:+.5e}  "
              f"=> {'CERTIFIED PD MINIMUM' if pd else 'NOT certified (increase --lam-margin)'}")
    if out:
        with open(out, "w") as fh:
            json.dump(report, fh, indent=1)
        if verbose:
            print(f"  saved report -> {out}")
    return theta.detach(), report


def main():
    ap = argparse.ArgumentParser(description="End-to-end specieswise fit to a certified PD minimum.")
    ap.add_argument("--size", default="666x80")
    ap.add_argument("--pi-iters", dest="pi_iters", type=int, default=128)
    ap.add_argument("--neumann", dest="neumann_terms", type=int, default=64)
    ap.add_argument("--adam-steps", dest="adam_steps", type=int, default=300)
    ap.add_argument("--lbfgs-iters", dest="lbfgs_iters", type=int, default=300)
    ap.add_argument("--deflate-rounds", dest="deflate_rounds", type=int, default=0,
                    help="negative-curvature deflation rounds before MAP (reduces the needed prior)")
    ap.add_argument("--lam-margin", dest="lam_margin", type=float, default=1.3,
                    help="prior lambda = lam_margin * (|lam_min| + residual)")
    ap.add_argument("--lanczos-m", dest="lanczos_m", type=int, default=200)
    ap.add_argument("--certify-m", dest="certify_m", type=int, default=200)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _, report = fit(args.size, pi_iters=args.pi_iters, neumann_terms=args.neumann_terms,
                    adam_steps=args.adam_steps, lbfgs_iters=args.lbfgs_iters,
                    deflate_rounds=args.deflate_rounds, lam_margin=args.lam_margin,
                    lanczos_m=args.lanczos_m, certify_m=args.certify_m, verbose=not args.quiet,
                    out=args.out)
    raise SystemExit(0 if report["certified_pd"] else 1)


if __name__ == "__main__":
    main()
