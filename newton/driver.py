"""Newton-CG driver: optimize the NLL over theta and compare against baselines.

    python newton/driver.py --size small                 # GGN Newton-CG, fp32
    python newton/driver.py --size small --method fd_hessian
    python newton/driver.py --size small --baseline       # + GD and L-BFGS
    python newton/driver.py --size small --verify          # tangent/GGN correctness gate

The analytic GGN/Fisher HVP is the requested method; ``fd_hessian`` (true Hessian via FD of the
gradient) and the GD/L-BFGS baselines are provided for calibration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from newton.vg import load_problem, make_value_and_grad  # noqa: E402
from newton.newton_cg import newton_cg, newton_lanczos, newton_tr  # noqa: E402


def _gnorm(static, theta, col_weights):
    f = make_value_and_grad(static, col_weights)
    loss, g, _, _ = f(theta.reshape(-1))
    return float(loss), float(g.norm())


def bench_hvp(static, theta, col_weights, n=10):
    """Time one fp32 GGN HVP (the production Newton path)."""
    from newton.ggn import make_ggn_hvp
    loss, sv = None, None
    f = make_value_and_grad(static, col_weights)
    _loss, _g, sv, _ = f(theta.reshape(-1))
    hvp = make_ggn_hvp(static, theta, col_weights, sv)
    p = 3 * int(static.state_helpers["S"])
    v = torch.ones(p, device=theta.device, dtype=torch.float32)
    hvp(v); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        hvp(v)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def run(args):
    cap, static, theta, col_weights = load_problem(args.size)
    S = int(static.state_helpers["S"])
    print(f"=== Newton-CG [{args.size}]  S={S}  p={3*S}  method={args.method} ===")

    if args.verify:
        from newton.verify import (check_e_tangent, check_wave_step_tangent,
                                    check_dts_tangent, check_forward_tangent, check_ggn)
        ok = (check_e_tangent(args.size) & check_wave_step_tangent(args.size, split=True)
              & check_wave_step_tangent(args.size, split=False) & check_dts_tangent(args.size)
              & check_forward_tangent(args.size) & check_ggn(args.size))
        print(f"VERIFY: {'ALL PASS' if ok else 'FAILURES'}")
        return

    t0 = time.time()
    if args.method == "lanczos":
        # Lanczos-initialized, witness-corrected damped Newton descent (fp64)
        theta_opt, hist = newton_lanczos(
            static, theta.double(), col_weights.double(), sigma=args.sigma,
            lanczos_m=args.lanczos_m, max_newton=args.iters, gtol=args.gtol,
            max_cg=args.max_cg, hvp_mode=args.hvp, verbose=True,
        )
        nfires = sum(len(h.get("witness_certs", [])) for h in hist)
        print(f"witness fires: {nfires}  grad-evals total: {hist[-1]['evals']}")
    elif args.method == "fd_tr":
        # trust-region Newton on the true (FD) Hessian, fp64 (precise loss -> exact rho test)
        theta_opt, hist = newton_tr(
            static, theta.double(), col_weights.double(), curvature="fd_hessian",
            max_newton=args.iters, gtol=args.gtol, delta0=args.delta0, max_cg=args.max_cg,
            verbose=True,
        )
    else:
        ms = bench_hvp(static, theta, col_weights)
        print(f"GGN HVP: {ms:.1f} ms/call (fp32)")
        theta_opt, hist = newton_cg(
            static, theta, col_weights, curvature=args.method, max_newton=args.iters,
            gtol=args.gtol, lam0=args.lam0, max_cg=args.max_cg, eta_max=args.eta, verbose=True,
        )
    dt = time.time() - t0
    l0, g0 = hist[0]["loss"], hist[0]["gnorm"]
    lf, gf = _gnorm(static, theta_opt, col_weights)
    print(f"\nNewton-CG[{args.method}]: loss {l0:.2f} -> {lf:.2f}  ||g|| {g0:.2e} -> {gf:.2e}  "
          f"in {len(hist)} steps, {dt:.1f}s")

    if args.baseline:
        from newton.baselines import gd, lbfgs_scipy
        print("\n--- baselines ---")
        t0 = time.time()
        th_gd, hg = gd(static, theta, col_weights, lr=args.gd_lr, steps=args.gd_steps)
        lg, gg = _gnorm(static, th_gd, col_weights)
        print(f"GD ({len(hg)} steps): loss -> {lg:.2f}  ||g|| -> {gg:.2e}  {time.time()-t0:.1f}s")
        t0 = time.time()
        th_lb, hl = lbfgs_scipy(static, theta, col_weights, maxiter=args.lbfgs_iters)
        print(f"L-BFGS ({len(hl)} evals): loss -> {hl[-1]['loss']:.2f}  "
              f"||g|| -> {hl[-1]['gnorm']:.2e}  {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="small")
    ap.add_argument("--method", default="ggn", choices=["ggn", "fd_hessian", "fd_tr", "lanczos"])
    ap.add_argument("--sigma", type=float, default=0.01, help="lam_damp0 = sigma*lam_max (lanczos)")
    ap.add_argument("--lanczos-m", dest="lanczos_m", type=int, default=10)
    ap.add_argument("--hvp", default="fd", choices=["fd", "exact"], help="HVP backend (lanczos method)")
    ap.add_argument("--iters", type=int, default=20, help="max Newton steps")
    ap.add_argument("--gtol", type=float, default=1e-2)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--delta0", type=float, default=1.0, help="initial trust radius (fd_tr)")
    ap.add_argument("--max-cg", dest="max_cg", type=int, default=40)
    ap.add_argument("--eta", type=float, default=0.05, help="CG forcing (residual reduction)")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--gd-lr", dest="gd_lr", type=float, default=5e-3)
    ap.add_argument("--gd-steps", dest="gd_steps", type=int, default=200)
    ap.add_argument("--lbfgs-iters", dest="lbfgs_iters", type=int, default=120)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
