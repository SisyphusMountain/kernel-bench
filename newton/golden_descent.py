"""Golden-standard fp32 Newton descent over the exact-Hessian HVP ("forward + Newton backward").

This is the reference harness for the runtime-optimization work: it runs `newton_lanczos` with
`hvp_mode="exact"` in fp32 from the fixture's own theta, records the full convergence trace and a
per-phase wall-time breakdown, and saves everything to `data/<label>/newton_golden_fp32.pt`.

Purpose:
  * "where we stand" -- end-to-end Newton-descent wall time + convergence on the representative
    fixtures (666x80, 1007x64).
  * a frozen reference: after a kernel optimization, re-run and confirm the descent reaches the
    same final loss / ||gF|| (within the forward solver's truncation floor) at lower wall time.

Usage:
    python -m newton.golden_descent --label 666x80
    python -m newton.golden_descent --label 1007x64 --save
    python -m newton.golden_descent --label 666x80 --compare   # diff against saved golden
"""

from __future__ import annotations

import argparse
import os
import time

# Match the pipeline's gate: the default 1 GiB reserve rejects feasible runs on the big fixtures.
os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB", "0.25")

import torch

from newton.vg import DATA, load_problem
from newton.newton_cg import newton_lanczos


def run_descent(label, *, max_newton, gtol, max_cg, lanczos_m, seed):
    cap, static, theta_fix, col_weights = load_problem(label)
    S = int(static.state_helpers["S"])
    p = 3 * S
    n_ccp = int(static.wave_layout["root_row_ids"].numel())
    theta0 = theta_fix.reshape(S, 3).float().contiguous()
    col_weights = col_weights.float().contiguous()
    print(f"=== golden descent [{label}]  S={S}  p={p}  CCPs={n_ccp}  (fp32, exact HVP) ===")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    theta_hat, hist = newton_lanczos(
        static, theta0, col_weights,
        hvp_mode="exact", max_newton=max_newton, gtol=gtol, max_cg=max_cg,
        lanczos_m=lanczos_m, verbose=True,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # convergence summary
    f0, fN = hist[0]["F"], hist[-1]["F"]
    g0, gN = hist[0]["gnorm"], hist[-1]["gnorm"]
    total_cg = sum(int(h.get("cg", 0)) for h in hist)
    total_evals = hist[-1]["evals"]
    print(f"\n--- summary [{label}] ---")
    print(f"  newton steps : {len(hist)}   total CG iters: {total_cg}   fwd/bwd evals: {total_evals}")
    print(f"  F   : {f0:.6f} -> {fN:.6f}   (dF={fN - f0:+.4e})")
    print(f"  ||gF||: {g0:.4e} -> {gN:.4e}")
    print(f"  wall : {wall:.2f}s   ({wall / max(1, len(hist)):.2f}s/newton-step, "
          f"{wall / max(1, total_cg):.3f}s/cg-iter)")

    return {
        "label": label, "S": S, "p": p, "n_ccp": n_ccp,
        "theta_hat": theta_hat.detach().cpu(),
        "history": hist, "wall_s": wall,
        "F0": f0, "FN": fN, "g0": g0, "gN": gN,
        "total_cg": total_cg, "total_evals": total_evals,
        "config": {"max_newton": max_newton, "gtol": gtol, "max_cg": max_cg,
                   "lanczos_m": lanczos_m, "seed": seed},
    }


def compare(label, result):
    path = DATA / label / "newton_golden_fp32.pt"
    if not path.exists():
        print(f"[compare] no saved golden at {path}; run with --save first")
        return
    g = torch.load(path, map_location="cpu", weights_only=False)
    dth = float((result["theta_hat"] - g["theta_hat"]).abs().max())
    print(f"\n--- compare vs saved golden [{label}] ---")
    print(f"  saved : steps={len(g['history'])} FN={g['FN']:.6f} gN={g['gN']:.4e} "
          f"wall={g['wall_s']:.2f}s cg={g['total_cg']}")
    print(f"  now   : steps={len(result['history'])} FN={result['FN']:.6f} gN={result['gN']:.4e} "
          f"wall={result['wall_s']:.2f}s cg={result['total_cg']}")
    print(f"  dFN={result['FN'] - g['FN']:+.3e}  dgN={result['gN'] - g['gN']:+.3e}  "
          f"max|dtheta_hat|={dth:.3e}")
    speed = g["wall_s"] / max(1e-9, result["wall_s"])
    print(f"  speedup x{speed:.3f}  (wall {g['wall_s']:.2f}s -> {result['wall_s']:.2f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="666x80")
    ap.add_argument("--max-newton", type=int, default=20)
    ap.add_argument("--gtol", type=float, default=1e-2)
    ap.add_argument("--max-cg", type=int, default=20)
    ap.add_argument("--lanczos-m", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write data/<label>/newton_golden_fp32.pt")
    ap.add_argument("--compare", action="store_true", help="diff against the saved golden")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    result = run_descent(args.label, max_newton=args.max_newton, gtol=args.gtol,
                         max_cg=args.max_cg, lanczos_m=args.lanczos_m, seed=args.seed)

    if args.compare:
        compare(args.label, result)
    if args.save:
        path = DATA / args.label / "newton_golden_fp32.pt"
        torch.save(result, path)
        print(f"\nsaved golden -> {path}")


if __name__ == "__main__":
    main()
