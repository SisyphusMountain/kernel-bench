"""Whole backward-pass benchmark: forward intermediates -> grad_theta, grad_col.

Times the implicit-diff adjoint solve + parameter VJP on the GOLDEN forward intermediates
(so the backward is measured and checked independently of the forward) and compares the
gradients against golden. Run after editing any backward kernel (wave_backward, e_step
backward).

    python bench/backward.py
    python bench/backward.py --size large --check-only
"""

from __future__ import annotations

import argparse


from _common import DEVICE, compare, list_sizes, load_whole, print_verdicts, timeit
from kbench.runtime import make_static, move_to_device, run_backward


def run_one(label: str, *, rtol: float, atol: float, check_only: bool,
            warmup: int, iters: int) -> bool:
    cap = load_whole(label)
    static = make_static(cap, DEVICE)
    theta = move_to_device(cap["inputs"]["theta"], DEVICE)
    rw = move_to_device(cap["inputs"]["col_weights"], DEVICE)
    saved = {k: move_to_device(v, DEVICE) for k, v in cap["forward_saved"].items()}
    m = cap["meta"]
    print(f"\n=== backward [{label}]  S={m['S']} C={m['C']} items={m['n_items']} "
          f"neumann={m['solver_options']['neumann_terms']} ===")

    grad_theta, grad_col = run_backward(static, theta, rw, saved)
    gold = cap["golden"]
    verdicts = [
        compare("grad_theta", grad_theta, gold["grad_theta"], rtol=rtol, atol=atol),
        compare("grad_col", grad_col, gold["grad_col"], rtol=rtol, atol=atol),
    ]
    ok = print_verdicts(verdicts)
    if not check_only:
        stats = timeit(lambda: run_backward(static, theta, rw, saved), warmup=warmup, iters=iters)
        print(f"    time: median={stats['median_ms']:.3f} ms  min={stats['min_ms']:.3f} ms  "
              f"p90={stats['p90_ms']:.3f} ms  (n={stats['iters']})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=str, default=None)
    ap.add_argument("--rtol", type=float, default=2e-3)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    sizes = [args.size] if args.size else list_sizes("whole")
    if not sizes:
        raise SystemExit("no captures found in data/ -- run capture/capture.py first")
    all_ok = True
    for label in sizes:
        all_ok &= run_one(label, rtol=args.rtol, atol=args.atol, check_only=args.check_only,
                          warmup=args.warmup, iters=args.iters)
    print(f"\nBACKWARD: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
