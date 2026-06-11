"""Whole forward-pass benchmark: theta -> loss.

Times the full forward (E-step fixed point + Pi wave forward + NLL) on captured inputs and
checks the result against the golden loss and key intermediates. Run after editing any
forward kernel (wave_step, dts_fused, e_step forward).

    python bench/forward.py                 # all captured sizes, time + check
    python bench/forward.py --size medium   # one size
    python bench/forward.py --check-only    # correctness gate only (exit 1 on mismatch)
"""

from __future__ import annotations

import argparse


from _common import DEVICE, compare, list_sizes, load_whole, print_verdicts, timeit
from kbench.runtime import make_static, move_to_device, run_forward


def run_one(label: str, *, rtol: float, atol: float, check_only: bool,
            warmup: int, iters: int) -> bool:
    cap = load_whole(label)
    static = make_static(cap, DEVICE)
    theta = move_to_device(cap["inputs"]["theta"], DEVICE)
    rw = move_to_device(cap["inputs"]["col_weights"], DEVICE)
    m = cap["meta"]
    print(f"\n=== forward [{label}]  S={m['S']} C={m['C']} items={m['n_items']} ===")

    loss, saved = run_forward(static, theta, rw)
    gold = cap["golden"]
    fsaved = cap["forward_saved"]
    verdicts = [
        compare("loss", loss, gold["loss"], rtol=rtol, atol=atol),
        compare("pi_wave", saved["pi_wave"], fsaved["pi_wave"], rtol=rtol, atol=atol),
        compare("E", saved["E"], fsaved["E"], rtol=rtol, atol=atol),
        compare("pibar_row_max", saved["pibar_row_max"], fsaved["pibar_row_max"], rtol=rtol, atol=atol),
    ]
    ok = print_verdicts(verdicts)
    if not check_only:
        stats = timeit(lambda: run_forward(static, theta, rw), warmup=warmup, iters=iters)
        print(f"    time: median={stats['median_ms']:.3f} ms  min={stats['min_ms']:.3f} ms  "
              f"p90={stats['p90_ms']:.3f} ms  (n={stats['iters']})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=str, default=None, help="single size label (default: all)")
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
    print(f"\nFORWARD: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
