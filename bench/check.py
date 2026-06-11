"""Correctness gate: verify forward, backward, and every captured kernel match golden.

This is the hard gate for the kernel optimizer -- it must pass before any timing number is
meaningful. Runs check-only (no timing) across all captured sizes and kernels and exits
nonzero if anything drifts beyond tolerance.

    python bench/check.py                 # everything
    python bench/check.py --rtol 1e-3 --atol 1e-3
"""

from __future__ import annotations

import argparse

import forward as fwd
import backward as bwd
import kernels as kern
from _common import DATA, list_sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rtol", type=float, default=2e-3)
    ap.add_argument("--atol", type=float, default=2e-3)
    args = ap.parse_args()
    rtol, atol = args.rtol, args.atol

    results = {}
    sizes = list_sizes("whole")
    if not sizes:
        raise SystemExit("no captures found in data/ -- run capture/capture.py first")

    print("############## FORWARD ##############")
    fwd_ok = True
    for label in sizes:
        fwd_ok &= fwd.run_one(label, rtol=rtol, atol=atol, check_only=True, warmup=0, iters=1)
    results["forward"] = fwd_ok

    print("\n############## BACKWARD ##############")
    bwd_ok = True
    for label in sizes:
        bwd_ok &= bwd.run_one(label, rtol=rtol, atol=atol, check_only=True, warmup=0, iters=1)
    results["backward"] = bwd_ok

    print("\n############## KERNELS ##############")
    kern_ok = True
    kdirs = [d.name for d in sorted(DATA.iterdir()) if (d / "kernels").is_dir()] if DATA.exists() else []
    for label in kdirs:
        for path in sorted((DATA / label / "kernels").glob("*.pt")):
            if path.name.startswith("_"):  # skip the shared pool file
                continue
            kern_ok &= kern.run_record(path, rtol=rtol, atol=atol, check_only=True, warmup=0, iters=1)
    if not kdirs:
        print("    (no kernel captures)")
    results["kernels"] = kern_ok

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"  {k:<10} {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print(f"  {'ALL PASS' if all_ok else 'GATE FAILED'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
