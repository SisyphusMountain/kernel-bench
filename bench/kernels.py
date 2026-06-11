"""Per-kernel micro-benchmarks: replay one captured Triton-wrapper call in isolation.

Each capture stores the kernel wrapper's pre-call inputs (with argument aliasing preserved and
each buffer stored once) and its golden post-call state (return value AND any tensors it mutates
in place), plus the kernel's run-to-run noise floor. We rebuild the inputs on device, run the
vendored wrapper, check the post-state against golden within that noise floor, then time it. This
gives the kernel optimizer a tight per-kernel signal without running a whole pass.

    python bench/kernels.py                       # every captured kernel, all sizes
    python bench/kernels.py --kernel compute_wave_step
    python bench/kernels.py --phase backward --check-only
"""

from __future__ import annotations

import argparse
import importlib

import torch

from _common import DATA, DEVICE, compare, print_verdicts, timeit
from kbench._capture_io import flatten_tensors, rebuild

NOISE_MARGIN = 4.0  # accept replay within margin * (reference kernel's own nondeterminism)
_POOLS = {}  # size label -> cached content-addressed pool (CPU)


def _resolve(vendored: str):
    mod_path, _, fn = vendored.partition(":")
    return getattr(importlib.import_module(mod_path), fn)


def _pool_for(path):
    """Load (and cache) the shared per-size buffer pool that this record's Refs index into."""
    kdir = path.parent
    if kdir not in _POOLS:
        _POOLS[kdir] = torch.load(kdir / "_pool.pt", map_location="cpu", weights_only=False)
    return _POOLS[kdir]


def _materialize(rec, pool):
    """Rebuild (args, kwargs) on device, re-sharing aliased buffers exactly as captured."""
    return rebuild(rec["pre_ref"], pool, device=DEVICE)


def _flatten_named(args, kwargs, ret) -> dict:
    out = []
    flatten_tensors(args, "arg", out)
    flatten_tensors(kwargs, "kwarg", out)
    flatten_tensors(ret, "ret", out)
    return out


def run_record(path, *, rtol: float, atol: float, check_only: bool,
               warmup: int, iters: int) -> bool:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    fn = _resolve(rec["vendored"])
    pool = _pool_for(path)

    args, kwargs = _materialize(rec, pool)
    ret = fn(*args, **kwargs)
    got = _flatten_named(args, kwargs, ret)

    gargs, gkwargs, gret = rebuild(rec["post_ref"], pool)
    gmap = dict(_flatten_named(gargs, gkwargs, gret))
    noise = rec.get("noise", {})
    valid_masks = rec.get("valid_masks", {})

    verdicts = []
    for name, t in got:
        if name not in gmap:
            continue
        na, nr = noise.get(name, (0.0, 0.0))
        g = gmap[name]
        t_cmp, g_cmp = t, g
        if name in valid_masks:  # drop uninitialized (active-mask-pruned) positions
            m = valid_masks[name]
            t_cmp = t.detach().cpu()[m]
            g_cmp = g[m] if torch.is_tensor(g) else g
        verdicts.append(compare(name, t_cmp, g_cmp, rtol=rtol, atol=atol,
                                noise_abs=na, noise_rel=nr, margin=NOISE_MARGIN))
    noisy = sum(1 for v in noise.values() if v[0] or v[1])
    masked = len(valid_masks)
    tag = (f"  (noise:{noisy}" + (f" masked:{masked}" if masked else "") + " tensors)") if (noisy or masked) else ""
    print(f"\n--- {rec['attr']}  [{rec['phase']}]  {path.name}  int_sig={rec['int_sig']}{tag} ---")
    ok = print_verdicts(verdicts) if verdicts else (print("    (no tensor outputs to compare)") or True)

    if not check_only:
        # Reuse one device copy for timing (values may drift for accumulators -- timing only).
        dargs, dkwargs = _materialize(rec, pool)
        stats = timeit(lambda: fn(*dargs, **dkwargs), warmup=warmup, iters=iters)
        print(f"    time: median={stats['median_ms']:.4f} ms  min={stats['min_ms']:.4f} ms  "
              f"p90={stats['p90_ms']:.4f} ms  (n={stats['iters']})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=str, default=None, help="size label (default: all with kernel captures)")
    ap.add_argument("--kernel", type=str, default=None, help="only this wrapper name")
    ap.add_argument("--phase", type=str, default=None, choices=["forward", "backward"])
    ap.add_argument("--rtol", type=float, default=2e-3)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    labels = [args.size] if args.size else (
        [d.name for d in sorted(DATA.iterdir()) if (d / "kernels").is_dir()] if DATA.exists() else [])
    if not labels:
        raise SystemExit("no kernel captures found -- run capture/capture.py (kernels enabled)")

    all_ok, found = True, 0
    for label in labels:
        kdir = DATA / label / "kernels"
        if not kdir.is_dir():
            continue
        print(f"\n========== kernels [{label}] ==========")
        for path in sorted(kdir.glob("*.pt")):
            if path.name.startswith("_"):  # skip the shared pool file
                continue
            attr = path.name.split("__")[0]
            if args.kernel and attr != args.kernel:
                continue
            if args.phase and torch.load(path, map_location="cpu", weights_only=False)["phase"] != args.phase:
                continue
            found += 1
            all_ok &= run_record(path, rtol=args.rtol, atol=args.atol, check_only=args.check_only,
                                 warmup=args.warmup, iters=args.iters)
    if not found:
        raise SystemExit("no matching kernel captures")
    print(f"\nKERNELS: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
