"""Shared helpers for the benchmark scripts: paths, timing, and golden comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # make `kbench` importable without install

DATA = REPO / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_sizes(kind: str) -> list[str]:
    """Return the captured size labels available for ``kind`` (``whole`` or ``kernels``)."""
    out = []
    if not DATA.exists():
        return out
    for d in sorted(DATA.iterdir()):
        if d.is_dir() and (d / f"{kind}.pt").exists():
            out.append(d.name)
        if kind == "kernels" and d.is_dir() and (d / "kernels").is_dir():
            out.append(d.name)
    # de-dup preserving order
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def load_whole(label: str) -> dict:
    path = DATA / label / "whole.pt"
    if not path.exists():
        raise FileNotFoundError(f"no capture at {path} -- run capture/capture.py first")
    return torch.load(path, map_location="cpu", weights_only=False)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, *, warmup: int = 5, iters: int = 30) -> dict:
    """Time ``fn`` (a no-arg callable) with CUDA events. Returns ms stats."""
    for _ in range(warmup):
        fn()
    cuda_sync()
    if torch.cuda.is_available():
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        cuda_sync()
        ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    else:
        import time
        ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ms.append((time.perf_counter() - t0) * 1e3)
        ms.sort()
    n = len(ms)
    return {
        "median_ms": ms[n // 2],
        "min_ms": ms[0],
        "p90_ms": ms[min(n - 1, int(0.9 * n))],
        "iters": n,
    }


def compare(name: str, got, ref, *, rtol: float, atol: float,
            noise_abs: float = 0.0, noise_rel: float = 0.0, margin: float = 4.0) -> dict:
    """Compare a tensor (or scalar) against golden, inf/NaN-safe and noise-aware.

    Positions where got == ref exactly (including matching +inf / -inf) and positions where
    both are NaN count as agreement -- log-space tensors legitimately hold -inf, and
    (-inf) - (-inf) = NaN must not be read as a mismatch. Only genuinely disagreeing positions
    contribute to max_abs / max_rel; a disagreement against a non-finite value scores inf.

    ``noise_abs`` / ``noise_rel`` are the reference kernel's own run-to-run nondeterminism floor
    (atomic accumulation): the effective tolerance is widened to ``max(tol, margin * noise)`` so
    a faithful rewrite is not failed by irreducible bit-drift.
    """
    got_t = got if torch.is_tensor(got) else torch.as_tensor(got)
    ref_t = ref if torch.is_tensor(ref) else torch.as_tensor(ref)
    got_t = got_t.detach().float().cpu()
    ref_t = ref_t.detach().float().cpu()
    if got_t.shape != ref_t.shape:
        return {"name": name, "ok": False, "reason": f"shape {tuple(got_t.shape)} != {tuple(ref_t.shape)}",
                "max_abs": float("inf"), "max_rel": float("inf")}

    agree = (got_t == ref_t) | (torch.isnan(got_t) & torch.isnan(ref_t))
    disagree = ~agree
    if not bool(disagree.any()):
        return {"name": name, "ok": True, "max_abs": 0.0, "max_rel": 0.0, "reason": ""}

    g, r = got_t[disagree], ref_t[disagree]
    diff = (g - r).abs()
    nonfinite_diff = not bool(torch.isfinite(diff).all())  # inf-vs-finite or NaN-vs-finite
    fdiff = diff[torch.isfinite(diff)]
    max_abs = float("inf") if nonfinite_diff else (float(fdiff.max()) if fdiff.numel() else 0.0)
    rel = diff / r.abs().clamp_min(1e-12)
    frel = rel[torch.isfinite(rel)]
    max_rel = float(frel.max()) if frel.numel() else float("inf")
    atol_eff = max(atol, margin * noise_abs)
    rtol_eff = max(rtol, margin * noise_rel)
    ok = (not nonfinite_diff) and (max_abs <= atol_eff or max_rel <= rtol_eff)
    reason = "non-finite mismatch" if nonfinite_diff else ""
    if not ok and (noise_abs or noise_rel):
        reason = f"exceeds noise floor (abs {noise_abs:.1e}, rel {noise_rel:.1e})"
    return {"name": name, "ok": ok, "max_abs": max_abs, "max_rel": max_rel, "reason": reason}


def print_verdicts(verdicts: list[dict]) -> bool:
    all_ok = True
    for v in verdicts:
        all_ok &= v["ok"]
        flag = "PASS" if v["ok"] else "FAIL"
        extra = f"  ({v['reason']})" if v.get("reason") else ""
        print(f"    [{flag}] {v['name']:<22} max_abs={v['max_abs']:.3e}  max_rel={v['max_rel']:.3e}{extra}")
    return all_ok
