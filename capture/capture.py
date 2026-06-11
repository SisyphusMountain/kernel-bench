"""Snapshot the golden reference for the bench harness — with NO external library import.

The static problem data + input parameters are committed, frozen fixtures in ``fixtures/<label>.pt``
(plain tensors). This script loads a fixture and computes the golden outputs by running the
**pristine vendored kbench kernels** themselves -- so the golden is exactly what the benchmark's own
(unedited) kernels produce, captured before any optimization. Run it once on the pristine kernels to
set the baseline; the optimizer then runs ``bench/`` (not this) and is checked against the frozen
golden.

Per ``data/<label>/`` it writes:
  * whole.pt        : meta + static + inputs (from the fixture) + forward intermediates + golden
                      (loss, grad_theta, grad_col)
  * kernels/*.pt    : pre/post snapshots of individual Triton-kernel wrapper calls (monkeypatched on
                      kbench where they are used) + a shared content-addressed pool, for the
                      per-kernel micro-bench.

Imports only torch + kbench. The fixtures are the sole externally-derived artifact; rebuilding them
from raw data files is out of scope for this self-contained project.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from kbench._capture_io import dedup_into, flatten_tensors, rebuild  # noqa: E402
from kbench.runtime import make_static, move_to_device, run_backward, run_forward  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = REPO / "data"
FIXTURES = REPO / "fixtures"

# Kernel wrappers to capture: (module where used, attribute, vendored import path, phase).
# Patched on the kbench glue that calls them (forward.py / _implicit_grad.py rebind on import),
# except _launch_e_step_forward_2d which is a module global inside e_step.py.
KERNEL_TARGETS = [
    ("kbench.core.inference.forward", "compute_leaf_initial_wave_step",
     "kbench.core.kernels.wave_step:compute_leaf_initial_wave_step", "forward"),
    ("kbench.core.inference.forward", "compute_wave_step",
     "kbench.core.kernels.wave_step:compute_wave_step", "forward"),
    ("kbench.core.inference.forward", "compute_dts_forward",
     "kbench.core.kernels.dts_fused:compute_dts_forward", "forward"),
    ("kbench.core.kernels.e_step", "_launch_e_step_forward_2d",
     "kbench.core.kernels.e_step:_launch_e_step_forward_2d", "forward"),
    ("kbench.api._implicit_grad", "wave_backward_uniform_fused",
     "kbench.core.kernels.wave_backward:wave_backward_uniform_fused", "backward"),
    ("kbench.api._implicit_grad", "dts_cross_backward_accum_fused",
     "kbench.core.kernels.wave_backward:dts_cross_backward_accum_fused", "backward"),
    ("kbench.api._implicit_grad", "uniform_cross_pibar_vjp_tree_from_ud_fused",
     "kbench.core.kernels.wave_backward:uniform_cross_pibar_vjp_tree_from_ud_fused", "backward"),
]


def _snap(obj):
    """Deep-clone tensors (to CPU) inside a nested structure; leave other values as-is."""
    if torch.is_tensor(obj):
        return obj.detach().clone().cpu()
    if isinstance(obj, dict):
        return {k: _snap(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_snap(v) for v in obj)
    return obj


def _int_signature(args, kwargs):
    """Distinct-wave key: the tuple of non-bool integer scalars in the call (ws, W, S, ...)."""
    sig = []
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, int) and not isinstance(v, bool):
            sig.append(v)
    return tuple(sig)


class _KernelRecorder:
    """Monkeypatch the vendored kernel wrappers, snapshot pre/post state for K distinct waves each."""

    def __init__(self, per_wrapper: int = 4):
        self.per_wrapper = per_wrapper
        self.records = []           # list of dicts (one per captured call)
        self.pool = []              # per-size content-addressed tensor pool (each buffer once)
        self._key2idx = {}          # content_key -> pool index
        self._seen = {}             # attr -> set of int-signatures already captured
        self._orig = []             # (module, attr, original) for restore

    def _wrap(self, module_path, attr, vendored, phase):
        import importlib
        mod = importlib.import_module(module_path)
        orig = getattr(mod, attr)
        self._orig.append((mod, attr, orig))
        seen = self._seen.setdefault(attr, set())

        def wrapper(*args, **kwargs):
            sig = _int_signature(args, kwargs)
            record = (len(seen) < self.per_wrapper) and (sig not in seen)
            if record:
                pre_ref = dedup_into([list(args), dict(kwargs)], self.pool, self._key2idx)
            ret = orig(*args, **kwargs)
            if record:
                seen.add(sig)
                post_ref = dedup_into([list(args), dict(kwargs), ret], self.pool, self._key2idx)
                self.records.append({
                    "attr": attr, "vendored": vendored, "phase": phase, "int_sig": sig,
                    "pre_ref": pre_ref, "post_ref": post_ref,
                })
            return ret

        setattr(mod, attr, wrapper)

    def __enter__(self):
        for mp, attr, vend, phase in KERNEL_TARGETS:
            try:
                self._wrap(mp, attr, vend, phase)
            except (ImportError, AttributeError) as exc:
                print(f"  [kernel-capture] skip {mp}.{attr}: {exc}")
        return self

    def __exit__(self, *exc):
        for mod, attr, orig in self._orig:
            setattr(mod, attr, orig)
        return False


_POISON = (3.0e30, -3.0e30)        # distinct output fills for the two runs
_POISON_THRESH = 1.0e15            # |run0 - run1| above this => position never written, not noise


class _fill_empty:
    """Within the context, make torch.empty/empty_like pre-fill float tensors with ``val``.

    The kernels allocate outputs with torch.empty and write only the active positions. Pre-filling
    with a sentinel means any position the kernel does NOT write keeps the sentinel, so running twice
    with different sentinels reliably exposes uninitialized output positions at any tensor size."""

    def __init__(self, val: float):
        self.val = val

    def __enter__(self):
        self._empty, self._empty_like = torch.empty, torch.empty_like
        val = self.val

        def emp(*a, **k):
            t = self._empty(*a, **k)
            return t.fill_(val) if t.is_floating_point() else t

        def emp_like(*a, **k):
            t = self._empty_like(*a, **k)
            return t.fill_(val) if t.is_floating_point() else t

        torch.empty, torch.empty_like = emp, emp_like
        return self

    def __exit__(self, *exc):
        torch.empty, torch.empty_like = self._empty, self._empty_like
        return False


def measure_noise(records, pool):
    """Per-tensor noise floor + valid mask: which output positions are meaningful, and how noisy.

    Runs each (vendored) kernel twice on fresh inputs, pre-filling every torch.empty output with a
    different sentinel each run so positions the kernel never writes diverge (|diff| > threshold) and
    are marked don't-care; among the written positions, the residual run-to-run difference is the
    atomic noise floor. The bench compares a rewrite against golden only at valid positions, within
    that noise floor.
    """
    import importlib
    for rec in records:
        mp, attr = rec["vendored"].split(":")
        fn = getattr(importlib.import_module(mp), attr)
        try:
            outs = []
            for run in range(2):
                args_list, kwargs = rebuild(rec["pre_ref"], pool, device=DEVICE)
                with _fill_empty(_POISON[run]):
                    ret = fn(*args_list, **kwargs)
                flat = []
                flatten_tensors(args_list, "arg", flat)
                flatten_tensors(kwargs, "kwarg", flat)
                flatten_tensors(ret, "ret", flat)
                outs.append({n: t.detach().float().cpu() for n, t in flat})
        except Exception as exc:  # noqa: BLE001 -- best-effort; default to exact comparison
            print(f"  [noise] {attr}: measurement skipped ({type(exc).__name__}: {exc})")
            rec["noise"], rec["valid_masks"] = {}, {}
            continue
        noise, valid_masks = {}, {}
        for n, a in outs[0].items():
            b = outs[1].get(n)
            if b is None or a.shape != b.shape:
                continue
            diff = (a - b).abs()
            agree = (a == b) | (torch.isnan(a) & torch.isnan(b))
            uninit = (~torch.isfinite(diff)) | (diff > _POISON_THRESH)
            if bool(uninit.any()):
                valid_masks[n] = ~uninit
            disagree = (~agree) & (~uninit)
            if bool(disagree.any()):
                d = diff[disagree]
                fd = d[torch.isfinite(d)]
                rel = (d / b[disagree].abs().clamp_min(1e-12))
                fr = rel[torch.isfinite(rel)]
                noise[n] = (float(fd.max()) if fd.numel() else float("inf"),
                            float(fr.max()) if fr.numel() else float("inf"))
        rec["noise"], rec["valid_masks"] = noise, valid_masks


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def capture_size(label: str, *, capture_kernels: bool, per_wrapper: int):
    t0 = time.time()
    fixture_path = FIXTURES / f"{label}.pt"
    if not fixture_path.exists():
        raise FileNotFoundError(f"no fixture at {fixture_path} (available: "
                                f"{[p.stem for p in sorted(FIXTURES.glob('*.pt'))]})")
    cap = torch.load(fixture_path, map_location="cpu", weights_only=False)
    static = make_static(cap, DEVICE)
    theta = move_to_device(cap["inputs"]["theta"], DEVICE)
    col_weights = move_to_device(cap["inputs"]["col_weights"], DEVICE)
    m = cap["meta"]
    print(f"[{label}] fixture loaded  S={m['S']}  C={m['C']}  items={m['n_items']}  ({time.time()-t0:.1f}s)")

    recorder = _KernelRecorder(per_wrapper) if capture_kernels else None
    with (recorder if recorder is not None else _Null()):
        loss, saved = run_forward(static, theta, col_weights)
        grad_theta, grad_col = run_backward(static, theta, col_weights, saved)

    whole = {
        "meta": cap["meta"],
        "static": cap["static"],
        "inputs": cap["inputs"],
        "forward_saved": {k: _snap(v) for k, v in saved.items()},
        "golden": {"loss": _snap(loss), "grad_theta": _snap(grad_theta), "grad_col": _snap(grad_col)},
    }
    outdir = DATA / label
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(whole, outdir / "whole.pt")
    print(f"[{label}] saved whole.pt  loss={float(loss):.4f}  |g_theta|inf={float(grad_theta.abs().max()):.3e}")

    if recorder is not None:
        measure_noise(recorder.records, recorder.pool)
        kdir = outdir / "kernels"
        kdir.mkdir(parents=True, exist_ok=True)
        torch.save(recorder.pool, kdir / "_pool.pt")  # shared content-addressed buffers (stored once)
        by_attr = {}
        for i, rec in enumerate(recorder.records):
            torch.save(rec, kdir / f"{rec['attr']}__{i:03d}.pt")
            by_attr[rec["attr"]] = by_attr.get(rec["attr"], 0) + 1
        print(f"[{label}] saved {len(recorder.records)} kernel snapshots (pool {len(recorder.pool)} tensors): "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_attr.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    available = [p.stem for p in sorted(FIXTURES.glob("*.pt"))]
    ap.add_argument("--sizes", type=str, default=",".join(available) or "small,medium,large",
                    help="comma list of fixture labels to snapshot (default: all in fixtures/)")
    ap.add_argument("--no-kernels", action="store_true", help="skip per-kernel snapshots")
    ap.add_argument("--kernels-only-on", type=str, default="small",
                    help="capture per-kernel snapshots only for this label (bounds disk)")
    ap.add_argument("--per-wrapper", type=int, default=4, help="distinct waves captured per kernel")
    args = ap.parse_args()

    print(f"device={DEVICE}  fixtures={FIXTURES}\n")
    for label in args.sizes.split(","):
        label = label.strip()
        if not label:
            continue
        capture_kernels = (not args.no_kernels) and (label == args.kernels_only_on)
        capture_size(label, capture_kernels=capture_kernels, per_wrapper=args.per_wrapper)
        print()


if __name__ == "__main__":
    main()
