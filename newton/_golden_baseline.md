# Golden baseline — fp32 Newton descent over the exact-Hessian HVP

"Where we stand now" reference for the runtime-optimization work on *forward + Newton backward*
(the analytic exact-Hessian HVP, `hvp_mode="exact"`). Regenerate with `newton/golden_descent.py`.

## How to reproduce

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPUREC_MEMORY_POLICY_RESERVE_GIB=0.1 \
  python -m newton.golden_descent --label 666x80 --save        # writes data/666x80/newton_golden_fp32.pt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPUREC_MEMORY_POLICY_RESERVE_GIB=0.1 \
  python -m newton.golden_descent --label 666x80 --compare     # diff a later run vs the saved golden
```

The descent runs `newton_lanczos` (Lanczos-initialized, witness-corrected damped Newton) in fp32
from the fixture's own `theta`, `hvp_mode="exact"`, `max_newton=20 gtol=1e-2 max_cg=20 lanczos_m=10`.
The fixture `theta` is an unconverged operating point with an indefinite Hessian, so the witness
fires (negative curvature) and the early steps damp toward scaled-GD — this is expected and is the
production regime. The descent is monotone in F throughout.

`data/` is gitignored (goldens are regenerable from `fixtures/`); only this summary + the harness
are tracked.

## 666x80 (S=1331, p=3993, 80 CCPs) — GOLDEN  [2026-06-14, clean GPU, no compute co-tenant]

| metric | value |
|---|---|
| Newton steps | 20 |
| total CG iters (= exact HVPs) | 213 |
| forward/backward evals (cache builds + grads) | 60 |
| F | 170130.41 → 140809.34  (dF = −2.93e4, monotone) |
| ‖gF‖ | 6.137e2 → 5.698e1  (10× reduction) |
| **wall** | **402.3 s**  (20.1 s/newton-step, 1.889 s/cg-iter) |

The wall is HVP-dominated: 213 HVPs is the entire cost. Steady isolated per-HVP on 666x80 is
~1.076 s (num_warps=4); the descent's 1.889 s/cg-iter includes the per-step cache rebuilds, the
step-0 Lanczos (10 HVPs), line-search forward solves, and the `empty_cache` releases the memory
fix introduces. The precise kernel-optimization metric is the **clean isolated per-HVP median**
(`prof_target_any.py --time --label 666x80`); the descent wall is the end-to-end validation.

## 1007x64 (S=2013, p=6039, 64 CCPs) — MEMORY-CEILINGED (no golden yet)

The fp32 exact HVP peaks at ~21 GiB on this fixture: within one `hvp(u)` the `[C,S]` buffers
`dPi`, `dPibar`, `d_rhs` (2.36 GiB each, C≈293k) coexist with `sv["pi_wave"]`/`sv["pibar_wave"]`
and the cached per-wave `v_k` — ~6 `[C,S]` tensors live at peak. On the 23.48 GiB GPU (minus
~0.7 GiB held by the user's firefox) this OOMs on a 2.36 GiB allocation mid-descent. Even a single
isolated HVP is within ~0.1 GiB of the gate. **Reducing the HVP's `[C,S]` peak by ~2–3 GiB is a
prerequisite for a 1007x64 golden, and is itself part of the optimization goal** (memory-for-
throughput). Tracked as the first 1007x64 work item.

## Correctness gates (run on `small`, fp64 — the big fixtures OOM in fp64)

Baseline reference values (any kernel change must reproduce these):

```
hvp small : dir0 max_rel=8.32e-5  dir1 max_rel=1.15e-4  symmetry rel_asym=9.41e-4   (tol 5e-4 / 5e-3)
wave_so   : d(A^T v) + d_aw* buckets  max_rel ~3e-9
dts_so    : d_rhs/d_grad_*           max_rel ~3e-9
e_so      : d_grad_*                 max_rel ~3e-9
```

Run: `python -m newton.verify {hvp,wave_so,dts_so,e_so} small`.

## Memory-robustness fix (newton_cg.py, this baseline)

`newton_lanczos` rebuilt `hvp_eff` while the previous point's closure (pinning its GB-sized cached
forward `sv`) was still referenced, so two points' forward state was live at once and the
backward's driver-free scratch gate tripped after step 0 on the big fixtures. Fix: drop the old
closure (`hvp_eff = None`) and `free_cuda_cache_if_tight()` before each cache build, plus once
before the initial Lanczos to release the first gradient eval's pooled scratch. No math change
(all `small` gates unchanged).
