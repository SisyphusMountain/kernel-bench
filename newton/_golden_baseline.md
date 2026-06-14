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

20-step fp32 Newton descent from the fixture theta. Convergence band (3 samples; see "Variance"):

| metric | baseline (pre-opt) | optimized (samples) |
|---|---|---|
| Newton steps | 20 | 20 |
| recorded CG iters | 213 | 168–170 |
| F (start 170130.41) → | 140809.34 | 141855.70 / 142938.31 |
| ‖gF‖ (start 6.14e2) → | 5.70e1 | 7.81e1 / 1.35e2 |
| descent wall | 402.3 s | 322–346 s |

**The recorded CG count undercounts actual HVPs**: each witness negative-curvature bump restarts
CG from scratch, and the step-0 Lanczos (m=10) HVPs aren't counted — so the true per-HVP is
~the isolated steady value below, and the descent wall is trajectory-dependent (not a clean
speedup signal).

### The reproducible runtime metric: steady isolated per-HVP

`prof_target_any.py --time --label 666x80` (one cache build, timed hvp(u); same env state):

| | per-HVP median | per-HVP min | tangent-kernel GPU time (nsys, 3 HVPs) |
|---|---|---|---|
| baseline (num_warps=4 era) | 1235 ms | 1163 ms | `_wave_step_tangent` 1041M ns (29%) |
| + fused self-loop + dts_tree warps | 1172 ms | 1075 ms | `_wave_step_tangent_selfloop` 817M ns (25%) |

≈ **−5% median / −7.6% min** per HVP, fully gated (fp64-identical, hvp gate unchanged). The
tangent self-loop kernel alone dropped −21.5% GPU time (launches 1888→118 per HVP).

### Variance (why the descent endpoint moves ~1%)

The exact-HVP backward uses `atomic_add` (nondeterministic order), and the witness/CG/line-search
make discrete decisions that flip on that ~1e-6 noise (plus the fused self-loop's ~3e-4 fp32
reordering). Two optimized runs differ by ~0.76% in final F (141856 vs 142938) and gN 78 vs 135 —
this is the descent's inherent run-to-run band on this flat/indefinite landscape, NOT a regression.
A `--compare` FN deviation within ~±1.5% is in-band; larger flags a real change. Convergence
quality (F 170130 → ~141–143k, gN reduced 5–8× over 20 steps) is preserved across the optimization.

## 1007x64 (S=2013, p=6039, 64 CCPs) — single HVP runs; full descent memory-blocked

`[C,S]` is 2.36 GiB fp32 (C=314,932). Measured memory by stage (`/tmp/claude-1000/mem_diag_1007.py`):

| stage | alloc (GiB) | peak (GiB) |
|---|---|---|
| after gradient eval (vg) | 0.15 | 8.45 (freed) |
| after empty_cache | 0.15 | reserved 8.54 → 0.21 (defrag works) |
| after forward_solve (sv pi+pibar) | 4.87 | |
| after make_exact_hvp (build cache) | 9.01 | 12.47 |
| **during hvp(u)** | 9.01 | **18.12** |

**The single steady HVP fits and runs at ~1441 ms** (reserve=0; fused-self-loop optimized) — peak
18.12 GiB = ~9 GiB persistent (sv 4.72 + cache ~4.3: per-wave `v_k`, `active_mask`, `dts_r`,
E-side) + ~9.1 GiB HVP transient (`dPi`+`dPibar`+`d_rhs` = 3×2.36 + forward-tangent intermediates
+ scratch). The **full multi-step descent OOMs** ~2.7 GiB above this (a CG/bump-phase spike) on the
23.48 GiB GPU shared with firefox (~0.77 GiB) — it needs to allocate one more 2.36 GiB `[C,S]` with
<1 GiB free.

The gate (`proposal0_memory_gate`) was a *separate*, earlier blocker fixed by `reserve=0` (it sets
`budget = driver-free`); the descent OOM is a true torch OOM, independent of reserve.

**Path to a 1007x64 descent:** eliminate one of the three transient `[C,S]` buffers — `dPibar` is
the candidate: it's recomputable from `dPi` + primal (the same `(dRS−dAS)/denom + dmc` the wave_so
ancestor walk already forms), saving 2.36 GiB at the cost of a per-wave recompute in `wave_so` +
`dts_split_so`. That is a verified-kernel change requiring re-gating (deferred — larger/riskier than
this session's scope). On a dedicated 24 GiB GPU (no firefox) the descent is ~marginal even today.

## Correctness gates (run on `small`, fp64 — the big fixtures OOM in fp64)

Baseline reference values (any kernel change must reproduce these):

```
hvp small : dir0 max_rel=8.32e-5  dir1 max_rel=1.15e-4  symmetry rel_asym=9.41e-4   (tol 5e-4 / 5e-3)
wave_so   : d(A^T v) + d_aw* buckets  max_rel ~3e-9
dts_so    : d_rhs/d_grad_*           max_rel ~3e-9
e_so      : d_grad_*                 max_rel ~3e-9
```

Run: `python -m newton.verify {hvp,wave_so,dts_so,e_so} small`.

## Memory-robustness changes (newton_cg.py)

`newton_lanczos` now manages the HVP cache to keep one point's forward state live at a time and to
defragment the pool on the big fixtures:
- **Drop the old closure** (`hvp_eff = None`) + `free_cuda_cache_if_tight()` before each rebuild —
  otherwise two points' GB-sized cached forward `sv` are live at once and the backward's driver-free
  scratch gate trips after step 0.
- **Reuse the cache when theta is unchanged** (the first Newton step after the Lanczos build, and
  any rejected step): skip the redundant rebuild + its free-old/build-new churn. Saves ~2
  cache builds/descent (evals 60→58 on 666x80) and reduces the memory high-water mark.
- **`empty_cache()` before the initial cache build**: the gradient eval peaks ~8.4 GiB whose freed
  blocks fragment the pool; this returns them to the driver (verified: reserved 8.54→0.21 GiB) so
  the cache build + CG run on a clean pool.

No math change — 666x80 convergence stays in-band across these (FN 141.9–144.0k), all `small` gates
unchanged. On the big fixtures run with `GPUREC_MEMORY_POLICY_RESERVE_GIB=0.0` (the gate's
`budget = driver-free`); 666x80 is insensitive.
