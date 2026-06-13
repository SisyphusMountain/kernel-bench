# Exact-HVP deep profiling report — nsys + ncu (2026-06-13)

Target: one steady-state analytic exact-Hessian HVP, `small` fixture, fp32, from the plateau
checkpoint. Clean GPU (no co-tenant, 21 GiB free), `RmProfilingAdminOnly=0`. Method: lean target
`/tmp/claude-1000/prof_target.py` — setup + 2 JIT warmup calls, then a `cudaProfilerStart`/NVTX
`steady` region of HVP calls. nsys captured the marked region (3 calls; per-HVP = totals ÷ 3);
ncu profiled the top kernels inside the `steady/` NVTX range. Artifacts in `/tmp/claude-1000/`:
`hvp_nsys.nsys-rep`, `ncu_top.csv`, `ncu_rest.csv`.

## 1. System view (nsys) — confirms launch-bound, surfaces a hidden driver-call cost

Per-HVP GPU-busy time ≈ **85 ms** (sum of kernel durations, single stream → serialized) vs steady
wall ≈ 136 ms ⇒ ~50 ms/HVP of CPU-dispatch + inter-kernel GPU-idle gap.

**GPU kernel time (per HVP, ÷3):**

| Kernel | Time | % | Launches |
|---|---|---|---|
| `_wave_step_tangent_kernel` | 23.6 ms | 28% | 2,272 (16 iters × 142 waves) |
| `_dts_tree_so_kernel` | 11.5 ms | 13% | 141 |
| `_dts_split_so_kernel` | 8.0 ms | 9% | 141 |
| `_dts_cross_backward_accum_kernel` | 6.3 ms | 7% | 141 |
| `_wave_so_kernel` | 5.8 ms | 6% | 142 |
| `_uniform_cross_pibar_vjp_tree_…compact` | 5.8 ms | 6% | 141 |
| `reduce_kernel` (aten) | 4.9 ms | 5% | 1,005 |
| `_dts_tangent_kernel` | 4.4 ms | 5% | 282 |
| `_dts_ge2_stage1_kernel` | 3.4 ms | 4% | 140 |
| vectorized add (aten) | 2.7 ms | 3% | 2,112 |
| others (neumann/precompute/param_store/col_grad/…) | ~8 ms | ~9% | — |

**CUDA API time (per HVP, ÷3) — the CPU side:**

| API | Time | Calls | Source |
|---|---|---|---|
| `cudaLaunchKernel` + `cuLaunchKernelEx` | **17.3 ms** | ~8,650 | kernel-launch dispatch (aten + Triton) |
| `cudaMemcpyAsync` | 2.36 ms | 143 | ~1/wave (D2D/D2H) |
| **`cudaMemGetInfo`** | **2.25 ms** | **285** | `free_cuda_cache_if_tight` + kbench scratch gates |
| `cudaStreamSynchronize` | 0.65 ms | 140 | bicgstab + data-dependent solves |

**Hidden cost surfaced (invisible in torch.profiler):** `cudaMemGetInfo` fires **285×/HVP** at
~7.9 µs each (≈4× a launch). These are blocking driver round-trips. ~142 of them are
`free_cuda_cache_if_tight()` called once per wave in the reverse sweep — on a clean/large-free GPU
it *never* empties the pool, so that half is pure waste. (The rest are kbench's per-kernel scratch
gates, inside the frozen kernels.)

## 2. Kernel view (ncu) — the kernels are LATENCY-bound, not compute/memory-bound

| Kernel | dur | SM % | Mem % | Occ (ach/theo) | grid |
|---|---|---|---|---|---|
| `_wave_step_tangent` (large wave) | 10.6 µs | **54.6** | 54.6 | 79 / 100 | 4208 |
| `_dts_tree_so` | 11.5 µs | 32.9 | 32.9 | 42 / 100 | 802 |
| `_dts_split_so` | 6.8 µs | 7.6 | 33.1 | 22 / 100 | 401 |
| `_dts_cross_backward_accum` | 7.4 µs | 17.5 | 19.1 | 35 / 83 | 401 |
| `_uniform_cross_pibar_vjp…compact` | 10.0 µs | 15.8 | 15.8 | 34 / 100 | 802 |
| `_wave_so` | 6.0 µs | **2.7** | 8.2 | **17** / 83 | **51** |

Every kernel except `_wave_step_tangent` on *large* waves runs at **low occupancy (17–42%) and low
throughput (SM 3–33%, Mem 8–33%)**. Root cause: **grids are tiny** — one program per wave-row, and
most waves have few rows (`_wave_so` here: grid=51 on a ~128-SM GPU). The GPU is under-occupied, so
these kernels are dominated by launch + memory latency, not by useful work. `_wave_step_tangent` is
the exception only because its big-wave instances (grid 4208) fill the machine (SM 55%, occ 79%);
its 2,272 launches still include many tiny waves.

This is the kernel-level confirmation of the system-level diagnosis: the HVP is **launch- and
latency-bound at both levels** — CPU spends 17 ms/HVP dispatching ~8,650 launches, and the launched
kernels are individually too small to fill the GPU.

## 3. What this means for optimization

**Do NOT optimize kernel bodies.** ncu shows they are not compute- or memory-throughput-bound
(SM/Mem mostly <35%); making the math faster won't help while occupancy is 17–42%. The earlier
glue-op fusion (Phase A) was correctly diagnosed as linear/bounded for the same reason.

**Ranked levers:**

1. **(cheap, DONE 2026-06-13) Gate the per-wave `free_cuda_cache_if_tight()`** — now fires every
   `K` waves (`NEWTON_FREE_CACHE_EVERY`, default 32; `1` = old per-wave behaviour). Not deleted:
   stays load-bearing on 1007x64 (memory pressure builds gradually, so a check every 32 waves
   still trips the gate in time). **Measured (small, fp32, clean GPU):** `free_cuda_cache_if_tight`
   `mem_get_info` calls 142 → ~5/HVP; total `cudaMemGetInfo` 285 → 148 (−48% blocking driver
   round-trips). hvp FD gate bit-identical (8.32e-5/1.15e-4/9.41e-4). Wall-clock effect ~1 ms,
   **below the ±2-3 ms steady-HVP noise floor** — confirms the kernel is launch/occupancy-bound,
   not driver-call-bound; this is a serialization-hygiene win, not a throughput lever.

2. **(structural, the real win) Raise per-launch work / cut launch count.** Two complementary
   routes, both attacking the small-grid + 8,650-launch problem:
   - **Batch waves at the same topological level into one kernel launch** (grid = Σ rows over the
     level instead of one launch per wave). This directly fixes the low occupancy (bigger grids)
     AND cuts launch count. Biggest structural payoff; needs per-level layout plumbing and careful
     handling of the self-loop iteration.
   - **CUDA-graph capture** of the per-wave sweep: replays the ~8,650 launches with one CPU
     submission, eliminating the 17 ms dispatch. Caveats: static shapes/pointers per captured
     graph (waves differ in `W` → capture per distinct shape, or pad), and keep bicgstab's
     data-dependent E-solve outside the graph.

3. **(note) `_dts_tree_so` is now the #2 GPU cost (11.5 ms, 13%)** with the A/dA pre-pass I added
   in Phase A; occupancy 42%, balanced SM/Mem. The in-kernel `debug_barrier`s between compact
   levels serialize it. If batching (lever 2) is pursued it absorbs this; standalone tuning is low
   ROI (still latency-bound).

**Bottom line:** the analytic HVP is at 136 ms/HVP and 4.3× faster than the fd-fp64 baseline. The
remaining headroom is **entirely launch/occupancy structure**, not arithmetic. The only
high-payoff moves are level-batching the wave sweep and/or CUDA-graph capture; everything else is
≤ a few percent. Recommend the cheap `free_cuda_cache_if_tight` gate now, and scoping level-batching
as the next real project (it helps both occupancy and launch count, and benefits 1007x64 most).
