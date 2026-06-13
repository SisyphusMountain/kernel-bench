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

**Bottom line (small / 3 CCPs):** the analytic HVP is at 136 ms/HVP and 4.3× faster than the
fd-fp64 baseline. The remaining headroom is **entirely launch/occupancy structure**, not
arithmetic — *but see §4: this conclusion is specific to the 3-CCP `small` fixture and inverts at
production scale.*

## 4. Many-CCP regime (666x80, 80 CCPs) — the diagnosis INVERTS (2026-06-13)

`small` has only **3 gene families (CCPs)**, so its waves are 19% mean-filled and 72% of them are
<25% of the 8192 cap — the GPU is starved, hence launch/latency-bound. Re-profiling on **666x80
(80 CCPs)** — different CCPs are different DAGs whose level-profiles interleave, so the 8192 cap
actually binds and the valleys fill — flips the picture completely:

| metric | small (3 CCPs) | 666x80 (80 CCPs) |
|---|---|---|
| wall / HVP | 136 ms | **1237 ms** |
| **GPU-busy / wall** | 85 / 136 = **62%** | **≈1230 / 1237 = ~99%** |
| per-family throughput | 43 ms/family | **15.5 ms/family (2.8× better)** |
| kernel durations | 6–11 µs (tiny) | 40 µs – 1.9 ms (real work) |
| `cudaMemGetInfo` / HVP | 95 (gated) | 123, **0.1% of time** |
| waves <25% full | 72.5% | 53.4% |

**nsys (666x80, ÷3):** `_wave_step_tangent` 411 ms/33% (1888 launches), `_wave_so` 222 ms/18%
(avg **1.88 ms**/launch), `_dts_tree_so` 177 ms/14%, `_dts_split_so` 108 ms/9%, `_dts_tangent`
80 ms/6%, `pibar_vjp` 60 ms/5%, `_dts_cross` 51 ms/4%. The launch-dispatch cost (`cuLaunchKernelEx`
median 2.6 µs) is now invisible against ms-scale kernels.

**ncu (666x80):** `_dts_tree_so` occ 56–65%, SM/Mem 42–49% (healthy); `_dts_split_so` **Mem
55–67%** (memory-bandwidth-bound); `_wave_step_tangent` large instances SM/Mem 38–43%, occ ~17%
(occupancy/register-capped, but 41 µs ≫ 2.6 µs launch → not launch-bound).

**Consequences for the optimization plan:**
- The HVP at production CCP counts is **GPU-throughput-bound, not launch-bound.** GPU-busy ≈ wall
  ⇒ only ~1% idle to recover, so **CUDA-graph capture is near-useless at 80 CCPs** — its entire
  payoff (collapsing dispatch + idle gaps) exists only in the few-CCP regime.
- **Batching more CCPs is itself the highest-leverage move**: it escapes the launch-bound regime
  *and* improves per-family throughput 2.8× (small→666x80). Cost is GPU memory (buffers scale with
  Σ W). This is a capture/scheduler-stage change, transparent to the kernels.
- Once GPU-bound, **kernel-body efficiency becomes the real lever** (the opposite of §3's
  conclusion): `_wave_step_tangent` occupancy/registers (33% of time) and `_dts_split_so` memory
  traffic (Mem 67%). These are worth tuning *at production scale*, were not worth it on `small`.

**Net:** "don't optimize kernel bodies / use CUDA graphs" was correct **only for the 3-CCP
microbenchmark**. The production lever is (1) batch more CCPs (memory-for-throughput, 2.8×/family,
no kernel changes), then (2) tune the now-saturated hot kernels (`_wave_step_tangent` occupancy,
`_dts_split_so` bandwidth). CUDA graphs drop to a niche win for small-batch / interactive use.

### 4a. Roofline per hot kernel (666x80, large/leaf-side instances, ncu `--set roofline`)

Each kernel's *largest* (leaf-side) instances — the ones that dominate runtime — sampled via
`--launch-skip`. %time is the nsys per-HVP share from §4.

| kernel | %time | large-inst dur | SM% | Mem% | DRAM% | L1/TEX% | L2% | **bound by** |
|---|---|---|---|---|---|---|---|---|
| `_wave_step_tangent` | 33% | 44 µs | 37 | 37 | 20 | **47** | 12 | **occupancy/latency** (occ ~17%, no pipe >47%, DRAM cool) |
| `_wave_so` | 18% | (big) 6.1 ms | 17 | 80 | 12 | 40 | **80** | **L2 bandwidth** (data fits in L2, DRAM cool → reuse problem) |
| `_dts_tree_so` | 14% | 2.2 ms | 71 | 71 | 45 | **72** | 64 | **near-saturated** (SM+L1 ~72%) → low tuning ROI |
| `_dts_split_so` | 9% | 1.6 ms | 15 | 67 | **67** | 21 | 41 | **DRAM bandwidth** (cut bytes/passes) |

**Ranked kernel-tuning targets (only relevant once batched into the GPU-bound regime):**
1. **`_wave_step_tangent` (33%, biggest + most under-utilized). → DONE: num_warps 8→4, −7.3% total
   HVP (2026-06-13).** Occupancy is *not* raisable cheaply: at BLOCK_S=2048 the kernel is pinned to
   1 block/SM by **206 reg/thread AND 9216 B shared** (both → Block Limit 1), giving 8 warps/SM =
   16.67% theoretical = 16.65% achieved (hits the cap exactly). Sweeping `num_warps` {2,4,8,16,32}:
   **4 wins** (1165→1080 ms total HVP on 666x80, bit-identical FD gate). Counter-intuitively the win
   is NOT occupancy — warps=4 also gives 8 active warps/SM (2 blocks × 4), same 16.67% — it's **more
   elements/thread (16 vs 8 → better ILP) + 2 resident blocks/SM hiding the cold-DRAM (20%) latency**.
   warps=2 spills catastrophically (1386 ms); 16/32 are slower than 4. Set as default in
   `wave_tangent.py` (env `NEWTON_WST_NUM_WARPS`). Neutral on small (S=119). Further headroom would
   need cutting the 206-reg footprint (the kernel does a lot per row: 6-term LSE + ancestor walk over
   MAX_ANC_DEPTH=29 + splits) — higher effort, deferred.
2. **`_wave_so` (18%), L2-bound (80%).** DRAM only 12% → it's not moving too much *total* data, it's
   re-fetching through L2. Lever: improve locality/reuse (more register/shared residency, restructure
   the access pattern) to cut L2 round-trips.
3. **`_dts_split_so` (9%), DRAM-bandwidth-bound (67%).** Lever: reduce DRAM bytes — fuse passes,
   better layout, fewer Pi/dPi-sized reads.
4. **`_dts_tree_so` (14%) is already ~72% saturated on SM+L1** → leave it; low ROI despite being #3
   by time. (The Phase-A A/dA pre-pass + per-level `debug_barrier`s are not the limiter; throughput
   is genuinely high.)
