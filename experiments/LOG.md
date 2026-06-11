# Optimization log — target: data/1007x64 (S=2013, C=314932, items=64)

GPU: RTX 4090 (24 GB, 128 SMs, ~72 MB L2, ~1 TB/s DRAM). torch 2.12.0+cu130, triton 3.7.0.
Pristine kernels backed up in `experiments/pristine_kernels/` (no git in this repo).
Tolerance gate: `python bench/check.py` (rtol=atol=2e-3), all sizes.

## Problem shape (1007x64)

- S=2013 columns (binary tree: 1007 leaves + 1006 internal), max_ancestor_depth=34,
  34 compact levels. pi_iters=16, neumann_terms=16, solver=neumann, pruning on.
- 148 waves: 17 "big" waves (W≈8192, 74% of rows), long tail down to W=1.

## Baseline (pristine kernels, 2026-06-11)

| pass | median | min | p90 |
|------|--------|-----|-----|
| forward  | 303.08 ms | 300.95 | 304.94 |
| backward | 485.21 ms | 483.79 | 485.83 |

### nsys kernel breakdown (per iteration, fwd+bwd ≈ 788 ms GPU)

| kernel | ms/iter | share | notes |
|--------|---------|-------|-------|
| `_wave_step_kernel` (fwd) | ~282 | 93% of fwd | 2220 calls = 148 waves × 15 iters |
| `_wave_backward_uniform_2d_jt_kernel` | ~263 | 54% of bwd | 2368 calls = 148 × 16 neumann terms |
| `_wave_backward_uniform_2d_precompute_kernel` | ~56 | | 148 calls |
| `_dts_cross_backward_accum_kernel` | ~40 | | |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | ~34 | | |
| torch reduce/add (`_scatter_accum` of aw*) | ~32 | | frozen glue |
| `_col_grad_from_pibar_self_loop_kernel` | ~24 | | |
| `_wave_backward_uniform_param_store_kernel` | ~22 | | |
| `_dts_eq1_kernel` + `_dts_ge2_stage*` | ~28 | | both passes |

Grid-size bucketing: big waves (W≥4096) hold 69% of wave_step time and 81% of jt time
→ optimize large-launch throughput, not launch overhead.

### ncu, big-wave `_wave_step_kernel` (8192×256)

SM compute 84.6%, L1/LSU 85%, DRAM 22% → **instruction-bound** (gather + exp2 volume),
not bandwidth. Cause: `_pibar_tile` ancestor chase = 8 column-tiles × 34 serial
gather+exp2 rounds per row (~68k ops/row) although `pathsum[s] = expw[s] +
pathsum[parent[s]]` is computable in O(S) per row with the level arrays (which the
backward jt kernel already uses).

### jt kernel analysis (no ncu needed; arithmetic)

block_w=1, BLOCK_S=2048, num_warps=2. The 16 Neumann terms are 16 separate launches;
each re-reads diag/pibar_coeff/p_prime/sl1 (4×W×S fp32 = 264 MB at W=8192) plus
term/corr/v traffic ≈ 530 MB/launch → ~8.5 GB per big wave → DRAM-bound.
All jt work is row-independent (every address is `rows*S + s`), so the 16 terms can
run in ONE launch with an in-program loop; coefficients then load once into registers.
Legal: the Neumann loop lives in `wave_backward.py` (editable); the captured wrapper
boundary is `wave_backward_uniform_fused`.

## Planned experiments

- **B1**: fuse the 16 jt launches into one kernel (in-program Neumann loop). Expect
  ~4-6× on the jt portion (263→~50 ms). Numerics: identical op order → ~bit-exact.
- **A1**: forward `_wave_step_kernel`: replace per-tile ancestor chase with levelized
  top-down pathsum on per-row scratch (expw → 34 level rounds → log2(row_sum-pathsum)).
  Expect ~2× on forward.
- **B2**: same levelized pathsum in `_wave_backward_uniform_2d_precompute_kernel`.
- **B3** (later): fuse precompute / col_grad / param_store around the Neumann loop.
- Then re-profile; candidates: dts kernels, vjp_tree, small-wave latency.

## Results

### Session 1 (2026-06-11)

**Interference note:** mid-session, a user job (`converge_full_archaea_mixed.py`, PID
1024932) shared the GPU and produced bimodal timings (uniform ~1.6× slowdown across all
kernels; clocks stable — pure time-slicing). All numbers below were re-measured after it
exited. Lesson: check `nvidia-smi` for co-tenants before trusting any benchmark run.

| experiment | forward | backward | gate |
|---|---|---|---|
| baseline (pristine) | 303.1 | 485.2 | PASS |
| B1 fused Neumann (16 jt launches → 1, coeffs in regs) | — | | PASS |
| A1 wave_step pathsum via level walk | 282.5 | 377.2 | PASS |
| B2 precompute pathsum via level walk | (incl. above) | | PASS |
| A2 pointer-doubling walk (6 rounds, ping-pong) | 311.3 | 378.1 | PASS (slower → kept levels) |
| tuning: wave_step BLOCK_S 256→512 (warps stay 8) | **267.7** | | PASS |
| tuning: fused jt num_warps 8→2 | | **342.8** | PASS |

Findings:
- B1: per-wave coefficient re-reads were the backward bottleneck (DRAM 10% after, was
  ~85%). Fused kernel is L2-bound at 33% occupancy (registers); num_warps=2 best.
- A1: the level walk costs ~100 ms/iter of forward (measured by N_LEVELS=0 ablation:
  281→181 ms). It is latency-bound: 34 barrier-separated L2 round-trips per call per row,
  ~30/256 useful lanes, only ~6 resident CTAs/SM to hide it.
- A2 (pointer doubling): fewer rounds (6) but 6× full-row L2 store/load traffic — net
  slower than levels. Reverted to levels (kept behind KBENCH_WS_WALK=1).
- Sweeps: wave_step (block_s × warps): 512@8 = 267.7 best; 2048@16 terrible (369).
  jt fused: warps 2 ≈ 4 < 8 ≪ 16.

Env knobs left in: KBENCH_WS_WALK, KBENCH_WS_WARPS, KBENCH_WS_BLOCK_S, KBENCH_JT_WARPS
(defaults = tuned values).

### Session 1, part 2 — register-resident kernels (the big win)

The scratch-based level walk was profiled L2-bound (80% L2, stalls split between
long-scoreboard and barriers): every walk round-trip (store → barrier → gather) goes
through L2 because global stores invalidate L1. Walk variants that reduced rounds
(pointer doubling ×6, hop-grouped rounds ×12) all LOST to the 34-round level walk —
round count was never the binding constraint; scratch traffic was.

The fix that worked: **keep the whole row in registers** (BLOCK_S = next_pow2(S) single
tile) and use `tl.gather` (register/SMEM permutation, no L2) for every within-row
indexed access:

- **A3 `_wave_step_kernel_reg`** (KBENCH_WS_WALK=4, default): row loaded once; max/sum
  computed directly (no running rescale); ancestor path-sum = binary lifting with 6
  in-register gathers; child gathers also tl.gather; STORE_FINAL_PIBAR pass reuses the
  result registers (no reload, no barrier). 117 regs/thread, 33% occupancy — still wins
  by a lot; maxnreg capping (80/96/64) made it worse (spills > occupancy).
- **B4 `_wave_backward_jt_neumann_reg_kernel`** (KBENCH_JT_MODE=1, default): whole
  Neumann loop in registers. The per-term subtree reduction = cumsum over a DFS
  ordering + interval difference (`corr[s] = cum[end-1] - cum[start-1]`), with
  tl.cumsum/tl.gather; DFS tables derived host-side from parent/child arrays, cached.
  Early-exit for pruned (inactive) rows. 263 → 18.5 ms/iter (14×).
- **B5 precompute**: in-register lifting for its ancestor sum (43 → 20 ms/iter).
- **B6 col_grad reg** (same cumsum trick): 24 → ~4 ms/iter.
- **B7 vjp_tree reg**: SLOWER than its level walk (one-shot walk per row; the register
  version's full-tile cumsum + extra store-back of the mutated buffer costs more).
  Kept behind KBENCH_VJP_MODE=1, default 0 (level walk). NOTE: the level-walk version
  mutates pibar_ud in place (part of the captured contract) — the reg version must
  store subtree sums back, which is one reason it loses.

Numerics: all reassociations (single-pass logsumexp, lifting order, cumsum interval
differences) drift ~1e-6 rel; gate (2e-3) passes everywhere incl. per-kernel replays.
Feared cumsum cancellation in signed adjoint sums did not materialize (max_abs ~2e-4).

### Final numbers (RTX 4090, this environment, median of 30)

| size | fwd before → after | bwd before → after |
|------|--------------------|---------------------|
| 1007x64 (S=2013) | 303.1 → **168–179** (~1.75×) | 485.2 → **206** (2.36×) |
| 666x80 (S=1331) | → 139.1 (no same-env baseline) | → 200.0 |
| large (S=119) | 99.4 (pristine, same env) → **86.4** | 273 (README) → **116.3** (2.35×) |
| medium (S=119) | ~47 (pristine, same env) → **46.1** | 97 (README) → **55.6** (1.75×) |
| small (S=119) | ~45.9 (pristine, same env) → **44.9** | 84 (README) → **55.0** (1.53×) |

Caveats: README reference numbers (38/38/92 fwd) do NOT reproduce in this
environment — pristine kernels measured here give ~46/47/99. Same-env pristine
comparison done via `_wave_step_kernel_classic` (KBENCH_WS_REG_MIN_S knob).
Process-to-process variance is ~±5% (allocator/address layout); within-process runs
are stable to <1%. Always check `nvidia-smi` for GPU co-tenants before benchmarking.

### Remaining backward profile (ms/iter at 1007x64, after all changes)

dts_cross 36 · vjp_tree 34 · reduce/elementwise glue (frozen) 37 · param_store 22 ·
precompute 20 · jt 18.5 · dts_eq1/ge2 28 · col_grad ~4 · active_mask 3.
Next candidates: dts_cross + param_store (streaming/atomics; no tree walks — needs a
different idea, e.g. fusing param_store into the jt epilogue to avoid re-reading
Pi/Pibar and recomputing e0..e5), and the dts forward kernels (shared with forward).
Forward is one kernel (`_wave_step_kernel_reg`, ~152 ms/iter): 117 regs → 33%
occupancy, memory ~71%; DRAM floor for pi row in+out per iteration ≈ 76 ms.
