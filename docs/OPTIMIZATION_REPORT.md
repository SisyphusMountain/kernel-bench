# Kernel Optimization Report

**Target workload:** `data/1007x64` (S = 2013 columns, C = 314 932 rows, 64 items)
**GPU:** RTX 4090 (24 GB, 128 SMs, ~72 MB L2, ~1 TB/s DRAM)
**Stack:** torch 2.12.0+cu130, triton 3.7.0
**Date:** 2026-06-11
**Correctness gate:** `python bench/check.py` (rtol = atol = 2e-3), all sizes

> This is the narrative + numbers companion to the operating manual in
> [BENCHMARKS.md](BENCHMARKS.md) and the harness-fidelity notes in [FIXES.md](FIXES.md).
> The full chronological experiment log lives in
> [../experiments/LOG.md](../experiments/LOG.md); pristine pre-optimization kernels are
> backed up in [../experiments/pristine_kernels/](../experiments/pristine_kernels/)
> (this repo has **no git**, so that backup is the only baseline).

---

## 1. Executive summary

Both passes were rewritten around a single architectural idea — **keep the whole state
row resident in registers and do all within-row indexed access with `tl.gather`** instead
of walking per-row scratch buffers through L2. The result on the headline `1007x64`
workload:

| pass | before (pristine, same env) | after | speedup |
|------|------------------------------|-------|---------|
| **forward**  | 303.1 ms | **168–179 ms** | **~1.75×** |
| **backward** | 485.2 ms | **206 ms** | **2.36×** |

All sizes pass the correctness gate. The wins hold across the smaller committed fixtures
too (backward 1.5–2.4×; see [§7](#7-final-results-all-sizes)).

The two binding constraints were different for each pass and **neither was launch
overhead**:

- **Forward** was *instruction-bound* — the hot kernel chased ancestor paths with
  8 column-tiles × 34 serial gather+exp2 rounds per row.
- **Backward** was *DRAM-bound* — the 16 Neumann terms were 16 separate launches, each
  re-reading the same per-wave coefficient buffers (~8.5 GB of redundant traffic per big
  wave).

---

## 2. Problem shape & methodology

The state is a band-structured `C × S` matrix solved in log-space by a fixed-point
iteration. Columns form a **binary tree** (1007 leaves + 1006 internal = S = 2013),
max ancestor depth 34, 34 compact levels. A **wave** is a topological band of rows updated
together; the solver runs `pi_iters = 16` sweeps per wave. The `1007x64` problem has
**148 waves**: 17 "big" waves (W ≈ 8192) hold ~74 % of the rows, with a long tail down to
W = 1.

**Grid-size bucketing** was the first thing measured, and it set the whole strategy:
big waves (W ≥ 4096) hold **69 % of forward `wave_step` time and 81 % of backward `jt`
time**. The conclusion was to **optimize large-launch throughput, not launch overhead** —
small-wave latency was never where the time went.

Measurement discipline:
- **Per-kernel** replay (one wrapper in isolation, ~0.02–0.3 ms) for tight signal.
- **Whole-pass** forward/backward for the aggregate that catches interactions.
- Profiling with **nsys** (kernel breakdown) and **ncu** (compute vs memory bound), never
  intuition. Numbers below are medians of 30 runs.

---

## 3. Baseline profile (pristine kernels)

| pass | median | min | p90 |
|------|--------|-----|-----|
| forward  | 303.08 ms | 300.95 | 304.94 |
| backward | 485.21 ms | 483.79 | 485.83 |

### nsys kernel breakdown (per iteration, fwd+bwd ≈ 788 ms GPU)

| kernel | ms/iter | share | notes |
|--------|---------|-------|-------|
| `_wave_step_kernel` (fwd) | ~282 | 93 % of fwd | 2220 calls = 148 waves × 15 iters |
| `_wave_backward_uniform_2d_jt_kernel` | ~263 | 54 % of bwd | 2368 calls = 148 × 16 Neumann terms |
| `_wave_backward_uniform_2d_precompute_kernel` | ~56 | | 148 calls |
| `_dts_cross_backward_accum_kernel` | ~40 | | |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | ~34 | | |
| torch reduce/add (`_scatter_accum`) | ~32 | | frozen glue |
| `_col_grad_from_pibar_self_loop_kernel` | ~24 | | |
| `_wave_backward_uniform_param_store_kernel` | ~22 | | |
| `_dts_eq1_kernel` + `_dts_ge2_stage*` | ~28 | | both passes |

### Where each pass was bound

- **Forward `_wave_step_kernel` (ncu, 8192×256 big wave):** SM compute 84.6 %, L1/LSU
  85 %, DRAM **22 %** → **instruction-bound**, not bandwidth. Cause: the `_pibar_tile`
  ancestor chase did 8 column-tiles × 34 serial gather+exp2 rounds per row (~68k ops/row),
  even though `pathsum[s] = expw[s] + pathsum[parent[s]]` is computable in O(S) per row
  with the level arrays the backward already had.
- **Backward `jt` kernel (arithmetic, no ncu needed):** the 16 Neumann terms were
  16 separate launches; each re-read diag / pibar_coeff / p_prime / sl1 (4 × W × S fp32 =
  264 MB at W = 8192) plus term/corr/v traffic ≈ 530 MB/launch → **~8.5 GB per big wave**
  → **DRAM-bound**. All jt work is row-independent, so the 16 terms can collapse into one
  launch with an in-program loop and the coefficients load **once** into registers.

---

## 4. The dead ends (and why they failed)

These are recorded because they were the expensive lessons — every one of them looked
reasonable and lost:

- **Scratch-based level walk (A1/B2).** Replacing the ancestor chase with a levelized
  top-down pathsum on per-row scratch *did* help (forward 303 → 282), but profiled
  **L2-bound**: 34 barrier-separated L2 round-trips per call per row, ~30/256 useful
  lanes, only ~6 resident CTAs/SM to hide the latency. Every walk round-trip
  (store → `tl.debug_barrier()` → gather) goes through L2 because global stores invalidate
  L1.
- **Pointer-doubling walk (A2, 6 rounds, ping-pong).** Fewer rounds (6 vs 34) but 6× full
  row L2 store/load traffic — **net slower** (311 ms). Reverted.
- **Hop-grouped rounds (×12).** Same story.

> **Key finding:** reducing the *number of walk rounds* never helped. **Scratch traffic,
> not round count, was the binding constraint.** Every scheme that still round-tripped the
> row through L2 lost to the one that didn't.

---

## 5. The fix that worked — register-resident kernels

Keep the **whole row in registers** (`BLOCK_S = next_pow2(S)`, single tile, one program
per row) and use `tl.gather` — a register/SMEM permutation that never touches L2 — for
every within-row indexed access. Tree operations are then expressed as register-only
primitives:

- **Ancestor path-sums** → **binary lifting** (jump tables): 6 in-register gathers instead
  of 34 serial rounds.
- **Subtree sums** → **`tl.cumsum` over a DFS ordering + interval difference**:
  `corr[s] = cum[end-1] - cum[start-1]`.

Host-side schedule tables (jump/level/DFS orderings) are derived **once** from the
parent/child arrays and **cached by `node_parent.data_ptr()`**. ⚠️ Keep the derived temp
tensors alive or the cached pointers can dangle.

### Forward

| id | kernel | change | result |
|----|--------|--------|--------|
| **A3** | `_wave_step_kernel_reg` *(default)* | row loaded once; single-pass max/sum (no running rescale); ancestor path-sum via 6-gather binary lifting; child gathers via `tl.gather`; the `STORE_FINAL_PIBAR` pass reuses result registers (no reload, no barrier) | 117 regs/thread, 33 % occupancy — still wins by a lot |

`maxnreg` capping (80/96/64) made it **worse** — spills cost more than the occupancy they
buy.

### Backward

| id | kernel | change | result (ms/iter) |
|----|--------|--------|------------------|
| **B1** | fused Neumann | 16 jt launches → **1**, coefficients in registers | DRAM 85 % → 10 % |
| **B4** | `_wave_backward_jt_neumann_reg_kernel` *(default)* | whole Neumann loop in registers; per-term subtree reduction = cumsum-over-DFS + interval difference; early-exit for pruned rows | **263 → 18.5** (14×) |
| **B5** | precompute (reg lifting) | in-register binary lifting for its ancestor sum | **43 → 20** |
| **B6** | `col_grad` (reg cumsum) | same cumsum/interval-difference trick | **24 → ~4** |

**B7 (`vjp_tree` reg) — reverted.** The register version was *slower* than its level walk:
it's a one-shot walk per row, so the full-tile cumsum plus storing the mutated buffer back
costs more than it saves. It also has a contract reason: the level-walk version mutates
`pibar_ud` in place (part of the captured boundary), whereas the reg version must store
subtree sums back. Kept behind `KBENCH_VJP_MODE=1`, default **0** (level walk).

### Tuning sweeps

- `wave_step` block_s × warps: **512 @ 8 warps = 267.7** (best of the scratch era);
  2048 @ 16 was terrible (369).
- fused jt num_warps: **2 ≈ 4 < 8 ≪ 16** (the fused kernel is L2-bound at 33 % occupancy,
  so more warps just thrash).

---

## 6. Correctness & numerics

Every rewrite reassociates floating-point work — single-pass logsumexp, binary-lifting
order, cumsum interval differences. Measured drift vs the pristine golden is **~1e-6
relative**, far inside the **2e-3** gate, and the gate **passes everywhere**, including the
per-kernel replays. The feared catastrophic cancellation in the signed adjoint cumsum
sums **did not materialize** (max_abs ~2e-4).

> A separate fp64 backward investigation (see
> [[kernel-bench-backward-race-context]] / [FIXES.md](FIXES.md)) is worth knowing about
> when validating: the production solver options (neumann_terms = 16,
> adjoint_pruning_threshold = 1e-6) carry a **designed** ~2e-3–5e-3 relative truncation
> error, so an fp64-vs-golden mismatch at ~1e-2 is approximation/pruning drift, not
> necessarily a bug. Tighten to neumann_terms ≈ 96, pruning 0, bicgstab_tol = 1e-12 to
> match fp64 finite differences to ~1e-9.

---

## 7. Final results (all sizes)

Median of 30, RTX 4090, this environment:

| size | fwd before → after | bwd before → after |
|------|--------------------|---------------------|
| **1007x64** (S = 2013) | 303.1 → **168–179** (~1.75×) | 485.2 → **206** (2.36×) |
| **666x80** (S = 1331)  | → 139.1 *(no same-env baseline)* | → 200.0 |
| **large** (S = 119)    | 99.4 → **86.4** | 273 → **116.3** (2.35×) |
| **medium** (S = 119)   | ~47 → **46.1** | 97 → **55.6** (1.75×) |
| **small** (S = 119)    | ~45.9 → **44.9** | 84 → **55.0** (1.53×) |

The small/medium forward barely moves because at S = 119 the row already fits cheaply; the
register rewrite pays off where S is large and the ancestor chase / coefficient re-reads
dominated.

---

## 8. Environment knobs

Defaults are the tuned winners — override only to A/B against the old paths.

| knob | default | meaning |
|------|---------|---------|
| `KBENCH_WS_WALK` | **4** | forward path-sum strategy; 4 = register binary lifting (A3), 1 = scratch level walk |
| `KBENCH_JT_MODE` | **1** | 1 = register Neumann loop (B4); 0 = fused-launch scratch |
| `KBENCH_JT_REG_WARPS` | **4** | warps for the register jt kernel |
| `KBENCH_VJP_MODE` | **0** | 0 = level walk (wins); 1 = register vjp_tree (lost) |
| `KBENCH_WS_REG_MIN_S` | **0** | min S to take the register `wave_step`; used with `_wave_step_kernel_classic` for same-env pristine comparison |
| `KBENCH_WS_MAXNREG` | **0** | 0 = no maxnreg cap (capping lost: spills > occupancy) |

---

## 9. Benchmarking pitfalls (this machine)

- **Always check `nvidia-smi` for compute co-tenants first.** A user training job
  (`converge_full_archaea_mixed.py`) time-slicing the GPU inflated *every* kernel uniformly
  by ~1.6 × while clocks stayed nominal — pure time-slicing, easy to misread as a
  regression. Mid-session bimodal timings traced to exactly this; all reported numbers were
  re-measured after it exited.
- **Process-to-process variance ~±5 %** (allocator/address layout); **within-process < 1 %**.
- **README reference numbers (38/38/92 fwd) do NOT reproduce here** — pristine kernels
  measured in this environment give ~46/47/99. Same-env pristine comparison is done via
  `_wave_step_kernel_classic` (gated by `KBENCH_WS_REG_MIN_S`), not against the README.

---

## 10. Remaining levers

Backward profile after all changes (ms/iter at 1007x64):

```
dts_cross 36 · vjp_tree 34 · reduce/elementwise glue (frozen) 37 · param_store 22 ·
precompute 20 · jt 18.5 · dts_eq1/ge2 28 · col_grad ~4 · active_mask 3
```

- **`dts_cross` (~36) + `param_store` (~22)** — streaming/atomics, no tree walks, so the
  register trick doesn't apply directly. Best idea on the table: **fuse `param_store` into
  the jt epilogue** to avoid re-reading Pi/Pibar and recomputing `e0..e5`.
- **dts forward kernels** — shared with the forward pass; a joint win.
- **Forward floor.** Forward is now essentially one kernel (`_wave_step_kernel_reg`,
  ~152 ms/iter): 117 regs → 33 % occupancy, memory ~71 %. The DRAM floor for streaming the
  pi rows in+out per iteration is **≈ 76 ms** — that's the hard wall for this approach.

---

*Cross-references: [[kernel-bench-optimization-state]], [[kernel-bench-backward-race-context]].*