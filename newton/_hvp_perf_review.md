# Exact-HVP performance review — directions to investigate (first pass, 2026-06-12)

Context: analytic exact-Hessian HVP (forward-over-reverse) is correct (gates
`python -m newton.verify {e_so,wave_so,dts_so,hvp} small` all green) but exact-fp64 is
*slower* than fd-fp64 (1762 ms vs 1369 ms per HVP on small) when on paper one analytic
HVP ≈ 1 fwd + 1 bwd vs 2 full gradients for FD. exact-fp32 is 597 ms (2.3× vs fd-fp64).
This file lists where the theoretical margin is likely leaking, from a code read of
`newton/hvp_exact.py`, `newton/forward_tangent.py`, `newton/ggn.py`, and the three SO
kernels. **Hypotheses, not measurements** — confirm with torch.profiler/nsys before
changing anything; re-run all four gates and re-benchmark after each change.

## PROFILE RESULTS (2026-06-13, small fp32, one steady-state HVP, co-tenant present)

Op-level profile (`/tmp/claude-1000/profile_fp32_cuda.txt`, self CUDA total 537 ms).
Absolute time is co-tenancy-inflated (steady HVP ~1.58 s here vs ~597 ms isolated), but
the breakdown is robust. **The diagnosis is unambiguous: the cost is the harness, not the
new kernels.**

| Op | Self CUDA | Calls (1 HVP) | What it is |
|---|---|---|---|
| `_wave_step_tangent_kernel` | 138.6 ms (25.8%) | **12,193** | tangent self-loop Jacobi (#1) |
| `aten::max` (+reduce_kernel) | 115 ms (21.4%) | **24,173** | per-iter convergence check `diff/scale` (#1) |
| `aten::index_add_` (+indexFunc) | 66 ms (12.3%) | 6,486 | dts SO tree host walk (#6) |
| `aten::abs` | 31.9 ms (5.9%) | 49,494 | convergence check + masks |
| `_local_scalar_dense` + Memcpy DtoH | 44 ms (8.2%) | **27,556 DtoH** | host syncs (#1, .any() in #6) |
| `_wave_so_kernel` | 5.5 ms (1.0%) | 142 | the new SO contraction |
| `_dts_split_so_kernel` | 6.5 ms (1.2%) | 141 | the new dts SO |
| `_dts_cross_backward_accum_kernel` | 5.5 ms (1.0%) | 141 | frozen reused solve |

**Confirmed:** suspect #1 dominates. ~142 waves × ~86 Jacobi iters each = 12,193 tangent
launches, **two host-syncing max-reductions per iteration** (24,173 max + 27,556 DtoH
copies — the CPU is 32% stuck in `_local_scalar_dense`, fully serializing CPU↔GPU). The
primal it differentiates uses **16 fixed Neumann terms, zero syncs**. Switching the tangent
self-loop to a fixed ~16–20-iter count should remove the 24k max + most of the 27k DtoH
syncs and cut the dominant kernel's launches ~5× — plausibly >50% of total time, and more
faithful to the truncated gradient.

**Confirmed:** suspect #6 is real and #2 (the dts tree host walk + `.any()` sync) — 12.3%.

**Overturned:** the new second-order Triton kernels (`_wave_so`/`_dts_split_so`/
`e_step_so`) are **~3% combined** — NOT worth optimizing. Suspect #7 (kernel-level rewrites)
is deprioritized. Fix the harness (#1, #6), not the kernels.

Order to implement: **#1 first** (biggest, also a semantics improvement), then **#6**
(Triton-ize or de-sync the dts tree walk), then #2/#3/#4 cleanups. Re-profile after #1 —
it will reshuffle the table.

## FIX #1 LANDED (2026-06-13) — fixed-iter tangent self-loop

`jvp_root_scores(..., self_iters=N)` now runs a fixed N-step sync-free Jacobi (vs adaptive
converge-to-`self_tol` with 2 host-syncing max-reductions per iter). `make_exact_hvp`
resolves N = arg → `NEWTON_TANGENT_SELF_ITERS` env → `solver_options.pi_iters` (16 here,
matching the primal forward truncation). Adaptive path preserved for `self_iters=None`
(verification gates). Files: `forward_tangent.py`, `hvp_exact.py`.

Result (small fp32, **isolated, co-tenant gone**):
- steady HVP **597 ms → 446 ms** wall (1.34×); self-CUDA **537 ms → ~218 ms** (2.46× less GPU work).
- `_wave_step_tangent_kernel`: 12,193 calls / 138 ms → **2,272 calls / 22.6 ms** (= 16×142).
- `aten::max` (24,173 calls / 115 ms) and the bulk of the 27,556 DtoH syncs: **gone**.
- Accuracy unchanged: hvp gate PASS, rel 8.3e-5 / 1.15e-4, symmetry 9.4e-4 (residual was
  always dominated by the backward neumann/bicgstab truncation, not the forward count).
  `forward` tangent gate (adaptive path) still 3.8e-8.

**New top of profile (`profile_fp32_cuda_AFTER_fix1.txt`):** `aten::index_add_` 30.4%
(66 ms, 6,486 calls) + `indexFuncLargeIndex` 29.6% — i.e. **suspect #6, the dts SO tree
host-side walk, is now #1.** Wall (446 ms) >> self-CUDA (218 ms) ⇒ now partly CPU/launch
bound. Next: #6 (Triton-ize / de-sync the dts tree walk), then #2/#3 (per-wave
`free_cuda_cache_if_tight`, hoist u-independent work) to close the wall-vs-GPU gap.

## FIX #6 LANDED (2026-06-13) — dts SO tree walk Triton-ized

Replaced the host-side parent-chain `index_add_` loop (per wave: 2 sides × `max_ancestor_depth`
levels × 2 launches + a `.any()` host sync per level) with one fused `_dts_tree_so_kernel`
(`dts_so.py`) mirroring the production compact level-walk: bottom-up subtree-or-self
accumulation on BOTH `ud→sub` and `dud→dsub` in place, then scatters
`dp'(A−sub) + p'(dA−dsub)` into d_rhs + d_grad_col. Staging changed to stacked `[2N,S]`
(contiguous views, no split-kernel change). New required `compact_level_*` args threaded
from `state_helpers` through `hvp_exact.py` and the `verify.py` gate.

Result (small fp32, isolated):
- steady HVP **446 ms → 140 ms** (3.2×); self-CUDA **~218 ms → 81 ms**.
- `index_add_`/`indexFuncLargeIndex` (66 ms, 6,486 calls) → **gone**; new `_dts_tree_so_kernel`
  9.9 ms / 141 calls. The host walk was also serializing the CPU (per-level syncs + thousands
  of launches), so killing it helped wall more than its GPU share suggested.
- Accuracy unchanged: dts_so gate ~3e-9 (incl. d_rhs/d_grad_col), hvp gate bit-identical
  (8.3e-5 / 1.15e-4, symmetry 9.4e-4).

**Cumulative: 597 ms → 140 ms (4.3×) on small fp32; ~9.8× vs the fd-fp64 baseline (1369 ms).**

New profile (healthy, no host hog): `_wave_step_tangent_kernel` 29% (23.7 ms, 2,272 = 16×142
tangent-sweep launches) is now top; `_dts_tree_so` 12%, `_dts_split_so` 8%, `aten::sum` 7.7%
(the A/dA totals I added — fuseable into the split kernel), `_wave_so` 7.5%, frozen kernels
~6–7% each. Wall 140 ms vs self-CUDA 81 ms ⇒ ~60 ms CPU/launch gap remains.

Remaining opportunities (smaller now): #3 hoist u-independent head/E-side work (CPU gap),
#2 per-wave `free_cuda_cache_if_tight` → once/sweep, fuse `aten::sum` A/dA into `_dts_split_so`,
#4 merge the 3 `e_step_backward_so` calls. The new SO kernels themselves remain ~3–20% and are
not obvious wins to micro-optimize further.

## FIX #3 LANDED (2026-06-13) — hoist u-independent head/E-side work

`e_bwd_params`, the primal cotangents (`cot_*`/`base_p`), and the smooth head forward graph +
first-order `g1` are now built ONCE in `make_exact_hvp` setup (theta fixed across CG), with the
head graph retained (`create_graph` + per-call `retain_graph=True`). Each `hvp(u)` now adds only
`phi2` + one backward. Numerically bit-identical (hvp gate 8.3e-5 / 1.15e-4 / 9.4e-4).

Result (small fp32, isolated): steady HVP **140 ms → 138 ms** — marginal here (the saved
work is one e_step backward + head double-pass, small vs the wave sweep), but it removes
genuinely redundant per-CG-iteration work that compounds in a full Newton solve (10–40 CG
iters/point). Kept as a correct cleanup.

## PHASE A GLUE-OP FUSION LANDED (2026-06-13)

A1: `_dts_tree_so_kernel` now computes the row totals `A`/`dA` internally (pre-pass sum before
the in-place level walk), dropping the host `ud.sum(1)`/`dud.sum(1)` (282 launches/HVP) + 2
buffers. A2: `_wave_so_kernel` folds the wave's own `d_rhs[ws:ws+W]` into `d_Av` (gated by
`FOLD_RHS`), so `d_Av` IS the frozen-solve seed — removes the host `seed = d_rhs + d_Av` add
(142/HVP) + the seed buffer. Both bit-identical (dts_so/wave_so ~3e-9; hvp 8.3e-5/1.15e-4/9.4e-4)
and memory-better (good for 1007x64).

Result (small fp32, isolated): steady HVP **138 → 136 ms** (~1.5%); `aten::sum` 1292→1005,
`aten::add` 1224→1060 (~450 fewer launches, ~10%); self-CUDA 80→77 ms. **Confirms the linear
nature of glue-op fusion**: each removed op is individually cheap (~1–5 µs GPU, ~1.8 µs CPU
dispatch), so a 10% launch cut buys ~1.5% wall.

**Phase B (scatter-accum fusion) NOT pursued.** It would remove ~1,000 launches (the 7 scatters
+ 6 `aw=c+l` adds/wave), but by the same linear logic that is ~3–4% wall at most, for a much
riskier change (one fused 7-target kernel handling both G=1 reductions and G>1 `index_add`).
Not worth it. The measurement makes the verdict clear: **the only lever that meaningfully moves
a launch-bound HVP is CUDA-graph capture** (eliminates per-launch CPU cost for all ~4,300 at
once), not incremental fusion. Recommend stopping glue-op work here.

## DIMINISHING RETURNS — now launch-overhead-bound

Profile after #1/#6/#3: wall 138 ms vs **self-CUDA 80 ms** (was 537 ms originally). The
~58 ms gap is CPU kernel-launch dispatch + GPU idle between launches: **4,340 `cuLaunchKernelEx`
per HVP** (~7.9 ms CPU dispatch) across ~142 waves × ~15–20 kernels each. `_wave_step_tangent`
(28.7%, 2,272 launches) is the largest single GPU item but is irreducible at pi_iters=16.

Cumulative: **597 → 138 ms (4.3×) on small fp32; ~9.9× vs fd-fp64 (1369 ms).** Further wall
gains require attacking launch count, NOT kernel bodies:
- **CUDA-graph capture** of the per-wave sweep (replay the launch sequence with one CPU call) —
  the right next lever, but a substantial project (static shapes/pointers, capture/replay
  plumbing, interaction with bicgstab's data-dependent iteration count).
- Kernel fusion across the per-wave sequence (wave_so + the following frozen solve seed; the
  several wave_backward_uniform sub-kernels).
Both are higher-effort/higher-risk; recommend stopping at 4.3× unless the big-fixture wall
(1007x64) justifies the CUDA-graph investment. Smaller leftover: fuse the `aten::sum` A/dA
reduction (6 ms GPU) into `_dts_split_so` — but we are CPU/launch-bound, so it won't move wall.

## Performance suspects (ranked by expected payoff)

1. **Tangent self-loop solve host-syncs every Jacobi iteration**
   (`forward_tangent.py::jvp_root_scores`, the `float((cur - prev).abs().max())`
   convergence check + per-iteration `.clone()`). Runs per wave, up to
   `self_max_iter=200`, to tol 1e-12 (fp64) / 1e-6 (fp32) — while the primal it
   differentiates uses **16 fixed Neumann terms with zero syncs**. Fix: fixed
   production-matched iteration count, no convergence check. Expected: the documented
   2–4× headroom on the tangent sweep, plus removal of all per-iteration syncs.
   Bonus: improves semantics — H becomes the true Jacobian of the *truncated* gradient
   we actually optimize (currently part of the ~1e-4 FD residual and ~9e-4 asymmetry,
   which CG/Lanczos implicitly assume away).

2. **`free_cuda_cache_if_tight()` once per wave** in the HVP reverse sweep
   (`hvp_exact.py`, top of the wave loop). Each call is a driver `mem_get_info`
   round-trip; when it fires, `empty_cache()` drops the allocator pool so subsequent
   waves re-allocate from the driver (slow), defeating the caching allocator. Fix: call
   once per sweep (before the E-side transition where the big frees happen), or only
   ahead of known-large allocations.

3. **u-independent work recomputed inside every `hvp(u)` call** — should be hoisted to
   `make_exact_hvp` setup (theta is fixed across all CG iterations):
   - `base_p = e_bwd_params(wE, acc["grad_Ebar"])` and the `cot_pS/pD/pL/mc/col`
     assembly (primal head cotangents).
   - The head graph for `phi1`/`g1`: `g1` is u-independent. Build the
     `extract_parameters_weighted_cols` forward + `g1 = grad(phi1, create_graph=True)`
     once; per call only `grad((g1*u).sum() + phi2, theta_req, retain_graph=True)`.
   Saves one head forward + one full create-graph backward + one e-step autograd pass
   per HVP.

4. **Three `e_step_backward_so` calls can be one.** The SO contraction outputs are
   linear in the cotangents `(g_new, g_ebar)`, so `so_p = so_w + so_aux` exactly, and
   `rhs_E` needs `so_aux[0] + so_w[0]` = combined`[0]`. One call with
   `(g_new=wE, g_ebar=acc["grad_Ebar"])` replaces all three. Small absolute win
   (G×S kernels) but free and simpler.

5. **`compute_dts_tangent` runs twice per split wave per HVP** — once in the forward
   tangent sweep, again in the reverse-sweep recompute (`keep_d_dts=False` was a
   1007x64 memory fix). On small the `d_dts` buffers fit. Fix: make `keep_d_dts`
   conditional on available memory / fixture size.

6. **The dts SO tree part is host-side Python** (`dts_so.py::dts_backward_so`, tree
   section): per wave per HVP, a Python loop over `max_ancestor_depth` × 2 sides of
   `index_add_` launches, with a `bool(valid.any())` device sync per level. Fix: port
   to a Triton kernel mirroring `uniform_cross_pibar_vjp_tree_from_ud_fused`'s
   compact-level walk (`compact_level_*` structures already in `state_helpers`);
   minimal fix: drop the `.any()` early-break.

7. **Kernel-level (only after profiling shows the kernels themselves matter):**
   `wave_so` is deliberately classic single-tile (BLOCK_S = next-pow2(S), 8 warps,
   two ancestor walks + atomic path scatter into `sub`/`dsub` scratch). Candidates:
   register-resident patterns from commit a1920aa; fusing the SO contraction into the
   seed computation of the immediately following frozen solve (same wave state read
   back-to-back); buffer reuse for the 6 `d_aw*` + `sub`/`dsub` allocations per wave
   and `ud_l/ud_r/dud_l/dud_r` per split wave (interacts with #2: `empty_cache`
   defeats allocator reuse).

## Correctness flags (none blocking)

- **Latent col-gradient inconsistency** (`dts_so.py`, tree part):
  `d_grad_col += contrib_t.sum(0)` accumulates unconditionally, even with
  `use_col_weights=False` where `p_prime` has no col dependence (true derivative 0).
  Invisible today because uniform col weights make `cot_col`'s head contribution
  vanish; will silently corrupt H if col weights are activated. Audit the whole SO path
  against what the primal kernels do with `grad_col_log_probs` under
  `use_col_weights=False`, and re-gate with col weights on.
- **All FD gates ran on a G=1 fixture.** Unexercised by any oracle: `_scatter_accum`'s
  G>1 branches, `e_step_so`'s per-g-row parallelism, the known-latent e_step atomic
  race surface. Direction: run the e_so/wave_so/dts_so gates on a (possibly synthetic)
  G>1 fixture.
- **`fam_factor` heuristic** (`hvp_exact.py`): `1.0 if G == n_fam else n_fam` mirrors
  the primal's two-case branch; a future G≠n_fam, G>1 fixture breaks both silently.
- **`norm.clamp_min(tiny)` tangent ignored** in the closed-form norm-term tangent —
  degenerate-case only (norm→0).
- **Truncation asymmetry** (~9e-4) mildly violates the symmetry CG/Lanczos assume;
  item #1 is the principled mitigation.

## Phase-2 protocol

1. Wait for the convergence experiment to free the GPU (shared card; don't contaminate
   its timings).
2. Profile ONE `hvp(u)` (torch.profiler or nsys) split into: tangent forward sweep /
   per-wave SO contraction / frozen tangent-adjoint solves / dts tree part / E-side /
   head. Count host syncs (nsys cudaStreamSynchronize / cudaMemcpyDtoH rows).
3. Implement the confirmed top items (expect 1–4 first), re-run
   `python -m newton.verify {e_so,wave_so,dts_so,hvp} small`, re-benchmark per-HVP ms
   (small, then 1007x64 fp32).
