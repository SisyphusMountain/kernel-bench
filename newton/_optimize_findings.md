# Likelihood-optimization recipe — findings (666x80, 2026-06-14)

`newton/optimize.py` is a dataset-agnostic optimizer: a pluggable `torch.optim` first-order stage
(any optimizer + any LR schedule, via assigning the hand-written Triton gradient to `theta.grad`)
followed by an optional exact-fp32 Newton polish. Below: what the comparison harness + endgame
characterization showed on the representative `666x80` fixture (80 CCPs, S=1331), and the recipe
that follows.

## First-order schedule sweep (200 steps, no polish, lr0=1.0)

All Adam/Adagrad arms land in the **same loss basin** (NLL 137564–137691, spread ~0.09%); the real
differentiators are **‖g‖** and **wall time**.

| arm | final NLL | final ‖g‖ | wall | steps |
|---|---|---|---|---|
| adam/constant | 137564 | 113 | 72 s | 200 |
| adam/adaptive | 137691 | **29.9** | **29 s** | 80 |
| adam/plateau | 137567 | 31.7 | 47 s | 133 |
| adam/cosine | 137648 | 46.9 | 27 s | 74 |
| adagrad/cosine | 137736 | 28 | 26 s | 73 |
| rmsprop/* | 142k–146k | 500+ | — | diverges |
| lbfgs (scipy, fp64) | 137675 | 26 | **1342 s** | 204 |

- **Constant lr=1 leaves ‖g‖ high (~113)** — the "bounce" (loss creeps down, gradient doesn't).
- **Decaying schedules reach ‖g‖~30 and stop ~2× sooner.** `adaptive` (loss-reactive bold-driver:
  grow while improving, shrink on a loss increase or gradient-direction reversal) gave the best
  ‖g‖/time. `plateau`/`cosine` are close.
- **rmsprop diverges at lr=1** — lr=1 is Adam/Adagrad-specific, not universal.
- **L-BFGS (scipy, fp64) is non-competitive** — 22 min (fp64 + CPU/scipy overhead + no warm-start
  across line-search evals). Dropped from the recipe.

## Endgame characterization (from an Adam/adaptive endpoint, ‖g‖≈30–38)

| endgame | final NLL | final ‖g‖ (unreg) | total wall | behavior |
|---|---|---|---|---|
| Adam-only (constant, 500 steps) | **137540** | 182 | 167 s | bounces along the valley; lowest loss, NOT stationary |
| Newton, **lam=0** (internal damping) | 137669 | 7.8 | 639 s | ‖g‖ 32.9→1.34→**bounces back to 9**; CG hits max_iter (ill-conditioned) |
| Newton, **ridge** (auto_lambda λ=13.7) | 137688 | **5.11** | **141 s** | ‖gF‖ 37.8→4.6→1.1→0.57 **monotone**; CG converges 8–11 iters; stalls at solver floor |

### The key finding: the optimum is a flat/indefinite valley, and it decouples low-loss from low-‖g‖
- The **lowest-loss** point (Adam, 137540) has the **highest ‖g‖** (182, bouncing). The
  **lowest-‖g‖** point (ridge Newton, 5.1) sits at **slightly higher loss** (137688, +0.1%).
- **lam=0 Newton is the worst of both**: as its internal damping decays toward the floor, undamped
  steps wander the flat directions — loss falls, ‖g‖ **rises** — and CG stops converging
  (max_iter=40 → ~40 HVPs/step → 610 s). Do not use lam=0 here.
- **Ridge fixes it**: the MAP term `F = L + λ/2‖θ−θ_ref‖²` (λ from a short exact-HVP Lanczos)
  conditions the system so CG converges in ~10 iters (4.5× faster) and ‖gF‖ descends monotonically
  to the solver-precision floor (~0.57). It does NOT reach the `small`-fixture's ‖g‖≈0.02 — on
  666x80 the flat valley + truncated solvers floor it higher. λ=13.7 also pins θ near θ_ref, so it
  finds the nearest stationary point (higher loss than the valley floor Adam drifts to); a smaller
  σ (→ smaller λ) would slide further down at the risk of re-introducing the bounce — tunable.

## Recommended recipe (defaults in `optimize.py`)

1. **First-order (basin entry):** Adam, `lr0=1.0`, **adaptive** schedule. Stops on relative-grad /
   loss-flat / lr-floor (~80 steps, ~30 s on 666x80, ‖g‖→~30). Dataset-agnostic (all-relative).
2. **Polish (clean stationary point):** **ridge** Newton (`ridge=True`, exact-fp32 HVP, auto_lambda).
   Monotone, CG-cheap, ‖g‖→solver floor (~141 s on 666x80). `--no-ridge` (lam=0) is NOT recommended.

**Choose by goal:**
- **Pure MLE (lowest NLL):** Adam alone is competitive and 4× cheaper — use `--no-polish`, optionally
  `--schedule constant` to grind the loss down the flat valley (167 s → 137540).
- **Stationary point / Fisher-Laplace uncertainty:** the ridge polish is required (the Hessian/Fisher
  is only meaningful at small ‖g‖). 141 s gets ‖g‖→5 cleanly.

## Caveats / open knobs
- lr=1 is Adam/Adagrad-specific; rmsprop/sgd need a smaller lr0.
- Ridge σ (default 0.01 → λ≈13.7) trades endgame stability vs how far it descends the valley.
- The ‖g‖ floor (~5 unreg) is set by the flat spectrum + truncated forward/adjoint solvers, not the
  optimizer — it won't go to ~0 here regardless of method.
- Per-step wall in the Newton history is stamped at return (newton_lanczos doesn't expose per-step
  timing); the final/total wall is accurate.

## Ridge-annealing (λ-continuation) polish — `ridge_anneal()` (2026-06-14)

`ridge_anneal()` starts λ at the auto_lambda rule (-min(λ_min,0)+σ·λ_max via exact-HVP Lanczos) and
anneals it down with an adaptive bold-driver (×0.3 clean/cheap, ×0.7 near the edge, stop on no
progress), running safe Newton steps (`cg_witness` δ self-correction + Armijo) at each level with a
moving (proximal) θ_ref. Memory discipline (only one point's forward intermediates + HVP closure
live; drop the per-step saved dicts — `_sv`/`st` — and free at the 8 GiB threshold) is load-bearing:
without it the backward's driver-free scratch gate trips at the 2nd inner step.

Shallow (σ=0.01, inner=3, max_levels=8): **137685 / ‖g‖2.49 / 156 s** — best stationarity of all
polishes (½ the single-λ ridge floor) at ~the same loss/wall. λ went 13.4→4.0 then hit the floor.
Deep (floor 1e-5, 14 levels): λ 13.6→0.024 slides loss 137687→**137660** (toward the Adam floor) but
‖g‖ **bounces back up** (2.5→22) and CG saturates max_iter from λ≈1.2 down (999 s). So **λ is a dial
trading loss for stationarity** — the flat-valley decoupling, made continuous.

## ⚠ Precision/truncation audit (2026-06-14) — the above absolute numbers are unreliable

The fixture captures `pi_iters=16`, `neumann_terms=16`. Both are **too low to converge the solve**,
and the bias is **larger than the inter-method differences above**:
- Forward Pi residual @pi=16 ≈ **0.97** (vs ~1e-12 @256). Loss is biased **+33 NLL high**:
  Adam 137689→137656, ridge_anneal 137685→137650, once pi_iters≥64 (converged by 64; 16/32 not).
- Backward ‖g‖ converges by **neumann_terms≈32–64** (gmres confirms). nt=16 underestimates the
  ridge_anneal gradient **2.6×** (4.79 vs true 12.57). The "‖g‖≈2.5 floor" is an artifact.
- **dtype is irrelevant**: fp32 ≈ fp64 to ~0.03 NLL / ~0.01 ‖g‖. The error is truncation, not
  rounding — fp64 alone does NOT fix it.
- On the TRUE (pi=128,nt=64) objective, ridge_anneal still beats Adam on both loss (137650<137656)
  and ‖g‖ (12.57<34.25) — the method ranking holds; the absolute numbers do not.

**Action**: re-run optimization + characterization with `pi_iters≥64`, `neumann_terms≥32` (64 safe),
and `NEWTON_TANGENT_SELF_ITERS≥64` (the HVP tangent loop is truncated too). See
`/tmp/claude-1000/{precision_test,neumann_sweep}.py`.

### fp32 convergence study (2026-06-14) — iterations, not precision

Per the consumer-GPU constraint (weak fp64), all of the following is **fp32**; converging the *solve*
is what matters. Set `solver_options.pi_iters` (propagates to forward + HVP tangent fallback) and
`neumann_terms`; `NEWTON_TANGENT_SELF_ITERS` overrides the tangent loop.

**Direction is corrupted at N=16, not just magnitude.** At the ra checkpoint (λ=13.5 fixed), as the
solver iteration count N (=pi=neumann=tangent) increases toward the N=128 reference:

| N | loss | ‖g‖ | cos(g, g₁₂₈) | cos(p_newton, p₁₂₈) |
|---|---|---|---|---|
| 16 | 137685 | 2.64 | **+0.215** (~78° off) | +0.909 (~25° off) |
| 32 | 137651 | 12.36 | +0.99997 | +0.99999 |
| 64 | 137650 | 12.57 | +1.000 | +1.000 |

So N=16 points the gradient ~78° away from the truth; the damped Newton step is less distorted
(λI regularizes it) but still ~25° off. **Converged by N=32; N=64 = N=128 (safe margin).**

**Optimizing against the converged objective reaches a genuinely cleaner stationary point**, and
makes the optimizer's own in-loop ‖g‖ trustworthy. fp32 Adam(adaptive)→ridge_anneal, judged on the
TRUE objective (pi=128, neumann=64):

| run | true loss | true ‖g‖ |
|---|---|---|
| truncated-opt (pi=neumann=16) Adam | 137656 | 34.25 |
| truncated-opt ridge_anneal | 137650 | 12.57 (in-loop claimed 2.49 — a lie) |
| **converged-opt (N=64) ridge_anneal** | 137653 | **2.83** (in-loop 2.8 — true) |

Converged-opt ridge_anneal gets true ‖g‖ to **2.83** (vs 12.57) at ~the same loss — 4.4× cleaner —
because the optimizer is no longer fooled by a truncated objective. Cost ~2× wall (Adam 30→63 s,
ridge_anneal 156→269 s), not 4×. Recipe: for any run that needs a real stationary point (e.g.
Fisher-Laplace), set N=64 (or 32) across the stack; it stays fp32. See
`/tmp/claude-1000/{converged_reopt,newton_dir_sensitivity}.py`.

### N=64 annealing from the best checkpoint — the bounce is CONDITIONING, not geometry (2026-06-14)

Started from the best (lowest-true-loss) checkpoint (137650 / ‖g‖12.57), deep-annealed λ at N=64
(fp32) with the new `ridge_anneal(spectrum_m=...)` per-level **bare-Hessian spectrum** diagnostic.

| phase | λ range | true loss | true ‖g‖ | bare-H λ_min | CG | verdict |
|---|---|---|---|---|---|---|
| shallow (floor 0.42) | 14→1.3 | 137650→**137646.9** | 12.57→**1.46** | +0.20 | 10→40 | monotone, NO bounce |
| deep (floor 1e-5, cg≤80) | 1.0→0.21 | →137640 | 1.34→**10.2** | +0.21 | hits max_iter ≤0.3 | loss↓ but ‖g‖ BOUNCES |
| decisive (damped-N, cg≤300) | 0.3 fixed | 137647→137642 | 1.46→4.33 | +0.20 | **300, still not converged** | CG can't solve it |

The spectrum is the smoking gun: **λ_min(H) ≈ +0.20 stays positive everywhere** — the Hessian is PD,
a genuine minimum exists, no witness ever fires. The bounce is purely **ill-conditioning**:
λ_max≈1500, so κ = λ_max/(λ_min+λ) ≈ 1500/0.2 ≈ **7500** as λ→0, and the spectrum is **clustered at
the bottom edge** (the `lanczos_extremes` docstring's warning) — CG's worst case. Plain CG hits
max_iter (even 300) at λ≤0.3, so the under-solved step reduces *loss* along easy directions but
leaves the high-curvature directions, and ‖g‖ rises.

**Answers to "can gradient AND likelihood converge at N=64 via annealing":**
- **Likelihood: yes** — monotone down to true ≈137640 (below Adam's true floor 137656 and the start).
- **Gradient: down to ~1.3** (8.6× from 12.57) at the conditioning sweet spot λ≈1 where CG converges
  (~53 iters); it will NOT go lower with plain CG — capped by conditioning, not by geometry (PD) or
  truncation (N=64 is converged).
- **To drive ‖g‖→0 needs a PRECONDITIONER** (or deflation / a Krylov method robust to bottom
  clustering) for the inner solve. `ridge_anneal` now records `lamH_min/max`,
  `ref_dist`, `gp`, `dF`, `step_norm` per step (gated behind `spectrum_m>0`).
  See `/tmp/claude-1000/{anneal_n64_diag,anneal_n64_deep,damped_newton_test}.py`.

### fp64 is NOT the lever — the endgame is conditioning-bound, not precision-bound (2026-06-15, A100)

Tested the endgame in **fp64 on an A100** (real fp64 hw) to check whether the fp32 HVP's rounding was
capping CG. It is not. At a near-optimal N=64 checkpoint (true loss 137650.7, ‖g‖ 12.4):

| | fp32 | fp64 |
|---|---|---|
| Q1 HVP rel-error vs fp64 | **~1.1e-4** | (ref) |
| Q2 CG resid @50 iters, λ=0.1 | 0.51 | 0.58 |
| Q3 one Newton step → true ‖g‖ | 12.4→**15.68** | 12.4→**15.72** |

fp64 gives the **identical** CG trajectory and Newton step at **2.5× the wall**. CG stalls at ~0.5
residual because κ≈5000 (λ=0.1) with a bottom-clustered spectrum — *conditioning*, not precision
(fp32's 1e-4 noise floor is nowhere near 0.5). **Precision is irrelevant; fp32 is correct and cheap;
the endgame is conditioning-bound.** Corollary: this work belongs on the **local GPU (fp32)**, not the
cluster. (`/tmp/claude-1000/fp64_focused.py`.)

### The open question that gates the next step: is the flat valley low-rank or high-dimensional?

The bottleneck is the ill-conditioned, bottom-clustered, **PD** Hessian (λ_min~0.2, λ_max~1500). The
right move depends on the SHAPE of the bottom spectrum, which we have NOT measured (only the extremes
λ_min/λ_max). Two regimes, opposite prescriptions:
- **Low-rank flat** (a few tiny eigenvalues = a few non-identifiable parameter combinations) →
  deflated CG (deflate the bottom-k eigenvectors) or a reparametrization breaks the floor; ‖g‖→0
  becomes reachable.
- **High-dim flat** (many tiny eigenvalues) → ‖g‖→0 is **ill-posed** (the minimizer is a flat
  region, not a point); deflation can't help; the right framing is the loss-minimizer (Adam) + a
  principled prior (the ridge/MAP we already use), with the flat directions BEING the posterior
  uncertainty (Laplace).

**Recommended next step: measure the Hessian's bottom eigenvalue spectrum near the optimum** (smallest
~30–50 eigenvalues / spectral density via Lanczos; fp32, local, cheap). That single diagnostic chooses
between "engineer a deflated/preconditioned CG" and "reframe — ‖g‖→0 is the wrong target." It also
reveals whether the ill-conditioning is **intrinsic non-identifiability** (a modeling fact) vs a
solver problem — and whether the downstream goal is even served by chasing ‖g‖→0 (MLE point estimate
is already solved by Adam; only a Laplace/uncertainty goal needs a clean stationary point).
