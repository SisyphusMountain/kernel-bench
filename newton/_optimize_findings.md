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
