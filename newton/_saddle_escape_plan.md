# Saddle escape (negative-curvature descent) plan

## Context — what we are actually fixing

On `666x80` (N=64-converged, fp32; dtype verified irrelevant) the "best" checkpoint is **not a
minimum**. Accurate `m=120` Lanczos gives `lam_min = -0.058` (negative), with a second Ritz value at
`+0.0018` and the rest positive: an **index-1 saddle** with a low-rank bottom. Confirmed across
fp64 seed0 (-0.0582), fp64 seed1 (-0.0515), fp32 (-0.0579) — same structure (1 negative, 1 near-zero,
rest positive). See `newton/_optimize_findings.md`.

This explains the whole `||g||->0` struggle: the ridge `lam*I` masked the `-0.058` eigenvalue while
`lam > 0.058`; annealing `lam` below it exposed the negative curvature -> CG broke down -> `||g||`
bounced. We were converging *at a saddle*. The coarse `m=20` Lanczos (`lam_min` Ritz biased HIGH,
"can miss the sign" — `newton/cg.py:16`) hid it as `+0.2`.

The fix is **not** a preconditioner and **not** fp64. It is **negative-curvature descent**: step along
the `lam_min` eigenvector to leave the saddle, re-optimize, and re-check the spectrum. Iterate until
`lam_min >= 0` (a real minimum) or we prove the saddle is intrinsic (non-identifiability).

Goal of this plan: reach a checkpoint that is **PD (lam_min >= 0 up to Lanczos bias) AND small,
stable `||g||`** — the convergence certificate the user asked for — or a defensible conclusion that no
such point exists below the current loss.

## Step 0 — do NOT escape a phantom (decisive pre-checks, cheap)

A `-0.058` eigenvalue is `~4e-5` of `lam_max~1476` — *below* the fp32 HVP relative-noise floor
(`~1e-4`). fp64 already agrees, but two artifacts could still fake a negative Ritz value, and both are
cheap to rule out before investing in the escape machinery:

- **0a. HVP symmetry.** The exact HVP is forward-over-reverse; truncation could make the operator
  slightly non-symmetric, and Lanczos on a non-symmetric operator can manufacture spurious tiny
  negative Ritz values. Check `|v^T H w - w^T H v| / (||Hv|| ||w||)` over ~5 random pairs. If the
  asymmetry is of order `|lam_min|`, the "saddle" is an operator artifact, not geometry -> STOP and
  reframe. (Expect it to be `<< |lam_min|`, since N=64 converges the solve — but this is the gate.)

- **0b. Kernel-independent curvature along `v_min`.** Confirm the negative curvature with the TRUE
  Hessian, bypassing the HVP kernels entirely: central finite difference of the gradient,
  `c(eps) = (g(theta + eps*v) - g(theta - eps*v)) . v / (2 eps)`, with `v = v_min` (unit), `eps`
  swept (e.g. 1e-3, 3e-4, 1e-4) for a stable plateau. `g` comes from `make_value_and_grad`
  (`grad_avg_K>=4` to beat the backward's atomic noise). If `c(eps) < 0` and matches `lam_min`, the
  saddle is real beyond any HVP-kernel concern -> PROCEED. If `c(eps) >= 0`, the HVP Ritz value was an
  artifact -> STOP and reframe.

Only if 0a and 0b both confirm do we run the escape. This is the highest-value, lowest-cost part of
the plan: it either certifies the saddle is real or saves us from chasing a numerical ghost.

## Step 1 — the negative-curvature direction `v_min`

`lanczos_extremes` (`newton/cg.py:13`) already runs the exact loop with full reorthogonalization and
stores the basis `Q`, but returns only eigenvalues. Add a sibling that also returns the bottom Ritz
**vector**, sharing the tridiagonalization (refactor a private `_lanczos_tridiag(Av, p, m, seed) ->
(Q, alphas, betas)` and build both `lanczos_extremes` and the new `lanczos_min_eigpair` on it — DRY,
existing callers untouched):

```
lanczos_min_eigpair(Av, p, *, m=120, seed=0) -> (lam_min, v_min)   # v_min unit, fp64
    Q, al, be = _lanczos_tridiag(Av, p, m, seed)
    w, S = eigh_tridiagonal(al, be, eigvals_only=False)   # full eigvecs of T
    s = S[:, 0]                                            # eigvec for smallest Ritz value
    v = sum(s[i] * Q[i])  ;  v /= ||v||
    return float(w[0]), v
```

`m=120` is required here (the bottom is low-rank but `m=20` missed the sign). Sanity-check the Ritz
vector's residual `||H v - lam_min v||` is small (well-converged) before trusting it.

## Step 2 — the escape step (`negative_curvature_step`)

`v_min` is a descent direction to second order regardless of sign (`v^T H v = lam_min < 0`). Pick the
sign so the (tiny) linear term does not fight: `d = -sign(g . v_min) * v_min` (at a near-stationary
saddle `g . v` is tiny, so curvature dominates and the line search settles it either way).

The quadratic model along `d` has no minimum (inverted parabola), so only the **true forward loss**
sets the scale. Use an expand-then-pick-best line search on the deterministic `forward_solve` loss
(`newton/vg.py:52`), NOT an Armijo-vs-gradient test (there is no first-order descent to anchor to):

```
t = t0                                   # t0 ~ small multiple of the parameter scale / sqrt|lam_min|
sample L(theta + t*d) for t in {t0, 2 t0, 4 t0, ...}  while loss keeps dropping   (expand, cap ~8)
        and for t in {t0/2, t0/4}                       if t0 overshoots            (backtrack)
theta_esc = argmin over sampled points of the true forward loss
require L(theta_esc) < L(theta) - c * (1/2) t^2 |lam_min|   # genuine 2nd-order decrease, else report "stuck"
```

Warm-start each trial solve with the previous `E` (the `warm_E=` path) to keep it cheap. ~8-12 forward
solves total.

## Step 3 — re-optimize from `theta_esc`

Resume the existing polish from the escaped point — it is already the right tool and needs no change:
`ridge_anneal(static, theta_esc, col_weights, spectrum_m=120, ...)` (`newton/optimize.py:202`). It
auto-picks `lam0` from a fresh spectrum (now reflecting the escaped geometry), runs witness-safe
lambda-continuation Newton, and returns `(theta, history, lam0)`. The `spectrum_m=120` diagnostic logs
per-level bare-H `lam_min` so we watch the negative eigenvalue lift off zero as we descend.

## Step 4 — re-check the spectrum, and loop

At the re-optimized `theta`, run `lanczos_min_eigpair(Av, p, m=120)` again:

- **`lam_min >= -tol` (tol ~ Lanczos bias, e.g. 1e-3) AND `||g||` small/stable** -> SUCCESS: PD
  minimum + near-zero gradient. This is the certificate. Stop.
- **`lam_min` still `< -tol`** -> a (new or residual) saddle. Repeat Steps 1-3 from here. Bound the
  outer loop to `K` rounds (e.g. 4).
- **Escape yields no loss decrease, or we keep returning to a saddle of the same depth** -> the
  negative direction is an **intrinsic non-identifiability**, not an escapable bump. CONCLUDE: the MLE
  region is a flat ridge / saddle (a modeling fact), and the right framing is the Adam loss-floor + a
  principled prior (the ridge/MAP we already have), with the flat/negative directions being posterior
  structure (Laplace). Report this explicitly rather than chasing `||g||->0`.

Wrap Steps 1-4 in `escape_saddle(static, theta, col_weights, *, rounds=4, lanczos_m=120, ...)` in
`newton/optimize.py`, returning `(theta, history)`. Expose via `optimize(..., polish_mode="escape")`
(append to the `{"ridge","ridge_anneal","lanczos","none"}` set; default unchanged) + the `--polish-mode`
CLI. Default `ridge_anneal` -> `escape_saddle` chaining is natural (escape, then re-polish) but keep
them separately selectable.

## Reuse (do not reimplement)

- `lanczos_extremes` / new `lanczos_min_eigpair` share `_lanczos_tridiag` — `newton/cg.py:13`.
- `cg_witness` + the `delta <- nu*(delta - cert)` bump — already inside `ridge_anneal`
  (`newton/optimize.py:312`); Step 3 calls `ridge_anneal` wholesale, no new solver.
- `make_exact_hvp` (HVP), `make_value_and_grad` / `forward_solve` (gradient, FD probe, line-search
  loss, warm-start) — `newton/hvp_exact.py`, `newton/vg.py:61` / `newton/vg.py:52`.
- Memory discipline (the saddle-escape loop holds one HVP closure at a time): null `sv`/`_sv`/`st`
  after use and `free_cuda_cache_if_tight(8.0)` before each gradient eval — same pattern as
  `ridge_anneal` (`newton/optimize.py:249-266, 286-289, 338`), or the backward's driver-free scratch
  gate trips.

## Verification protocol (666x80, local, fp32, N=64)

Set `pi_iters=64`, `neumann_terms>=32`, `NEWTON_TANGENT_SELF_ITERS=64` (the converged regime — see
`kernel-bench-truncation-convergence`). Then:

1. **Step 0 first** — report HVP asymmetry and the FD-along-`v_min` curvature `c(eps)` plateau. Gate
   the rest on both confirming the saddle.
2. From the saved `anneal_n64_checkpoint.pt` saddle, run `escape_saddle`. Headline artifact: the
   per-round table **`round -> escape dL -> ||g|| -> lam_min(m=120) -> # negative Ritz`** — does the
   negative eigenvalue lift to `>= 0` while loss decreases?
3. **Success criteria**: terminal `lam_min >= -1e-3` (PD up to Lanczos bias), `||g||` small and stable
   (no late bounce), terminal loss `<=` the saddle's loss. If unmet, the Step-4 non-identifiability
   conclusion is the deliverable instead (equally valid — it answers the user's convergence question).
4. Sanity: terminal loss matches an independent `forward_solve`; the FD curvature at the terminal
   point is `>= 0` (kernel-independent confirmation of PD), not just the Lanczos value.
5. Cross-check one round in fp64 on the A100 (the escape direction + terminal `lam_min`) to confirm
   the trajectory is not fp32-specific — but expect agreement (dtype already shown irrelevant). Only
   if convenient; not required.
6. Save JSON + append findings to `newton/_optimize_findings.md`; update the
   `kernel-bench-truncation-convergence` memory with the outcome (escaped-to-minimum vs intrinsic
   saddle). Commit.

## Out of scope

- HVP kernel re-tuning (`_hvp_profiling_report.md`).
- A preconditioned/deflated inner CG — explicitly ruled out as the next step: the problem is an
  indefinite saddle, not a PD ill-conditioned solve. (Deflation may return *later*, for the low-rank
  flat bottom, only after the negative direction is resolved.)
- Multi-fixture sweep — validate on `666x80` first; the method is fixture-agnostic by construction.
```
