# MAP / penalized-likelihood with cross-validated prior — plan (Sanderson 2002)

Next phase after the raw-NLL basin investigation (`_specieswise_basin_findings.md`). That work showed
the raw MLE of `theta[S,3]` is **non-identifiable and partly on the 0/1 boundary** (~half the per-state
rates run to θ→−∞ on sparse events; ~3 NLL run-to-run variance lives in those unidentifiable
directions). So the raw MLE is not a well-posed estimator. The fix — statistically, not just
numerically — is a **prior whose strength is chosen by cross-validation**, exactly Sanderson's
penalized-likelihood program (`sanderson.pdf`).

## Sanderson 2002 in one paragraph (the template)

A **saturated** model (every branch its own rate) is over-parameterized and **non-identifiable** — the
direct analog of our per-state DTL rates. Sanderson maximizes a **penalized** likelihood
`Ψ(θ) = log L(θ) − λ·Φ(θ)`, where Φ is a roughness penalty and λ trades fit against smoothness:
λ=0 → saturated/overfit, λ→∞ → fully pooled ("clock"). λ is chosen by **cross-validation**: prune a
terminal observation, refit at λ on the rest, **predict** the held-out count from the fitted model,
and pick λ minimizing the average prediction error `CV(λ) = Σ_m (x_m − x*_m)² / x*_m` (eq. 5). The CV
curve is U-shaped with an interior minimum (his Fig. 1) — the data-driven optimal smoothing.

## Mapping to our model

| Sanderson | here |
|---|---|
| per-branch substitution rate `r_k` | per-state DTL log-rate row `theta[s,:]` |
| saturated model (non-identifiable) | full `theta[S,3]` (we proved non-ID, [[kernel-bench-landscape-nonidentifiability]]) |
| log-likelihood `log L` | `−NLL(theta)` (the reconciliation likelihood, full-depth solver) |
| roughness penalty `Φ` | prior `R(theta)` (Gaussian shrinkage now; phylogenetic roughness later) |
| smoothing `λ` | prior precision `λ` |
| prune terminal branch, predict count | hold out a data unit, predict its held-out NLL |

## The penalized objective

`F_λ(θ) = NLL(θ) + λ·R(θ)`. Two penalty choices, in order of effort:

1. **Gaussian shrinkage (start here — already implemented):** `R = ½‖θ − θ_ref‖²` ⇒ `H_MAP = H + λI`,
   the certifiable PD form we already use (`ridge_anneal`, `specieswise_fit.py`). `θ_ref` = a sensible
   pooled rate (per-column mean of a moderate-rate fit, or a single global DTL rate). This directly
   pulls the unidentifiable boundary rates (θ≈−20) back to `θ_ref` instead of letting them wander.
   λ=0 → raw MLE (boundary, our 137384); λ→∞ → all rates = θ_ref (the "clock" analog).
2. **Phylogenetic roughness (closer to Sanderson, later):** penalize squared rate differences between
   adjacent species-tree states `Σ_(s,s') (θ_s − θ_s')²`. Shrinks toward *local* smoothness rather than
   a global mean. Needs the species-tree adjacency from the capture. Defer until (1) is working.

## Cross-validation design — the one real open question

Sanderson holds out **terminal-branch observations**. Our held-out unit depends on the data layout,
which must be confirmed first (FIRST STEP below). Candidate fold structures:

- **(A) k-fold over gene families** — if the fixture has `G>1` families: train on a subset, evaluate
  predictive NLL on held-out families. Cleanest, most standard. (NB: notes say fixtures may have `G=1`
  E-rows — must verify; if G=1 this fold is unavailable and we need B or C.)
- **(B) k-fold over CCPs / observations** — hold out a subset of the conditional-clade observations
  the reconciliation scores; predict their NLL under the fitted rates.
- **(C) leave-out species-tree states (Sanderson-literal)** — drop the data attributable to a state,
  refit, predict that state's contribution from its neighbors. Only meaningful with the **roughness**
  penalty (2) that couples neighbors; pairs naturally with penalty choice (2).

CV criterion: **held-out predictive NLL** (the reconciliation NLL of the held-out unit under the
train-fold MAP fit), averaged over folds → `CV(λ)`. Pick `λ* = argmin_λ CV(λ)`.

## Algorithm

```
0. CONFIRM data layout: how many gene families G, what is the natural held-out unit, how to evaluate
   the solver's NLL on a held-out subset. (Inspect the capture: data/666x80/whole.pt 'inputs'/'static'.)
1. Choose penalty R and theta_ref (start: Gaussian shrinkage, theta_ref = pooled moderate-rate fit).
2. Build K CV folds over the chosen unit.
3. For lambda in a log-spaced grid (e.g. 1e-3 .. 1e2, ~10-12 points):
     for each fold k:
        fit theta_hat(lambda,k) = argmin F_lambda on the TRAIN data  (Adam -> L-BFGS -> ridge_anneal
                                  on F_lambda; reuse specieswise_fit machinery with the prior on)
        record held-out predictive NLL on fold k
     CV(lambda) = mean_k held-out NLL
4. lambda* = argmin CV(lambda); refit on ALL data at lambda* -> the CV-optimal MAP estimate.
5. Certify: H_MAP = H + lambda* I  PD via residual-gated Lanczos (already have lanczos_min_eigpair) or
   a dense reduced Hessian at the final theta (p=3993 is tractable as a one-off, per the reviewer).
6. Report: the CV curve (NLL vs log lambda, Sanderson Fig 1 analog), lambda*, the final estimate +
   PD certificate, and the implied rate variation / how many rates the prior pulled off the boundary
   (theta_diagnostics before vs after).
```

## What we already have (reuse, don't rebuild)

- MAP fit at fixed λ: `ridge_anneal` / `specieswise_fit.py` minimize `NLL + (λ/2)‖θ−θ_ref‖²`.
- Basin-robust optimizer: Adam (best-snapshot) → L-BFGS (`maxcor`), with the init-scale lesson
  (start rate ~0.2–0.25). Under a prior the boundary runaway is bounded, so basins should be far more
  **reproducible** — a thing to confirm early (does the ~3 NLL variance collapse with λ>0?).
- PD certificate machinery (`lanczos_min_eigpair`, residual-gated) + the dense-Hessian fallback.
- `theta_diagnostics` to show the prior pulling rates off the boundary.

## First steps (next session)

1. **Inspect the capture data layout** (`data/666x80/whole.pt`) to fix the CV fold unit — this gates
   everything. Determine G (gene families) and how the solver's NLL decomposes over held-out data.
2. **Sanity experiment:** re-run `basin_search` under a small fixed prior (λ≈0.05–0.1) and check
   whether the basins become reproducible (variance collapses) and the boundary saturation drops
   (`theta_diagnostics`). If yes, that alone validates the pivot before building full CV.
3. Implement the λ-grid CV loop (`newton/map_cv.py`) once the fold unit is fixed.
