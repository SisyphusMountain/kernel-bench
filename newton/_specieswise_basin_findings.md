# Specieswise basin investigation — findings (2026-06-15)

Investigation of the `666x80` raw-NLL landscape for the specieswise rate tensor `theta[S,3]`
(per-state DTL log-rates; `S=1331`, `p=3993`). Triggered by an engineer-review plan recommending
basin-finding (Adam → L-BFGS) over Newton, gauge-fixing, subspace deflation, and a careful
"lock the objective" audit. **Headline: the raw MLE is the wrong target — it is intrinsically
non-identifiable and partly on the 0/1 boundary. Next phase is MAP + cross-validation
(`_map_cv_plan.md`).** Extends [[kernel-bench-landscape-nonidentifiability]].

## What moved the loss: initialization scale, not solver tricks

The Adam→exact-Newton pipeline stalled at a saddle (137640); prior work reached ~137466 via
L-BFGS-from-Adam. This session found the **initialization rate** is the real lever:

| init DTL rate | Adam.best | → L-BFGS (raw NLL) |
|---|---|---|
| 1e-5 | 138115 | 137897 (steep trap basin, λ_min=−0.23) |
| 0.01 | 137641 | — |
| 0.05 | 137538 | 137518 |
| **fixture ~0.077** | 137479 | **137470.6** |
| 0.15 | 137474 | 137461 |
| **0.22 – 0.25** | 137400 | **≈137384** (floor, flat) |

Raw-NLL ladder **137470 → 137461 → 137384** purely by raising the init rate. The floor is flat at
**≈137384** across rates 0.22–0.25 (reproduce: `python -m newton.basin_search`). Things that did
**NOT** help (ruled out with evidence):
- **L-BFGS memory** `maxcor` 10/50/100 → identical fixed point. Not the lever.
- **L-BFGS restarts / maxiter 3000** from a basin floor → 0.0 further descent. The 137466 floor was a
  true L-BFGS fixed point, not an under-resourced one.
- **Smaller inits** (1e-5…1e-3) are strictly *worse* (monotone) and the 1e-5 run lands in a different,
  steep trap basin (137897, a clean λ_min=−0.23) — confirms real basin multiplicity, init-dependent.

## The 137384 basin is REAL, not a truncation artifact (the "lock the objective" audit)

`python -m newton.convergence_audit` at the deep basin:
- **Forward/Pi:** loss is bit-identical from **pi=64** through pi=1024 and across e_tol∈{1e-6,1e-8,1e-10}.
  We ran pi=128 — ~64 iterations of margin. The 80-NLL drop is not a forward-solve under-shoot.
- **Backward/Neumann:** the analytic gradient at **neumann=64** matches neumann=256 to 4 decimals
  (cos=1.000000, rel 7e-4); a hi-fidelity re-opt (pi=256/neumann=128) descends **0.0** further. The
  objective was correctly locked; L-BFGS minimized the *true* objective, not a truncated one.
- Caveat that validates the old warning: **neumann=16 is catastrophic** (cos(g,g₂₅₆)=0.15 — gradient
  81° off true). The captured fixture's pi=16/neumann=16 defaults were genuinely broken; we were safely
  above that the whole time.

## The gauge audit: the parameterization is already gauge-fixed

`python -m newton.gauge_audit`. The reviewer flagged a possible softmax gauge (per-row 1-vector null
direction → up to S=1331 artificial flat modes). It does **not** exist here: `extract_parameters`
softmaxes over **4 categories with the reference logit pinned to 0**, so the per-row all-ones
direction has `‖Hu‖=6.7`, curvature `+1.86` (not 0), and a unit row-shift moves NLL by +0.69. No
1331-dim null space. ⇒ The residual non-identifiability we see is **genuine statistical**
non-identifiability, not a parameterization artifact, and **no gauge-fixing / row-centering is needed.**

## Why the raw MLE is the wrong target: boundary saturation + non-identifiability

`python -m newton.theta_diagnostics`. In **every** optimized solution (old 137466 and deep 137384
alike — nearly identical profiles), about **half the rates run to the 0/1 boundary**:

| | fixture | old 137466 | deep 137384 |
|---|---|---|---|
| \|θ\|>5 | 0% | 50.5% | 50.9% |
| DTL prob < 1e-3 | 0% | 47.0% | 46.9% |
| rows with a prob > 0.99 | 0% | 8.9% | 8.8% |

This is the MLE behaving exactly as it must with **sparse per-state events**: a state with no observed
event of a type drives that rate to 0 (θ→−∞), and the likelihood is **flat** there (the data can't
tell a rate of 1e-4 from 1e-6). Consequences:
- Those flat, boundary-running rates **are** the near-zero Hessian eigenvalues / non-identifiable
  directions. The "minimum" is a **plateau at the boundary** in ~half its coordinates.
- It explains the **~3 NLL run-to-run variance**: runs park the unidentifiable rates at different
  extreme values with negligible loss change. So "deeper basin" is *partly* chasing how far the
  unidentifiable rates wandered — the deep-basin advantage is real fit, but ~half its parameters are
  statistically meaningless point estimates.
- ⇒ The raw MLE is not a well-posed estimator here. A small prior is statistically (not just
  numerically) the right move.

## The two deep basins are separated by a real barrier (mode geometry)

Why does a rate-0.15 init reach ~137461 but a rate-0.25 init reach ~137384 (a 77-NLL gap)? Compared the
two checkpoints (`newton/basin_compare.py`, `newton/basin_interp.py`):

- **What changed in parameter space — diffuse and coordinated, favoring duplication.** ~61% of
  species-states moved their logits by >1 (factor ≥2 in odds); only 2.9% (39 rows) flipped which event
  dominates, and **19 of those 39 flipped to pD-dominant** (pD/duplication is the most-shifted
  probability). The changed rows are not specifically the saturated ones (corr 0.14). It is a
  system-wide re-balancing toward more duplication, not a localized fix.
- **The 77 NLL is non-additive — no subset of rows carries it.** Copying the most-changed rows
  137461→137384 (rest left at 137461) makes the fit *worse* until ~most rows move: K=10 → −23%, K=100 →
  −26%, K=250 → +1%, K=800 → +63%, all 1331 → 100%. Reverting only the 39 flips is catastrophic
  (−167%). Because the reconciliation likelihood couples every state through the tree, a few B-rows
  among A-rows is an internally inconsistent rate field that fits worse than either pure basin.
  ⇒ 137384 is a different **globally self-consistent configuration**, not a patched 137461.
- **A clear barrier separates them on the straight line** `theta(a)=(1-a)θ_461+aθ_384`
  (`newton/_figures/interp_461_384.png`): NLL climbs from 137461 to a peak **137546 at a=0.65**
  (**+84.7 above 461, +161.8 above 384**), then descends to 137384; both endpoints are wells (NLL rises
  for a<0 and a>1). So a local optimizer in 137461 sees only uphill toward 137384 — it would have to
  climb ~85 NLL — which is exactly why L-BFGS-from-461 stalled and only a higher init rate (landing on
  the far side of the ridge) reaches 137384.
- **The theta-barrier is largely a log-odds PARAMETERIZATION ARTIFACT, and the basins are the SAME mode.**
  Three paths between the two checkpoints (`newton/basin_interp.py`, `newton/basin_connectivity.py`,
  figures in `newton/_figures/`):

  | path A(137461) -> B(137384) | barrier above A |
  |---|---|
  | straight line in **theta** (log-odds) | **+84.7** |
  | straight line in **probability** (simplex, `INTERP_SPACE=prob`) | **+10.0** |
  | optimized curved (Bezier) path | **+1.6** |

  Interpolating the actual event probabilities linearly (instead of log-odds) cuts the barrier from
  +85 to +10, and an optimized curved path removes it (+1.6). theta is a log-odds coordinate that
  compresses the boundary (where ~half the rates live), so a straight theta-line crosses an artificial
  ridge; in probability space the path is nearly downhill. ⇒ **137461 and 137384 are the same mode** —
  a continuous duplication-favoring deformation — and local L-BFGS in theta-space stalled only because
  the log-odds straight-line directions carry that artificial ridge. (All path NLLs are deterministic;
  see the variance note below — the +1.6 residual is real, not eval noise.)

  > Reframes the MAP+CV question: not "which discrete basin wins" but "how far along this connected
  > duplication-favoring ridge does cross-validation support going."

## Determinism vs run-to-run variance (`/tmp/.../variance_check.py`, `repeats_variance.py`)

The **objective and gradient are deterministic**: the forward loss at a fixed theta is bit-identical
over 8 repeats (spread 0.000000), and the gradient is reproducible to ~2e-4 relative (cos=1.000000)
even at grad_avg_K=1 — the backward's atomic non-determinism is negligible. So **all the path NLLs and
basin losses above are exact**, not noisy reads.

The "run-to-run variance" is a different thing: the **optimizer endpoint** varies, not the loss
evaluation. Measured over **n=5** identical Adam->L-BFGS runs from the same rate-0.15 init
(`repeats_variance.py`): final spread **7.3 NLL** (137457.4–137464.7); the divergence already enters in
Adam (adam.best spread 4.4 NLL) and L-BFGS amplifies it. With a deterministic objective this is purely
the trajectory diverging on the flat/connected near-optimum region (consistent with the mode being
connected). Do NOT conflate with eval noise: a 1-2 NLL barrier on a fixed path is real curvature, but a
few-NLL difference between two optimizer endpoints is just where each run stopped on the flat ridge.

**Consequence:** ~7 NLL endpoint variance means **differences below ~7 NLL between "basins" are not
meaningful** — e.g. the old 137466 region and the "137461" rate-0.15 region are the SAME region within
the scatter (and the committed `basin_137461.pt` was a mediocre draw; runs reach as low as 137457). The
only robust distinction is the **~77 NLL gap down to the connected 137384 region** (rate >=0.25). This
is the degeneracy/non-strict-convexity in action and is exactly what a small MAP prior should remove
(curvature lambda on the flat directions -> unique local min -> variance collapses): a testable
prediction for the MAP+CV phase.

## Subspace deflation works, but deflation is not a basin-finder

`a100_subspace_deflate.py` (A100, fp64, job 4634822). Block/subspace saddle-free deflation **fixed the
bounce** of single-vector deflation — 5 monotone rounds 137466.42 → 137466.33 — but:
- It moved the loss only **~0.1 NLL**. Deflation is a curvature/certification tool, not a basin-finder.
- The spectrum bottom stays at the **numerical near-degeneracy resolution wall**: the most-negative
  restricted Ritz value is −6.7e-3 with **residual 3.1e-2** (unresolved even at fp64 m=200). "PSD to
  tolerance," not a hard certificate. ⇒ **bare-H MLE is not cleanly certifiable; MAP is the clean,
  certifiable route** (confirmed across single-vector & subspace, fp32 & fp64). By Sylvester, no SPD
  preconditioner can fix the inertia either.

## Reproduce

```
python -m newton.basin_search          # init-scale sweep -> raw-NLL floor ~137384 (+ run-to-run variance)
python -m newton.convergence_audit     # pi/neumann convergence + hi-fidelity reopt at the deep basin
python -m newton.gauge_audit           # softmax gauge test (already gauge-fixed)
python -m newton.theta_diagnostics     # boundary/saturation profile (pure CPU)
python -m newton.basin_compare         # 137461 vs 137384: per-row prob diff + NLL row-attribution
python -m newton.basin_interp          # linear interpolation -> the +84.7 NLL barrier (saves a plot)
python -m newton.basin_connectivity    # mode-connectivity: is there a curved low path? (saves a plot)
```
Checkpoints: `newton/_checkpoints/specieswise_best_137384.pt` (deep basin, rate 0.25),
`newton/_checkpoints/basin_137461.pt` (rate 0.15), `newton/_checkpoints/old_basin_137466.pt`
(reference). Figures in `newton/_figures/`. Objective locked at pi=128/neumann=64/tangent=64.

## Conclusion → next phase

The raw MLE is non-identifiable and partly meaningless (boundary rates). The basin hunt has hit
diminishing, partly-illusory returns. **Pivot to MAP / penalized likelihood with a CV-chosen prior**
(Sanderson 2002, `sanderson.pdf`) — see `newton/_map_cv_plan.md`. The prior both (a) regularizes the
unidentifiable boundary rates to sensible values and (b) yields the certifiable PD minimum bare-H
cannot.
