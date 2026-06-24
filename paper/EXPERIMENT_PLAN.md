# Experiment plan — `gpurec` paper

Maps each `\expbox` hole in `gpurec_paper.tex` to: what it must show, what is already in hand,
what must be run, the output artifact (figure/table), dependencies, risk, and effort. Narrative
spine (per `suggestions.md`): **differentiable reconciliation is the enabling paradigm; every
experiment is a consequence of it** — speed, certified optima, uncertainty, priors, transfer
heterogeneity.

## Status at a glance

| # | §   | Experiment                      | Readiness | Blocker / next action |
|---|-----|---------------------------------|-----------|-----------------------|
| 1 | 5.1 | Speed vs AleRax (genewise)      | �a Half   | gpurec timings exist; **run AleRax** (binary is local) + fairness harness |
| 2 | 5.2 | Certified local optimality      | 🟢 In hand| **A100 job 4637220 finishing now** = the certified specieswise min; make figure; fix §5.2 TODO |
| 3 | 5.3 | Fisher info / identifiability   | 🟢 In hand| compute I_obs spectrum + D–L eigenvectors + SEs from a certified θ (p=357 → exact eigh OK) |
| 4 | 5.4 | Receiver weights                | 🔴 Code   | **wire `_g_recv` into grad + HVP** (currently discarded), then fit ±w |
| 5 | 5.5 | CV tree-smoothing on Hogenom    | 🟢 Mostly | CV reg-helps already confirmed; package curve + cost; maybe one clean grid re-run |
| 6 | 5.6 | Ablation (held-out NLL)         | 🟡 Partial| baseline/+smoothing in hand; +weights rows need #4 |

Order: **#2 → #3 → #5** (the conceptual core, nearly done) ∥ **#4 (code) → #6** ∥ **#1 (AleRax)**.

---

## #2 — Certified local optimality (§5.2)  [IN HAND — finishing now]

**Claim.** Second-order info *verifies* local convergence and detects spurious (saddle) stops that
first-order tools cannot. Strict-*local* only (wording already fixed).

**In hand (this session):**
- 256-fam fp64 specieswise λ=0.03 bounded fit: **CERTIFIED** — |Pg|=1.1e-4, reduced-Hessian
  PD λ_min_free=+0.062, KKT signs clean, 0 escapes.
- **A100 job 4637220 (running)** = the full-archaea (5446) fp64 certified bound-constrained min:
  the headline specieswise certificate. |Pg| already ≤3e-3, descending to gtol; cert pending.
- Saddle contrast: cold scratch fit is a *resolved* saddle (λ_min≈−0.21, m≥160); the
  negative-curvature direction is a true-loss descent (escape reached 357037 < warm); cold↔warm
  downhill-connected (zero activation energy). All reproducible.

**⚠ Fix the §5.2 TODO before writing.** The current TODO cites "warm 357039 = PD min (λ_min=+0.017)".
**That is falsified** — fp64 gradcheck shows 357039 is non-stationary (|g|=5.25), the *unconstrained*
λ=0.03 problem is ill-conditioned (κ≈38k, boundary runaway). The clean certified minimum is the
**bound-constrained** one (job 4637220) — or the CV-optimal λ≈1 (already PD, λ_min=+0.22). Rewrite the
narrative as: *cold fit halts at a saddle → the neg-curvature direction gives a descent first-order
can't take → box/penalty + free-subspace Newton reaches a certified PD minimum.*

**To run:** (a) finalize job 4637220 numbers; (b) Fig: loss profile along the bottom eigenvector at the
cold saddle (shows the first-order-invisible descent) and/or a |Pg|, λ_min table (cold saddle vs
certified min). Effort: ~½ day (job already running).

## #3 — Fisher information / identifiability (§5.3)  [IN HAND — strongest figure]

**Claim.** Observed information gives SEs/CIs and exposes confounded directions — impossible with
first-order tools. (Observed-vs-expected wording already fixed.)

**In hand:** the D–L confounding is established and *quantified* — at the certified 256-fam bounded min,
Duplication+Loss are pinned at the rate floor (126/167 active), Transfer free and identified (median
0.33). Hessian-structure work: confounded pair = Dup–Loss per species (turnover D+L soft, net D−L
stiff), Transfer decoupled.

**To run (one focused script, p=357 so the exact 357×357 Hessian is cheap — ~357 HVPs once):**
1. **Fig 3a** — eigenvalue spectrum of I_obs = H(θ̂) at a specieswise optimum (log scale, near-zero
   tail marked). Use a fit where the spectrum is meaningful: the λ=0.03 specieswise H (stark near-zero
   tail = the identifiability story) for the spectrum; a regularized fit (λ≈1, PD) for invertibility.
2. **Fig 3b** — bottom eigenvectors as per-species (δ,τ,λ) loadings; show the D−L anti-correlation
   (turnover well-determined, net not). This is the biological payoff.
3. **(a) SEs/CIs** — from I_obs⁻¹ at a PD optimum (λ≈1 or the free-subspace block of the bounded min);
   report SEs for a few per-species (δ,τ,λ).

Effort: ~1 day. Risk: low (all computable from a saved θ). Decision: which optimum (recommend: show the
λ=0.03 spectrum for the confounding + λ=1 for the SEs, noting the regularization is what restores
identifiability — directly motivates #5).

## #5 — Cross-validated tree-smoothing prior on Hogenom (§5.5)  [HEADLINE — mostly done]

**Claim.** A principled, CV'd model-selection workflow for DTL rates, *enabled by GPU throughput*,
impractical on CPU — the "only-possible-because-differentiable-and-fast" result.

**In hand:** GBM tree-Laplacian prior + λ-homotopy CV on full Hogenom-1055 already run
(`experiments/sanderson_cv/`); **CV regularization helps** (held-out NLL 1427→841), κ*≈1, boundary
saturation 0.64→0.05, exact-HVP PD cert. fp32 solver (fp64 27× slower on the 4090).

**To run:** (a) locate/clean the hogenom CV artifacts; produce **Fig 2** = held-out NLL vs κ (CV curve)
with κ* marked; (b) the boundary-saturation reduction numbers; (c) compute cost (#κ × K fits, wall-clock)
+ a CPU-time estimate for the same protocol. Possibly one clean re-run over a tidy κ grid / K folds for a
publication-quality curve. Effort: ~1 day. Risk: low.

## #4 — Receiver weights (§5.4)  [CODE TASK first]

**Claim.** Learned per-recipient weights infer transfer sinks (a feature AleRax lists as future work),
at no derivation cost.

**Blocker (confirmed):** `make_value_and_grad` computes the receiver-weight VJP but **discards it**
(`value_and_grad.py:137`, `_g_recv`); the exact HVP is θ-only. Required code:
1. Optimize α jointly: stop discarding `_g_recv`; add α (per-species recipient logits, softmax,
   Σw=1) to the parameter vector; project/regularize per §2.4 (tree-Laplacian on α or Dirichlet on w).
2. Extend the matrix-free HVP / Fisher block to the (θ,α) block (autograd target → receiver logits).
3. **Verify** the new grad + HVP against finite differences (the project's standard fp64 FD recipe).

**Then run:** fit ±w on a dataset with transfers (archaea is transfer-rich); (a) inferred w_s identify
sinks, relate to biology; (b) held-out NLL ±w (controlling for added params). Effort: ~2–3 days
(code + FD verify + experiment). Risk: medium (backward surgery). This unblocks #6's weight rows.

## #1 — Speed vs AleRax (§5.1)  [gpurec ready; run the baseline]

**Claim.** Order-of-magnitude faster genewise (per-family) MLE at equal/better NLL.

**In hand:** gpurec genewise timings exist (`_artifacts/genewise_adaptive/*.json`: hogenom 512/1055/full,
archaea; e.g. all-5446 archaea ~23–39s; hogenom-1055 ~430s warm-rebatch). **AleRax binary is local**
(`/usr/local/bin/alerax`).

**To run:**
1. AleRax genewise (per-family) on the *same* CCP inputs (archaea `ale_gene_tree_distributions`;
   Hogenom), CPU with N MPI ranks, matched stopping (final NLL within tol).
2. Fairness harness: identical inputs, common convergence criterion, **report hardware explicitly**
   (GPU model vs CPU+#ranks), final NLL parity check.
3. **Table** (dataset, #families, tool, hardware, wall-clock, final NLL) + **Fig 1** scaling
   (time vs #families).

Effort: ~2–3 days (AleRax runs at several family counts + harness). Risk: medium — the reviewer-scrutinized
fairness controls (same CCPs, same NLL tolerance, honest hardware accounting) are the whole ballgame.

## #6 — Ablation (§5.6)  [after #4]

Held-out NLL (CV, Hogenom specieswise) for: baseline / +smoothing / +weights / +both. Baseline &
+smoothing come from #5; +weights rows need #4. Effort: ~1 day after #4.

---

## Open decisions
- **Datasets:** pin "Hogenom-Core" — 666 (Morel2024) vs the 1055 we have? Fix exact family counts +
  any subsetting (paper §4 TODO). Use the same set across #1/#3/#5/#6 for coherence.
- **Which optimum for §5.3:** recommend λ=0.03 spectrum (confounding) + λ=1 SEs (identifiability
  restored). Ties #3 → #5.
- **§5.2 endpoint:** the bound-constrained λ=0.03 certified min (job 4637220) as the headline cert, with
  the λ=1 PD min as the clean alternative. Update the §5.2 TODO (357039 story is falsified).
- **Receiver-weight regularization:** tree-Laplacian on α vs symmetric Dirichlet on w — pick per #4.
