# The hunt for the minimum — a war story

*A write-up of why finding the optimum of a tree-reconciliation likelihood turned out to be
genuinely hard, what was actually going on, and the recipe that finally worked.*

---

## The setup

We're fitting a phylogenetic **tree-reconciliation** model. The free parameters are a matrix
`theta[S, 3]` — for each of `S` species-tree states, three **DTL log-rates** (Duplication /
Transfer / Loss, stored as base-2 logits; a softmax turns them into event probabilities). We
minimize the negative log-likelihood (NLL) of the observed gene trees.

Two things make this not a textbook optimization:

1. **The objective is the output of an iterative solver, not a closed form.** Evaluating the NLL
   runs a forward fixed-point iteration (the "Pi" solve) and the gradient/Hessian run a backward
   self-loop (a Neumann series). So even the *exact* gradient and Hessian are only as exact as the
   solver's truncation depth. Get the depth wrong and you're optimizing a subtly different function
   than you think.

2. **It's big and the curvature is interesting.** Our representative fixture is `666x80`:
   `S = 1331`, so `p = 3993` parameters. We compute **exact Hessian-vector products** via
   forward-over-reverse autodiff (Triton kernels), which lets us do Newton-CG and Lanczos
   eigenvalue estimates — that machinery is what eventually let us *see* what was going wrong.

The goal sounds simple: drive the gradient to zero and confirm we're at a minimum. It took a long
time to realize why that kept failing.

---

## Struggle 1 — we were optimizing the wrong function

The captured fixture shipped with shallow solver settings (forward iters = 16, backward Neumann
terms = 16). Those **don't converge**: the loss was biased by ~33 NLL and the gradient was off by
up to 2.6×. Early "minima" were artifacts of an inconsistent objective — the truncation error moved
the apparent stationary point.

**Fix:** crank the solver to a converged regime (forward = 128, backward = 64, tangent self-loop =
64) and verify the loss and gradient are stable to many digits as you push the depth further. Only
then is the geometry you're measuring real. Everything below is at that converged objective.

> Lesson: when your objective comes out of an inner solver, *the truncation is part of the model*.
> Validate convergence of the objective before you trust any stationary point.

---

## Struggle 2 — the gradient would not go to zero

We built an exact-Hessian **Newton-CG** polish. It kept misbehaving:

- With no regularization, `‖g‖` would **bounce** instead of decreasing, and CG hit its iteration cap.
- Adding a ridge term (a MAP / Tikhonov `λ‖θ−θ_ref‖²`) conditioned the system so CG converged fast
  and `‖g‖` fell monotonically — **but the ridge biases the solution**, so it stalled ~0.1% high.
- We built **ridge-annealing** (λ-continuation: start with a big stabilizing λ, shrink it
  warm-started so the target slides toward the true optimum) to remove that bias.

The annealing helped the conditioning but the endgame *still* wouldn't stationarize at low loss.
Something was structurally wrong, not just ill-conditioned.

---

## The diagnosis — the "best" point was never a minimum

The point the Adam→Newton pipeline kept converging to (NLL ≈ **137640**) is **not a minimum. It's a
saddle.** Its Hessian has a small **negative** eigenvalue (`λ_min ≈ −0.05 to −0.06`).

This explains the whole Struggle-2 mess at once:

- The ridge had been **masking** the negative eigenvalue (`H + λI` is positive if `λ > |λ_min|`).
- **Annealing λ down past `|λ_min|` re-exposed the negative curvature**, which is exactly when CG
  broke (CG assumes positive-definiteness; it has no answer for a descent direction of negative
  curvature).
- All the `‖g‖→0` effort was trying to converge *at a saddle*. You can drive the gradient small
  there, but it's the wrong critical point.

We confirmed the saddle is **real**, not a numerical ghost: the negative curvature reproduces under
a finite-difference probe of the gradient (kernel-independent), the HVP operator is symmetric to
~1e-7, and the eigenvalue is stable as you vary the solver depth. It's geometry, not truncation.

---

## Why it was so sneaky — the bad direction is orthogonal to the gradient

Here's the part that makes this a good story. The negative-curvature direction `v_min` is
**orthogonal to the gradient** at every low-loss point we examined. Three consequences, each of
which had been quietly sabotaging us:

1. **Our negative-curvature detector was blind to it.** A standard trick is to watch the Krylov
   subspace CG builds from the gradient for a direction of negative curvature. But `v_min ⊥ g`
   means it lives *outside* `Krylov(g)` — CG ran 120 iterations and never saw it.

2. **Adam dodged the saddle and never told us.** First-order methods avoid saddles generically.
   Adam sat in the low-loss valley oscillating with a **large** gradient (`‖g‖ ≈ 200`), never
   stationarizing — so it never felt the negative direction and never converged to it.

3. **Newton ran *straight at* it.** Newton's method homes to the *nearest critical point*,
   minimum or not. So the more "Newton" our polish became, the more reliably it snapped onto the
   saddle. We had built a precision instrument for finding exactly the wrong point.

This produces a counterintuitive **decoupling** of "low loss" from "small gradient":

> **low loss ⟺ huge `‖g‖`** (you're on the steep wall of a ravine)
> **small `‖g‖` ⟺ higher loss** (you're at the saddle)

Chasing `‖g‖→0` was actively walking us *away* from the good region.

---

## Where the real minimum is, and how to reach it

The true low-loss basin sits at NLL ≈ **137466** — about **173 NLL below** the saddle the pipeline
kept stalling at. It is reached **only** by a **quasi-Newton method with a line search (L-BFGS),
launched from Adam's deep valley floor.**

Why L-BFGS and not Newton? Because L-BFGS **descends loss without trying to stationarize.** It
slides *along* the ravine, past the saddle, down to the basin — precisely because it isn't seeking
a critical point. Newton can't do this here: from anywhere near the saddle it just snaps back onto
it.

So the practical recipe inverted what we started with:

> **Adam (find the low-loss valley) → L-BFGS (descend the ravine to the basin)** —
> *not* Adam → exact-Newton polish.

Newton still has a job, but a different one (see the certificate section).

---

## A methodology trap worth flagging — Lanczos lies if you under-resolve it

We estimate `λ_min` with **Lanczos** on the exact HVP. Critical subtlety we got bitten by: the
smallest **Ritz value** (Lanczos's estimate) is an **upper bound** on the true `λ_min`, and it
converges *downward* as you add iterations `m`. So a coarse `m` **over-reads the sign**:

| Lanczos iterations `m` | reported `λ_min` at the *same* point |
|---|---|
| 20  | **+0.35**  (looks positive-definite!) |
| 200 | **−0.034** (actually indefinite) |

I briefly declared a point "positive-definite" off an under-resolved `m=120` read. It wasn't —
`m=200/300` in fp64 showed it was negative. **A near-zero smallest Ritz value is not a PD
certificate unless the Ritz residual is also small.** We now residual-gate every spectrum read and
use `m ≥ 200` (fp64 on an A100 for the high-`m` runs).

> Lesson: an iterative eigensolver gives you a *bound*, not the eigenvalue. Always carry the
> residual, and never certify a sign from a bound that's still moving.

---

## Is the minimum even positive-definite? The non-identifiability twist

At the *basin floor* (not the saddle), a well-resolved fp64 `m=300` Lanczos gives the bottom of the
spectrum as:

```
−0.0362,  −0.00478,  +0.00044
```

Two distinct things are going on here, and conflating them was a mistake worth naming:

- The **negative** eigenvalues are genuine saddle directions — real residual non-convexity.
- The **+0.00044 ≈ 0** eigenvalue is a genuine **non-identifiable direction**: a combination of
  rates the data simply cannot pin down. *Non-identifiability shows up as **zero** eigenvalues, not
  negative ones* — an important correction we had been sloppy about. Flatness ≠ saddle.

So: can we certify a strict positive-definite minimum?

- **Deflation** (step along each negative eigenvector to its 1-D loss-minimum, which flips that
  direction's curvature positive and reduces the Morse index by one) removes the *deep* negatives
  — `λ_min` goes from −0.064 to ~−0.005, a 10× improvement — but then **bounces**. The bottom of
  the spectrum collapses into a near-degenerate cluster of near-zero eigenvalues that you can't
  resolve well enough to isolate a single direction to deflate. **The bare Hessian is not cleanly
  certifiable as PSD** — you hit a numerical / near-degeneracy floor where the tiny residual
  negatives are plausibly true zeros.

- **A diagonal (Jacobi) preconditioner doesn't help** and can't in principle. Empirically: the
  Hessian is only ~52% diagonal, its diagonal is wildly irregular (sign-indefinite, scales
  spanning ~11 orders of magnitude), so `D^{-1/2} H D^{-1/2}` only halves `λ_max` and blows
  `λ_min` to −1104. And fundamentally, by **Sylvester's law of inertia**, *no* symmetric positive
  preconditioner can change the **sign** of an eigenvalue — it can't manufacture a minimum out of a
  saddle.

- **A small Gaussian prior (MAP) can, and it's principled.** `H_MAP = H + λI` shifts *every*
  eigenvalue up by `λ`. A modest prior (`λ ≈ 0.10–0.13`, i.e. a std of ~3 on log-rates whose
  magnitude is ~6) makes the Hessian strictly PD — and **certifiably** so. The certificate
  lower-bounds

  ```
  λ_min(H_MAP)  ≥  (smallest Ritz value − Ritz residual) + λ
  ```

  so the Lanczos residual *cannot hide* a negative eigenvalue. Crucially, the prior regularizes
  exactly the flat non-identifiable direction the data couldn't constrain — that's a modeling
  decision, not a numerical hack. The required `λ` is set by **resolution** (how tightly we can
  bound the spectrum), not by the size of `|λ_min|`.

> Take-away: the **MLE** here has no clean strictly-PD minimum (a real ~0 non-identifiable mode plus
> an unresolvable near-zero cluster). The **MAP estimate** with a small principled prior *does*, and
> it sits in the same good basin ~173 NLL below the saddle the naive pipeline converges to.

---

## The recipe that works

End to end, on the converged objective:

```
Adam            constant-LR dive + best-loss tracking   →  reach the deep low-loss valley
  → L-BFGS      quasi-Newton + line search              →  descend the ravine into the good basin
  → [deflate]   optional negative-curvature steps       →  shrink the prior you'll need
  → MAP polish  Newton-CG on H + λI, λ auto-iterated     →  stationarize on the regularized objective
  → certify     residual-gated Lanczos λ_min(H_MAP) > 0  →  rigorous PD certificate
```

From a raw initialization this reaches a **certified PD minimum at NLL ≈ 137469**
(`λ_min(H_MAP) ≥ +0.036` at `λ = 0.128`), versus the naive Adam→Newton pipeline that stalls at the
saddle at 137640.

---

## Update — what we found by executing this plan (the open questions, resolved)

After this write-up we ran your recommended program (gauge audit → portfolio Adam→L-BFGS →
subspace deflation → "lock the objective" audit). Results:

- **The "3.4 NLL gap" was the wrong thing to chase — the real lever is the *initialization scale*.**
  Sweeping the initial DTL rate, the raw-NLL floor drops far below the old 137466: 137470 (init
  ~0.077) → 137461 (0.15) → **≈137384 (0.22–0.25)**. L-BFGS memory/restarts were *not* the lever
  (`maxcor` 10/50/100 land identically; the 137466 floor was a true L-BFGS fixed point). So it was
  never "longer L-BFGS" — it was "start from a higher rate."

- **Basin sensitivity is now characterized: strongly init-dependent, and the small-init direction is a
  trap.** A 1e-5 init falls into a *steep* trap basin (137897, a clean λ_min=−0.23); higher inits reach
  progressively deeper basins. The good basin is **not** a global attractor.

- **The new deep basin is REAL, not a truncation artifact** (your "lock the objective" point was
  decisive). At 137384 the loss is bit-identical from pi=64→1024 and across e_tol 1e-6→1e-10, and the
  neumann=64 gradient matches neumann=256 to 4 decimals — a hi-fidelity re-opt descends 0.0 further.
  (Aside: neumann=16 gives a gradient 81° off true — the original broken fixture defaults.)

- **No softmax gauge** — the parameterization is already gauge-fixed (4-category softmax, reference
  logit pinned to 0), so there are no ~1331 artificial null directions and no row-centering is needed.

- **Subspace deflation works** (fixed the single-vector bounce — 5 monotone rounds) but moves only
  ~0.1 NLL: deflation is a curvature/certification tool, not a basin-finder. The bare-H bottom stays at
  the numerical near-degeneracy wall (most-negative restricted Ritz −6.7e-3, residual 3.1e-2,
  unresolved even fp64 m=200) — so **bare-H PSD stays a "no,"** and MAP remains the clean certifiable
  route. Confirmed across single-vector & subspace, fp32 & fp64.

- **The deepest finding reframes the whole goal.** In *every* optimized solution (old and deep alike),
  **~half the per-state rates run to the 0/1 boundary** (47% of DTL probabilities < 1e-3; logits to
  −20). That's the MLE responding to **sparse per-state events** — those rates are *unidentifiable*
  (the data can't tell 1e-4 from 1e-6), the likelihood is flat there, and that flatness **is** the
  near-zero Hessian directions. The "deeper basin" is partly just the unidentifiable rates wandering
  further (≈3 NLL of the gap is run-to-run scatter in those directions). So the **raw MLE is not a
  well-posed estimator** — ~half its parameters are statistically meaningless point estimates.

⇒ **Next phase: stop chasing the raw MLE; switch to MAP / penalized likelihood with a
cross-validated prior** (Sanderson 2002). The prior both regularizes the unidentifiable boundary rates
to sensible values and yields the certifiable PD minimum that bare-H cannot. Plan + the reproducible
audit scripts are in the repo (`newton/_map_cv_plan.md`, `newton/_specieswise_basin_findings.md`,
`newton/{basin_search,convergence_audit,gauge_audit,theta_diagnostics}.py`).

---

### TL;DR for the impatient

We spent a long time trying to drive the gradient to zero, not realizing the point we were
converging to was a **saddle**, not a minimum — and the bad direction was **orthogonal to the
gradient**, which made it invisible to Adam (which avoids saddles) and irresistible to Newton
(which snaps onto them). The fix was to stop using Newton to *find* the optimum (use Adam→L-BFGS,
which slides past the saddle into a basin ~173 NLL lower) and bring Newton back only to *polish and
certify* a MAP-regularized objective — because the raw likelihood has a genuine non-identifiable
flat direction and no clean positive-definite minimum, while a small principled Gaussian prior
gives one you can rigorously certify.
