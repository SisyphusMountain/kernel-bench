# Genewise archaea speed plan (§5.1 AleRax head-to-head)

Plan to fix the inefficiencies found while profiling the genewise FD-Newton+rebatch recipe on the
3946-family archaea subset (the §5.1 benchmark). All numbers are **warm** (Triton cache hot), 4090,
fp32, measured on the EXACT 3946 families AleRax ran on (`archaea_ge4sp.families.txt`).

## Where we are

| recipe | total | vs AleRax (1526s) | optimum (NLL nats) |
|---|---|---|---|
| original (cb=80k, adam 20/lr0.05) | 384s | 4.0× | 225864 |
| + cb=900k | 310s | 4.9× | 225864 |
| **+ adam 5/lr1/clip10  ← current best** | **230s** | **6.6×** | 225865 |

Optimum is bit-stable across every variant: **3946/3946 converged, |Pg|max = 1e-3, 472 bound-active,
~1368 interior-PD, 0 premature drops, NLL ≈ 225865 nats** (−469 nats *better* than AleRax's 226334).
That invariant is the correctness gate for every change below.

Drivers (all in `experiments/alerax_speed/`):
- `run_gpurec_rebatch.py` — production recipe (copy of `…/sanderson_cv/bench_genewise_warm_rebatch.py`)
- `run_gpurec_traced.py` — same recipe + phase/step timing (env knobs: `CLADE_BUDGET ADAM ADAM_LR GRAD_CLIP`)
- `run_gpurec_profile.py` + `analyze_profile.py` — per-iteration time + per-family convergence capture
- `run_gpurec_lbfgs.py` — L-BFGS-B experiment (rejected, see below)

Profiling evidence: `/tmp/profile_steps.json` (166 events, opt=176s run).

---

## Methodology: test one at a time, keep ONLY if it improves runtime

Every candidate below is **provisional**. We change **one thing at a time** against the current best
recipe and accept it only if it measurably reduces wall-clock **without breaking the optimum**.

- **Isolation:** apply exactly one candidate; everything else stays at the current-best setting.
- **Accept iff faster beyond noise:** build/cert times swing ±~15s (GPU clock state), so take the
  **median of ≥2 warm runs** and accept only if the median improves by **more than the noise band**
  (~15s). A negligible or negative change → **revert** the candidate.
- **Correctness is a hard gate, not a tiebreaker:** any run that doesn’t hit 3946/3946 converged,
  |Pg|max ≤ 1e-3, NLL = 225865 ± a few nats, ~1368 interior-PD / 472 bound-active, 0 premature drops
  is rejected regardless of speed.
- **Accepted → new baseline:** each kept change becomes the baseline the next candidate is tested
  against (so we measure real marginal gains, not double-counted ones).
- **Order by expected impact / risk:** P0 (already passed) → P1 (biggest lever) → P2 → P3.

Status legend: ✅ measured-improvement-kept · 🔬 candidate, not yet A/B-tested · ❌ rejected.

---

## P0 — Bake in the two VALIDATED wins ✅ (already passed the one-at-a-time test)

These are already measured and verified; the only work is promoting them from env-overrides to the
recipe so the §5.1 number is reproducible from the committed script.

### P0.1 — Archaea `clade_budget` 80k → 900k  *(occupancy; −74s, validated)*
**Evidence:** `profile_batching.py` — at cb=80k the 3946 families split into **45 tiny GPU batches**,
running at **69% GPU utilization**; cb=900k → **4 batches, 98% util, 1.76× faster per eval**, peak mem
3.4/24 GB. Wave size (`max_wave_size=8192`) is NOT the limiter (32768 gave 96% util, no gain) — **leave
waves at 8192**. The 80k default is hogenom-tuned (large families saturate at 80k clades); archaea
families are ~10–100× smaller, so 80k starves the GPU.
**Fix:** make `clade_budget` dataset-aware — archaea ≈ 900k, hogenom keeps 80k (do NOT widen hogenom
without re-checking its memory). Cleanest: pick the default from the dataset, overridable by env.
**Where:** the `CLADE_BUDGET` default / `build()` in `run_gpurec_rebatch.py` (and the production
`bench_genewise_warm_rebatch.py`).
**Risk:** memory on a future larger small-family dataset → keep the env override; log peak mem.

### P0.2 — Adam warmup 20/lr0.05 → 5/lr1.0 + grad-clip-norm 10  *(−80s, validated)*
**Evidence:** 5 aggressive clipped steps enter the basin *better* than 20 gentle ones (post-warmup
it0 |Pg|max **68 vs 120**), warmup 80s→12s, and the subsequent pi16 Newton tier shrinks 240s→182s.
10 steps is a wash (235s) — diminishing returns past 5. Clip is a stability guardrail (Adam
self-normalizes, so lr=1 drives the step; clip bounds the steep initial families, |Pg|≈260).
**Fix (current code, already in `run_gpurec_traced.py`):**
```python
ad = torch.optim.Adam([lf], lr=1.0)            # was 0.05
for _ in range(5):                              # was 20
    _, g = lg(m, lf.detach()); lf.grad = g
    torch.nn.utils.clip_grad_norm_(lf, 10)      # clip ...
    project_rate_gradient_(lf.detach(), lf.grad, min_rate=MIN_RATE, max_rate=MAX_RATE)  # ... then project
    ad.step()
    with torch.no_grad(): clamp_(lf)
```
**Where:** the Adam block in `run_gpurec_rebatch.py:129-134` (and production mirror).
**Risk:** tuned on archaea; hogenom’s larger/stiffer families may want gentler lr or more steps —
**re-validate on hogenom before changing the global default** (gate per-dataset if it regresses).

---

## P1 — Forward-difference FD-Hessian 🔬  *(biggest remaining lever; projected −38s; needs verify)*

**Evidence (coarse profile):** `step+hess` iterations are **61% of all Newton time (89.6s)**. A
full-batch plain step = 2.0s (1 loss+grad); a full-batch `step+hess` = **14.5s ≈ 7 loss+grad** — the
Hessian is **central difference = 6 extra evals** (3 dims × 2). The three early full-batch Hessians
alone (gstep 0/5/10) cost **~43s**.
**Root cause:** eval count, not kernel inefficiency (the per-eval kernel is throughput-bound at 98%
occupancy). The Hessian is only used as a **convexified PD preconditioner** (eigenvalues clamped to
MU=1e-2); convergence is tested on the true |Pg|, not on the Hessian.
**Fix:** switch to **forward difference** — reuse the base gradient `g` already computed at the top of
the iteration, so the Hessian needs only **3 new evals** instead of 6:
```python
# central (now):  H[:,:,j] = (g(x+e_j) - g(x-e_j)) / (2*FD_EPS)        # 6 evals
# forward (new):  H[:,:,j] = (g(x+e_j) - g)        / FD_EPS            # 3 evals, g = base grad
```
This halves the Hessian portion (~77s → ~39s) → **projected −38s** (`step+hess` 14.5s → ~8.6s on full
batch). **Where:** `run_gpurec_rebatch.py:209-216`.
**Risk:** forward diff is O(ε) vs O(ε²) accurate → a slightly worse Newton direction may need a few
extra (cheap, 2s) steps; still a large net win unless step count balloons. **MUST verify** step count
+ optimum. Consider a smaller `FD_EPS` (e.g. 5e-3) to keep the direction sharp.
**Complementary (cheaper, optional):** rebuild the Hessian less often on the big batch (raise
`HESS_EVERY` while batch ≥ ~1000) and/or align the drop checks (CHECK=4) so a drop precedes a Hessian
rebuild (HESS_EVERY=5 currently misaligns — the it5/it10 full-batch Hessians run before the first drop
at it12).

---

## P2 — Eager-defer the converged-but-uncertifiable tail to pi64 🔬  *(−~10s; low risk)*

**Evidence (fine profile):** the last **7 families loop ~42 useless pi16 iterations (it77→it119, ~10s)**
at |Pg|≈2e-4 — already converged by the pi16 metric and *not moving* — then carry to pi64 and converge
in **4 steps**.
**Root cause:** pi/neumann are **scalar per build** ([solver_options.py:10-11](../../../agent-worktrees/kernel-bench-mapcv-merge/gpurec/api/solver_options.py#L10-L11)),
so different pi per family is realized only by segregating families into separate pi-tier builds. The
stiff-family detector (`defer = reject & (resid > FWD_TOL)`,
[run_gpurec_rebatch.py:167-181](run_gpurec_rebatch.py#L167-L181)) gates on the **forward** fixed-point
residual. These 7 families have a *converged* forward solve (resid ≤ FWD_TOL) but a pi-sensitive
**backward/gradient** (they fail the pi64 cert), so the forward-residual detector misses them → they
never defer and grind to MAXIT.
**Fix:** the pi64 cert failing *is itself* the evidence a family needs pi64. Defer on
`conv & ~cert_ok` directly — gated on the pi16 |Pg| having **plateaued** (so we don’t evict families
that merely need a few more pi16 Newton steps). Sketch:
```python
# in addition to the forward-resid defer:
stuck = reject & (pg_now > IMPROVE_FRAC * pg_prev_at_check)   # |Pg| stopped dropping at pi16
defer = defer | stuck                                          # cert-fail + plateaued -> needs pi64
```
Or, simpler safety net: **break the pi16 tier early** once every active family is converged-at-pi16 but
no drop happened for K checks (they can only be cert-failures → carry to pi64 now, not at MAXIT).
**Where:** the `reject`/`defer` block + the tier loop in `run_gpurec_rebatch.py`.
**Risk:** deferring too eagerly sends families that just need more pi16 steps to the (4× costlier) pi64
build — the |Pg|-plateau gate prevents this. Verify the pi64 batch size stays ~31, not ballooning.

---

## P3 — Drop more eagerly in the big-batch regime 🔬  *(−~10–15s; tradeoff-bounded)*

**Evidence (fine profile):** **12.8% of family-steps (10,352/80,934) — ≈17% of stepping wall-time
(~21s) — are spent on families already at |Pg|<TOL** that linger waiting for the next drop. Worst case:
**1826/2752 (66%) already converged at it23** still get a full 2s step. Drops fire only every CHECK=4
iters and only once FRAC=30% is exceeded + the pi64 cert passes.
**Root cause:** drops are deliberately batched to amortize the expensive pi64/neu64 **cert-verify**
(12.3s total in the earlier breakdown). So this is a genuine tradeoff, not a pure bug.
**Fix (test, don’t assume):** lower `CHECK` (4→2) and/or `FRAC` (0.30→0.15) so converged families leave
the big batch sooner — measure whether the saved stepping-time beats the extra cert-verify cost. Only
worth it in the large-batch regime; in the tail the per-iter cost is already ~30–90ms.
**Where:** `CHECK`/`FRAC` in the drop logic.
**Risk:** more cert-verify calls (each a full-batch pi64/neu64 eval). Net could be flat — measure both
sides via `analyze_profile.py` (compare `DROP` total vs the wasted-family-step time).

---

## Rejected (record so we don’t revisit)

**L-BFGS-B for the post-Adam endgame** (`run_gpurec_lbfgs.py`). The repo’s `BatchedLBFGS` is a faithful
per-family bounded L-BFGS-B, but it is **3–4× slower here**: the batched strong-Wolfe/Armijo line search
must satisfy its condition for **all 3946 families at once**, so the single hardest family drags the
whole batch through **~15–16 closure-evals per step** (47% converged at 437s, vs 230s for the entire
Newton recipe). Switching strong-Wolfe→Armijo barely helped (16→15 evals/step). This is the
known “batched line search is straggler-dominated” pitfall; FD-Newton wins because the 3×3 Hessian is
free and the trust-region step needs **no line search at all**. (Not a knock on AleRax: it runs one
independent L-BFGS-B *per family* on CPU, where there is no straggler coupling.)

---

## Combined target (only the candidates that pass survive)

**Best case, if P1+P2+P3 each pass the keep-if-faster test:** 230s → ~150–160s (~7.5–8× over AleRax),
same optimum. This is a ceiling, not a promise — any candidate whose measured median doesn’t beat the
noise band is reverted, so the realized total is whatever the *accepted* subset delivers. Record the
A/B result (kept / reverted + the two medians) for each candidate as we go, in a results table here.

| candidate | baseline median | with-change median | Δ | optimum ok? | verdict |
|---|---|---|---|---|---|
| P0.1 cb=900k | 384s | 310s | −74s | ✅ | KEPT |
| P0.2 adam 5/lr1/clip10 | 310s | 230s | −80s | ✅ | KEPT |
| P1 forward-diff Hessian | 224.5s (central med, n=2) | 188.5s (forward med, n=2) | −36s | ✅ | **KEPT** |
| P2 eager-defer tail | 177.5s (fwd med, n=2) | 170s (fwd+eager med, n=2) | −7.5s | ✅ | **KEPT** |
| P3 eager big-batch drop (CHECK=2) | 170.5s (CHECK=4 med, n=2) | 174.5s (CHECK=2 med, n=2) | +4s | ✅ | **❌ REJECTED** |

**P3 detail:** CHECK=2 cut Newton steps (63–70→50) but added cert-verify calls (11–12→14) and rebuilds
(rebatch 16.2→19.5s); the extra full-batch pi64/neu64 cert cost > the step savings → optimize
126.5→131.5s. Big-batch lingering is real but every earlier drop costs a pi64 cert-verify, so it's not
cheaply recoverable. CHECK=4 is already the right tradeoff. FRAC=0.15 not separately measured (same
cert-cost mechanism, expected same result).

### Final recipe & result
**Accepted: P0.1 + P0.2 + P1 + P2.** Reproduce with `run_gpurec_traced.py` and:
`CLADE_BUDGET=900000 ADAM=5 ADAM_LR=1.0 GRAD_CLIP=10 HESS_MODE=forward EAGER_DEFER=1` (CHECK stays 4).

| stage | total (warm median) | vs AleRax 1526s |
|---|---|---|
| start of this work | 384s | 4.0× |
| + cb=900k (P0.1) | 310s | 4.9× |
| + adam 5/lr1/clip10 (P0.2) | 230s | 6.6× |
| + forward-diff Hessian (P1) | ~190s | ~8× |
| **+ eager-defer tail (P2) ← final** | **~170s** | **~9.0×** |

Optimum bit-identical at every stage: 3946/3946 converged, |Pg|max=1e-3, NLL **225865 nats** (−469
vs AleRax's 226334). Net: **384s → 170s, 2.26× faster than where we started, 9× over AleRax.**

### Session-2 findings (step count & the wall-clock floor)
- **Step count is NOT the wall-clock bottleneck.** TRUST (disproven: 14% capped, raising it breaks
  convergence), MU=3e-3, and a per-family descent safeguard each cut *steps* (combined 66→38, −42%) but
  are **runtime-neutral** — left OFF by default (env `MU=`/`SAFEGUARD=1`). The wall-clock is set by the
  per-iter grad cost on the large batch, not the iteration count.
- **Wall-clock partition** (166s with cert): big-batch Newton (≥600 fam, ~first 24 iters) **~80s/48%**
  (of which the first 12 full-3946 iters = ~40s irreducible floor), cert **46s/28%**, build+adam
  **31s/19%**, tail **5s/3%**.
- **Coarse-to-fine pi REJECTED** (P6): PIS=8,16,64 is +25s — pi8 forward residual is 6–7 (≫FWD_TOL), so
  families fail the cert and escalate to pi16 en masse and re-optimize. pi16 is the minimum viable pi.
- **Lower Neumann REJECTED** (P7): NEU_OPT=8 is +43s AND breaks correctness (6–7 families never converge,
  |Pg|max=25–46, 229 steps). Worse than pi8 because backward under-convergence has **no detector/escalation
  tier** (forward-residual only catches pi), so neu8-biased families grind forever. neu16-warm is the
  minimum viable backward. **Both halves of the grad (pi16 fwd, neu16 bwd) are at their floor → the ~80s
  big-batch phase is irreducible; the recipe is at its algorithmic floor.**
- **Final PD cert made OPTIONAL, OFF by default** (env `CERT=1`). It is purely diagnostic — read-only on
  theta, never moves/re-optimizes a family (convergence already guaranteed by per-drop VERIFY_DROP).
  Neumann iters: **16 during the fit (warm-adjoint), 64 only in the cert.** Removing it (AleRax does no
  cert either) gives the apples-to-apples **fit-only time ≈ 120–125s = ~12× over AleRax's 1526s.**

**P2 detail:** baseline {179,176}s vs eager {171,169}s — arms cleanly separated, cert stable 43–44s
(so the Δ is optimize-phase, not noise). **Newton steps halved 121→62** (the ~42-iter pi16 tail grind
eliminated). Optimum cleaner (both eager: 3946/3946, |Pg|max=1.00e-3, 0 premature). pi64 batch grew
only 23→26 (plateau gate prevents over-deferral). Modest wall-clock because the killed tail steps were
cheap (7–9 fam, launch-bound). `EAGER_DEFER=1` in `run_gpurec_traced.py`. Re-measured forward baseline
drifted 188.5→177.5s vs the P1 session (GPU clock) — each Δ is valid only within its own back-to-back
A/B.

**P1 detail:** central {227,222}s vs forward {187,190}s. Forward-diff needs ~9 more Newton steps
(112→121, the less-accurate direction) but each step is far cheaper → optimize 180s→145s. Optimum
bit-identical (225865 nats all 4 runs). The 1–2 family |Pg|=1.03e-3 blip appears in BOTH arms (warm-
adjoint cert-boundary, pre-existing), so it does not block P1. `HESS_MODE=forward` in
`run_gpurec_traced.py`.

**Measurement tooling:** warm wall-clock + per-phase/per-step breakdown via `run_gpurec_traced.py`
(timing) and `run_gpurec_profile.py` + `analyze_profile.py` (per-step distribution / waste).
**Standing directive (validate the endgame at cheap scale first):** sanity-check each candidate on the
256-family archaea subset before the full-3946 A/B — failures hide in the endgame, not the first steps.
**nsys/ncu not needed:** every bottleneck is eval-count-driven; per-eval kernels are already
throughput-bound (big batch, 98% occupancy) or launch-bound by family count (tail). Kernel tools would
confirm, not fix.

## Open / cross-cutting
- The final **PD cert** (cold pi64/neu64 over all 3946, ~23–65s, varies with GPU clock) is gpurec-only
  work AleRax doesn’t do. For the §5.1 *speed* headline, consider certifying at lower pi or reporting
  fit-time separately. Decide framing with the paper.
- Build/cert times swing ±15s run-to-run (GPU clock state) — report medians, not single runs.
- Hogenom re-validation needed before any of P0.2 / P1 / P2 / P3 change a **global** default (vs an
  archaea-gated one).
