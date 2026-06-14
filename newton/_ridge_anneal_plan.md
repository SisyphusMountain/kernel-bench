# Ridge-annealing (λ-continuation) Newton polish

## Context

The single-λ ridge Newton polish in `newton/optimize.py` (committed `f0a5cf0`) revealed a tradeoff on
the representative `666x80` fixture (characterized 2026-06-14):

| endgame | final NLL | final ‖g‖ (unreg) | wall | behavior |
|---|---|---|---|---|
| Adam-only (valley floor) | **137540** | 182 | 167 s | lowest loss, NOT stationary |
| Newton lam=0 (solver damping only) | 137669 | 7.8 | 639 s | ‖g‖ **bounces**, CG hits max_iter |
| Newton ridge λ=13.7 (objective MAP term) | 137688 | **5.11** | 141 s | CG converges ~10 iters, ‖gF‖ **monotone** |

A **big ridge λ** conditions the system so CG converges fast and ‖g‖ descends monotonically — but the
MAP term `λ/2‖θ−θ_ref‖²` *biases the minimizer toward θ_ref*, so it stalls ~0.1% above the valley
floor. A **small λ** targets the true optimum but is ill-conditioned and bounces. The user's insight:
**anneal λ** — start big (stable, fast CG), shrink it warm-started so the target slides toward the
valley floor, using the **safe-step (CG negative-curvature witness) logic** as the per-step safety net
*and* the stop signal for how low λ can safely go. This is classic λ-continuation / homotopy, and it
maps directly onto the existing `cg_witness` + exact-HVP machinery.

Goal: a `ridge_anneal` polish that reaches the valley floor (≈137540) with a small, stable ‖g‖, in
≲ the single-λ ridge wall time, and is dataset-agnostic.

## Two regularizers, clean division of labor (why this works)

- **Outer ridge λ** (`λ/2‖θ−θ_ref‖²`): adds `λI` to the Hessian AND moves the minimizer. This is the
  *global conditioning schedule* we anneal. Large λ ⇒ PD, fast CG, no bounce, biased; λ↓ ⇒ target →
  true optimum.
- **Inner witness damping** (`cg_witness` bump `δ ← ν(δ−cert)`): adds `δI` to the *solve only* (does
  not move the minimizer). The *per-subproblem safety net*: when annealing pushes `H+λI` indefinite,
  the witness fires (`status="neg_curv"`, `cert = dᵀAd/‖d‖² ≤ 0`) and bumps `δ` to keep the step safe.
- **Stop signal**: while the witness stays quiet, there is conditioning headroom → keep shrinking λ.
  When it fires every step (and δ climbs to compensate), the bare problem is in the bounce regime →
  stop annealing. Read straight from the per-step history.

## Design — new `ridge_anneal()` in `newton/optimize.py`

A purpose-built continuation loop that owns the λ schedule and reuses the verified safe-step
primitive directly (avoids the double-damping that calling `newton_lanczos` would introduce, since
that always injects its own `σ·λ_max` damping). ~80 lines.

```
ridge_anneal(static, theta0, col_weights, *, lam0=None, sigma=0.01, theta_ref_mode="moving",
             inner_steps=3, max_levels=8, max_cg=30, nu=1.5, max_bumps=3, gtol=1e-2, verbose):
  build exact-fp32 hvp + value/grad at theta (reuse make_exact_hvp, make_value_and_grad)
  lam_min,lam_max = lanczos_extremes(exact-HVP, m≈20)      # up-front spectrum (reuse cg.py:13)
  lam = lam0 or (-min(lam_min,0) + sigma*lam_max)          # auto_lambda rule (the validated start)
  theta_ref = theta
  for level in range(max_levels):
    rebuild hvp at current theta (theta moved); A = v -> hvp(v) + lam*v   # ridge-damped operator
    for _ in range(inner_steps):                            # LOOSE subproblem solve, warm-started
      gL = grad_L(theta); gF = gL + lam*(theta - theta_ref)
      delta = 0.0
      for bump in range(max_bumps+1):                       # safe-step: witness self-correction
        p, it, status, cert = cg_witness(lambda v: A(v)+delta*v, -gF, tol=eta*||gF||, max_iter=max_cg)
        if status != "neg_curv": break
        delta = nu*(delta - cert)                           # reuse the newton_cg.py:165-177 pattern
      theta = armijo_backtrack(theta, p, F_lam)             # deterministic forward loss (forward_solve)
      record {level, lam, loss, gLnorm, gFnorm, cg=it, fired=(cert is not None), delta}
    if theta_ref_mode == "moving": theta_ref = theta        # proximal: re-center each level (default)
    lam = anneal(lam, level_history)                        # ADAPTIVE schedule (below); None -> stop
    if lam is None or lam < lam_floor or gLnorm < gtol: break
  return theta, history
```

**Adaptive λ schedule** (same bold-driver philosophy as the adaptive LR already in `optimize.py`):

```
def anneal(lam, level_hist):
    fired   = any(h["fired"] for h in level_hist)
    cg_hard = level_hist[-1]["cg"] >= max_cg
    improved = level_hist[-1]["loss"] < level_hist[0]["loss"] - tol
    if not fired and not cg_hard and improved: return lam * 0.3   # clean & cheap -> shrink fast
    elif fired or cg_hard:                       return lam * 0.7   # near the edge -> ease off
    else:                                        return None        # no progress -> stop
```

**θ_ref policy**: default `"moving"` (proximal: re-center on the current iterate each level → each
subproblem is a damped step *from where you are*, radius grows as λ↓; most robust against the bounce
and slides toward the floor). Option `"fixed"` (Tikhonov homotopy from θ0 to the MLE).

### Reuse (do not reimplement)
- `cg_witness` ([newton/cg.py:92](newton/cg.py)) — the safe step + certificate.
- the bump loop `δ ← ν(δ−cert)` — mirror the proven pattern in `newton_lanczos`
  ([newton/newton_cg.py:165-177](newton/newton_cg.py)).
- `lanczos_extremes` ([newton/cg.py:13](newton/cg.py)) — up-front λ_min/λ_max on the exact HVP.
- `make_exact_hvp` ([newton/hvp_exact.py](newton/hvp_exact.py)), `make_value_and_grad` /
  `forward_solve` ([newton/vg.py](newton/vg.py)) — HVP, gradient, and Armijo loss.
- `_exact_ridge_lambda` (already in `optimize.py`) — the auto_lambda(exact-HVP) λ0.

## Integration
- Add `ridge_anneal()` to `newton/optimize.py`; expose via `optimize(..., polish_mode=...)` with
  `polish_mode ∈ {"ridge", "ridge_anneal", "lanczos", "none"}` (keep `"ridge"` default, add the new
  mode) and a CLI `--polish-mode`. The `bench`/`newton_polish` paths stay as-is.

## Verification (666x80)
1. From the Adam/adaptive endpoint, run `ridge_anneal`; compare to the three baselines above
   (single-λ ridge 137688/5.1/141 s; lam=0 137669/7.8/639 s; Adam floor 137540/182).
2. **Success criteria**: final NLL ≤ ~137560 (within ~0.015% of the Adam valley floor) AND ‖g‖ small
   and *stable* (≤ ~5, no late bounce), in ≲ the single-λ-ridge wall (~141 s).
3. Report the **λ → loss → ‖g‖ → witness-fires** trajectory per level (the headline artifact: does
   annealing slide loss down while ‖g‖ stays controlled, and where does the witness start firing —
   the λ-floor).
4. Sanity: final loss matches an independent `forward_solve`; no kernel changes ⇒ HVP FD gates
   unaffected.
5. Save JSON + append findings to `newton/_optimize_findings.md`.
6. Operational: `/home/enzo/miniforge3/bin/python3.12`; check `nvidia-smi` for a compute co-tenant;
   `GPUREC_MEMORY_POLICY_RESERVE_GIB=0.25`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if a
   bigger fixture is tried.

## Deliverables / steps
1. **Persist this plan to `newton/_ridge_anneal_plan.md`** (repo copy of the design doc).
2. Implement `ridge_anneal()` + adaptive `anneal()` + `polish_mode` wiring in `newton/optimize.py`.
3. Run the 666x80 characterization; iterate the schedule constants (0.3/0.7, inner_steps) if the
   floor isn't reached or the witness fires too early.
4. Update `newton/_optimize_findings.md`; commit.

## Out of scope
- Re-tuning HVP kernels (tracked in `_hvp_profiling_report.md`).
- A full multi-dataset sweep — validate on 666x80 first; the method is dataset-agnostic by
  construction (relative λ schedule, witness-driven stop).
