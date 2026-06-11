# Capture/replay fidelity fixes

This document records the problems found while building the benchmark harness and how each was
diagnosed and fixed. **None of these were bugs in the vendored kernels.** They were fidelity problems in
*replaying a kernel in isolation and comparing it to a golden reference* — the hard part of making an
automated optimizer's pass/fail signal trustworthy. Each fix was confirmed with a negative test
(inject a known error → the gate must catch it; and a known-irrelevant change → the gate must ignore it).

The naive harness (clone every argument independently, snapshot the live output, compare exactly)
**passed the whole-pass forward/backward but failed many per-kernel checks on byte-identical kernels**
— i.e. it reported false failures. Four root causes, found one at a time with evidence.

---

## Issue 1 — `-inf` in log-space tensors read as a mismatch

**Symptom.** Several per-kernel comparisons reported `FAIL … nan/inf mismatch` while showing
`max_abs = 0.0, max_rel = 0.0`. The values were identical.

**Evidence.** The failing tensors were log-space (Pibar, the `SL`/`DL` constants, `dts_r`) and hold
`-inf`. The comparison computed `diff = |got - ref|`; at a position where both are `-inf`,
`(-inf) - (-inf) = NaN`, and a blanket `torch.isfinite(diff).all()` check then flagged the whole
tensor as non-finite — even though every element was equal.

**Fix.** Make the comparison inf/NaN-safe (`bench/_common.py::compare`): treat a position as
**agreement** when `got == ref` exactly (which is true for matching `+inf`/`-inf`) or when both are
NaN. Only genuinely disagreeing positions contribute to `max_abs`/`max_rel`; a disagreement against a
non-finite value scores `inf` (a real failure).

---

## Issue 2 — broken argument aliasing

**Symptom.** `compute_wave_step` failed on argument `Pibar` (`max_abs ≈ 23`), yet the kernel was
**deterministic** — running it twice on the captured inputs gave identical results, which merely
disagreed with the golden captured during the real forward.

**Evidence.** Inspecting the captured call (`int_sig = (ws=1583, W=4899, S=119, …)`, an even wave
iteration) showed `arg[1]` (`Pi_out`) and `arg[2]` (`Pibar`) were both the full `[C, S]` buffer and
both mutated by the call. In the forward loop the same `pibar` buffer is passed as **both** the
output and the `Pibar` argument on even iterations — they *alias*. The naive capture deep-cloned each
argument independently, so on replay they were two separate tensors; the kernel's in-place writes no
longer interfered as they did in the real run, so the post-state diverged.

**Fix.** Preserve aliasing across capture and replay (`kbench/_capture_io.py`). Inputs are
**content-addressed** into a per-size pool and each call records, per tensor, a group id assigned by
*storage identity within that call*. On rebuild, one fresh device tensor is materialized per group —
so arguments that shared storage in the original share one object again (in-place writes alias
correctly), while distinct arguments stay distinct.

**Verification.** With aliasing restored, `compute_wave_step` (and the other forward kernels) matched
golden exactly.

---

## Issue 3 — uninitialized (active-mask-pruned) output regions

**Symptom.** Backward kernels failed on their **return** values — `wave_backward_uniform_fused`'s
`v_k`, `dts_cross_backward_accum_fused`'s `grad_Pibar_*` — with huge `max_abs` (hundreds to
thousands), even after aliasing was fixed.

**Evidence.** For `wave_backward_uniform_fused` the captured call had 8 active rows out of 20. Comparing
replay vs golden: **all 12 mismatching rows were exactly the *inactive* rows; 0 active rows
mismatched.** These kernels allocate outputs with `torch.empty` and only write the rows selected by
`active_mask` (the adjoint is pruned); inactive rows are uninitialized garbage. The golden froze the
real run's garbage; the replay produced different garbage.

**First attempt (rejected).** Detect uninitialized positions by running the kernel twice with GPU
*free memory poisoned* differently between runs, expecting unwritten positions to read different
poison. **This failed for small tensors**: `v_k` is `[20, 119]`, and a 384 MB poison buffer never
dirties such a tiny allocation — both runs returned identical `v_k` (diff `0.0`, no poison values), so
nothing was flagged.

**Fix.** Poison the *outputs directly*, at any size: during the two capture runs, monkeypatch
`torch.empty`/`torch.empty_like` to pre-fill float tensors with a different sentinel each run
(`capture/capture.py::_fill_empty`). Any position the kernel does **not** write keeps the sentinel, so
the two runs diverge there (`|diff| > 1e15`) and that position is recorded as a **don't-care** mask;
genuinely-written positions are sentinel-independent and stable. The bench then compares only the valid
(written) positions.

**Verification.** Negative test on `wave_backward_uniform_fused`'s `v_k`:
- corrupt **active** rows → `FAIL` (caught, `max_abs = 0.5`);
- corrupt **inactive** rows only → `PASS` (correctly ignored).

So the mask excludes uninitialized garbage without hiding real errors.

---

## Issue 4 — atomic-accumulation nondeterminism

**Symptom.** Some backward outputs differed between two **live** runs of the *same* kernel on the
*same* inputs — e.g. `dts_cross_backward_accum_fused`'s `grad_Pibar_r` varied by up to ~4700 in
absolute terms run-to-run.

**Evidence.** These kernels accumulate across rows/contributions with GPU atomics; floating-point atomic
add order is nondeterministic, so the result is not bit-reproducible. A single stored golden snapshot
can never be matched exactly.

**Fix.** Record a per-tensor **noise floor** at capture time (`measure_noise`): run the kernel twice
and store, per output, the max abs/rel difference between the two runs (at valid positions). The bench
widens the effective tolerance to `max(tol, 4 × noise)`, so a faithful rewrite is judged against the
reference's *own* irreducible noise rather than spurious bit-drift. The whole-pass gradients, being
reductions, stay within the normal `2e-3` tolerance regardless.

---

## Disk footprint

A side effect of the naive capture was size: each per-kernel record stored the full `[C, S]` Pi/Pibar
buffers in both pre- and post-state, and every record stored its own copy — `data/small/kernels` was
**7.3 GB**, with individual files up to ~580 MB.

The content-addressed pool from Issue 2 fixes this too: every distinct buffer (by bytes) is stored
**once** per size in `kernels/_pool.pt`, and each per-call record is just a nested structure of
`Ref(content_idx, group_id)` sentinels plus metadata. Records dropped to **~4 KB** each; `data/small`
went **7.3 GB → ~2.7 GB**. The remainder is genuinely-distinct `[C, S]` buffers captured at different
waves — inherent to capturing multiple waves, and tunable with `--per-wrapper`.

---

## How to re-verify the harness itself

The negative tests above are the way to confirm the gate still detects regressions after any change to
the harness:

```python
# inject a known error into a vendored kernel wrapper, then run the bench check — it MUST fail
import kbench.core.kernels.wave_step as ws
orig = ws.compute_wave_step
ws.compute_wave_step = lambda *a, **k: (orig(*a, **k), a[1].narrow(0, int(a[3]), int(a[4])).add_(0.01))[0]
# bench/forward.py and bench/kernels.py --kernel compute_wave_step should now report FAIL
```

If an injected active-region error does **not** produce a `FAIL`, the harness is over-masking and must
be investigated before trusting its PASS verdicts.
