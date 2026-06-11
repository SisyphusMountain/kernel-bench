# kernel-bench

A **self-contained** benchmark for a set of Triton kernels, with **golden-reference correctness
gating**. Built so an automated system can rewrite the kernels and get a fast, trustworthy signal —
*did it stay correct, and did it get faster* — for the **forward pass** and the **backward pass**
separately. It depends on nothing but `torch` + `triton`; there is no external library to import.

The work reduces, underneath, to two GPU operations repeated thousands of times:

| pass | maps | dominant kernels (edit surface) |
|------|------|---------------------------------|
| **forward**  | `theta → loss` | `e_step` (forward), `wave_step`, `dts_fused` |
| **backward** | `forward intermediates → grad_theta, grad_col` | `wave_backward`, `e_step` (backward) |

Both are an iterative log-space solver: a primary fixed point over a band-structured state matrix
(`C` rows × `S` columns), an auxiliary fixed point (the "E-step"), a scalar loss, and an
implicit-differentiation adjoint for the gradient. The kernels do the heavy per-row work.

**Docs:** [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — full guide to what you can run and edit, and how
correctness is judged. [docs/FIXES.md](docs/FIXES.md) — the capture/replay fidelity problems found
while building this and how each was fixed.

---

## Layout

```
kbench/                        # the vendored solver (self-contained; runs the real code path)
  core/kernels/                # ←★ EDIT SURFACE: optimize these Triton kernels
      wave_step.py             #    primary fixed-point update over a band of rows
      dts_fused.py             #    per-row reduction over child contributions
      e_step.py                #    auxiliary (E-step) fixed point: forward + backward kernels
      wave_backward.py         #    backward adjoint solve + parameter VJP
      _dts_layout_contract.py
  core/inference/              # FROZEN glue (forward orchestration, loss)
  core/parameters/             # FROZEN (parameter extraction)
  core/memory_policy.py        # FROZEN
  api/_implicit_grad.py        # FROZEN (backward orchestration)
  api/solver_options.py        # FROZEN
  runtime.py                   # FROZEN driver: run_forward / run_backward from captured tensors

fixtures/<size>.pt             # COMMITTED problem definitions: static data + input parameters
capture/capture.py             # run ONCE on pristine kernels → computes golden into data/ (no ext deps)
bench/
  forward.py                   # whole forward: time + check vs golden
  backward.py                  # whole backward: time + check vs golden
  kernels.py                   # per-kernel micro-bench: replay one wrapper call in isolation
  check.py                     # ★ correctness GATE across forward+backward+kernels
  _common.py                   # timing (CUDA events) + golden comparison
data/<size>/                   # golden + per-kernel snapshots (git-ignored, rebuilt from fixtures)
```

**Only `kbench/core/kernels/*.py` should be edited.** Everything else is frozen glue. The whole
`kbench/` package is a vendored, domain-neutralized copy of an upstream solver — identifiers renamed
to neutral terms, comments stripped — but it runs the real code path unchanged.

---

## Workflow for the optimizer

```bash
# 0. ONE TIME, on the PRISTINE kernels: compute the golden baseline from the committed fixtures.
python capture/capture.py                       # all sizes; per-kernel snapshots for `small`

# 1. edit a kernel in kbench/core/kernels/

# 2. CORRECTNESS GATE — must pass before any speed number counts
python bench/check.py                           # exits nonzero on any drift beyond tolerance

# 3. measure the pass(es) affected
python bench/forward.py                          # theta → loss
python bench/backward.py                         # intermediates → grads
python bench/kernels.py --kernel compute_wave_step    # one kernel in isolation
```

No install is required — the scripts put the repo root on `sys.path` so `import kbench` works
directly. (`pip install -e .` also works.) Each bench prints `median / min / p90` ms (CUDA-event
timed, warmed up) and a PASS/FAIL line per compared tensor. `--check-only` skips timing; `--size
<label>` restricts to one size.

> **`capture/` snapshots the *current* kernels as the golden.** Run it on the pristine kernels to set
> the baseline; while optimizing, run `bench/` (not `capture/`). Re-running `capture/` after an edit
> would move the baseline to the edited output.

---

## Golden reference — what "correct" means

`capture/capture.py` loads each committed fixture (the static problem data + input parameters) and
computes the golden by running the **pristine vendored kernels** on it, then freezes, per size:

- **forward golden**: the scalar `loss`, plus intermediates (`pi_wave`, `E`, `pibar_row_max`, …).
- **backward golden**: `grad_theta`, `grad_col`, computed from the golden forward intermediates — so
  the backward bench feeds those *fixed* intermediates in and is therefore measured/checked
  independently of the forward.
- **per-kernel golden**: for each captured wrapper call, its return value **and** any tensors it
  mutates in place. The micro-bench rebuilds the inputs and compares the post-call state. Three
  subtleties are handled at capture time:
    - **argument aliasing** is preserved — some calls pass the same buffer as two arguments and mutate
      through both; inputs are deduped by storage and re-shared on replay.
    - **uninitialized outputs** are excluded — these kernels `torch.empty` their outputs and only write
      the *active* rows (an `active_mask` prunes the rest). Capture runs each kernel twice with
      `torch.empty` pre-filled with a different sentinel each run, so any position the kernel never
      writes diverges and is flagged don't-care (reliable at any tensor size).
    - **atomic nondeterminism** is tolerated — kernels that accumulate with atomics differ run-to-run;
      capture records that noise floor per tensor and the bench widens tolerance to `max(tol, 4 × noise)`.

A rewritten kernel is **correct** iff every compared tensor is within tolerance of golden
(`max_abs ≤ atol` **or** `max_rel ≤ rtol`; default `2e-3`), at valid positions, within the noise floor.
NaN/Inf mismatches always fail. These kernels are float32, so exact equality is not expected — match
the reference, don't chase bit-identity. The whole-pass forward/backward gates are the
**authoritative** correctness signal; the per-kernel checks are a tighter localized aid.

---

## Sizes

Captured at several scales so an optimization that only helps one regime is visible. The whole-pass
unit is **one batch**, whose row count `C` is fixed by the fixture (`S = 119` columns throughout).

| label  | C (rows) | fwd / bwd median |
|--------|---------:|------------------|
| small  | ~225k    | 38 ms / 84 ms    |
| medium | ~310k    | 38 ms / 97 ms    |
| large  | ~1.50M   | 92 ms / 273 ms   |

Per-kernel snapshots are captured only at `small` by default (`--kernels-only-on`) to bound disk
(~2.7 GB; whole-pass goldens are ~310 MB small/medium, ~1.5 GB large — all under git-ignored `data/`).
Backward is ~3× forward — the implicit adjoint solve is the headline target.

## Provenance

`kbench/` was vendored once from an upstream solver and domain-neutralized (renamed identifiers,
stripped comments); it is now a standalone, hand-owned copy with no live link to the source. The
`fixtures/<size>.pt` are frozen problem definitions derived once from that solver's preprocessing of a
dataset; they are the only externally-derived artifact and are treated as committed data.
