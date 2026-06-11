# Benchmarks & code you can iterate on

This document explains, in detail, **what you can run**, **what you can edit**, and **how the
results are judged**. It is the operating manual for optimizing the Triton kernels in this repo. For a
one-page overview see [../README.md](../README.md); for the diagnostic story of how the harness was
made trustworthy see [FIXES.md](FIXES.md).

The project is **self-contained**: it depends on nothing but `torch` + `triton`. The solver
(`kbench/`) is a vendored, domain-neutralized copy; the problem definitions are committed fixtures;
the golden reference is computed from the pristine vendored kernels themselves.

---

## 1. The two elementary operations

| pass | computes | inputs | output |
|------|----------|--------|--------|
| **forward**  | auxiliary fixed point (E-step) → primary fixed point (the "wave" sweep) → scalar loss | `theta`, `col_weights` (+ a frozen `static`: topology, wave layout, solver options) | scalar `loss` (+ intermediates the backward needs) |
| **backward** | implicit-differentiation adjoint solve + parameter VJP | the forward intermediates | `grad_theta`, `grad_col` |

The state is a band-structured matrix of `C` rows × `S` columns, solved in log-space by a fixed-point
iteration. A **wave** is a band of rows (a topological layer of the row dependency graph) updated
together in one sweep; the solver runs `pi_iters` sweeps per wave.

The two passes are benchmarked **separately**: the forward bench is `theta → loss`; the backward bench
feeds in the *golden* forward intermediates and measures `→ grads`, so a backward change is timed and
checked independently of the forward.

You optimize at two granularities, both provided here:

- **Per-kernel** — one Triton wrapper call replayed in isolation (tight signal, ~0.02–0.3 ms each).
- **Whole-pass** — the entire forward or backward (the aggregate signal that catches interactions).

---

## 2. Quickstart

```bash
cd kernel-bench

# 0. ONE TIME, on the PRISTINE kernels: compute the golden baseline from the committed fixtures.
python capture/capture.py                                # all sizes; per-kernel snapshots for `small`

# 1. edit a kernel in kbench/core/kernels/

# 2. CORRECTNESS GATE — must pass before any speed number means anything
python bench/check.py

# 3. measure what you changed
python bench/forward.py                                  # whole forward, all sizes
python bench/backward.py                                 # whole backward, all sizes
python bench/kernels.py --kernel compute_wave_step       # one kernel in isolation
```

No install or external dependency is required — the scripts put the repo root on `sys.path` so
`import kbench` works directly. (`pip install -e .` also works.)

---

## 3. The edit surface — `kbench/core/kernels/`

**These are the only files you should change.** The golden reference was captured from them, so a
correct rewrite must reproduce their numerics.

### Forward kernels

| file | wrapper(s) | what it computes | called from |
|------|-----------|------------------|-------------|
| [wave_step.py](../kbench/core/kernels/wave_step.py) | `compute_wave_step`, `compute_leaf_initial_wave_step` | one fixed-point iteration of the primary update for a band of rows (the `leaf_initial` variant seeds the first wave). This is the **hot forward kernel** — it runs `pi_iters` times per wave, for every wave. | `core/inference/forward.py` |
| [dts_fused.py](../kbench/core/kernels/dts_fused.py) | `compute_dts_forward` | the per-row reduction over child contributions (a closed-form path for a single child pair, a two-stage tiled reduction for fan-out ≥ 2). | `core/inference/forward.py`, and again inside the backward to recompute its intermediate |
| [e_step.py](../kbench/core/kernels/e_step.py) (forward part) | `e_fixed_point_triton`, `_launch_e_step_forward_2d` | the auxiliary (E-step) fixed point over the `S`-column state, iterated to `e_tol`. Runs once per forward, before the wave sweep. | `core/inference/solver.py` |

### Backward kernels

| file | wrapper(s) | what it computes | called from |
|------|-----------|------------------|-------------|
| [wave_backward.py](../kbench/core/kernels/wave_backward.py) | `wave_backward_uniform_fused` | the **hot backward kernel**: per wave, solves the self-loop adjoint (Neumann series or GMRES) and accumulates parameter-gradient contributions. Reverse wave order. | `api/_implicit_grad.py` |
| | `dts_cross_backward_accum_fused` | backward of the per-row reduction — gradients w.r.t. the child state and the rate parameters. | `api/_implicit_grad.py` |
| | `uniform_cross_pibar_vjp_tree_from_ud_fused` | the cross-row VJP via a bottom-up reduction over the topology. | `api/_implicit_grad.py` |
| | `active_mask_from_rhs_absmax_fused` | builds the per-wave active-row mask used to prune the adjoint. | `api/_implicit_grad.py` |
| [e_step.py](../kbench/core/kernels/e_step.py) (backward part) | `e_step_triton_autograd` (`_TritonEStep2D`) | the differentiable E-step used by the E-adjoint BiCGSTAB solve. | `api/_implicit_grad.py` |

`_dts_layout_contract.py` is a small layout helper for the `dts` kernels — you will rarely touch it.

> **Backward dominates.** On the reference fixtures the backward is ~3× the forward (e.g. large:
> 273 ms vs 92 ms). The implicit adjoint solve in `wave_backward_uniform_fused` is the biggest single
> lever; `neumann_terms` (16 in the fixtures) sets how many series terms it runs.

> **Note on names.** The symbol names (`compute_wave_step`, `_launch_e_step_forward_2d`, the argument
> names, the data-dict keys) are domain-neutral but otherwise arbitrary — treat them as opaque handles
> to the operation each performs.

### What you must preserve

The orchestration calls these wrappers with **exact positional/keyword contracts** and relies on:
- writing results **in place** into pre-allocated buffers (the output and accumulator tensors);
- **argument aliasing** — some calls pass the same buffer as two arguments and mutate through both;
- only **active rows** being meaningful in masked outputs (inactive rows may stay uninitialized).

You are free to change block sizes, grid shapes, tiling, fusion, dtypes of intermediates, autotune
configs, the math's associativity — anything that keeps the outputs within tolerance.

---

## 4. What is frozen (do not edit)

Everything outside `core/kernels/` is **frozen glue** — the orchestration that drives the kernels:

- `core/inference/forward.py` — the wave loop that schedules the wave kernels.
- `core/inference/solver.py` — the forward driver (E-step + wave sweep) and the loss.
- `core/inference/logspace.py`, `core/parameters/extract_parameters.py`, `core/memory_policy.py`.
- `api/_implicit_grad.py` — the backward orchestration (adjoint loop, E-adjoint BiCGSTAB).
- `api/solver_options.py`, `runtime.py`.

Changing a kernel's *interface* means also changing the glue that calls it; prefer interface-preserving
changes so the contract with the captured data and the bench stays intact.

---

## 5. The benchmarks in detail

All four scripts share flags: `--rtol`/`--atol` (tolerance, default `2e-3`), `--size <label>`
(restrict to one size), `--check-only` (skip timing), `--warmup`/`--iters` (timing loop).
Timing uses CUDA events after warmup and reports `median / min / p90` ms.

### `bench/forward.py` — whole forward pass

Rebuilds the `static` from a capture, runs `run_forward(theta) → (loss, intermediates)`, and checks
`loss` (the primary signal) plus `pi_wave`, `E`, `pibar_row_max` (intermediates — stronger
localization of where a forward kernel drifted).

```
=== forward [large]  S=119 C=1499645 items=128 ===
    [PASS] loss            max_abs=... max_rel=...
    time: median=92.4 ms  min=91.3 ms  p90=93.5 ms  (n=30)
```

Run after editing `wave_step.py`, `dts_fused.py`, or the **forward** part of `e_step.py`.

### `bench/backward.py` — whole backward pass

Feeds the **golden** forward intermediates in as fixed input, runs
`run_backward(...) → (grad_theta, grad_col)`, and checks both gradients. Because the inputs are the
frozen golden forward, this isolates the backward. Run after editing `wave_backward.py` or the
**backward** part of `e_step.py`.

### `bench/kernels.py` — per-kernel micro-bench

Replays a single captured wrapper call: rebuilds its inputs on device (with aliasing preserved), runs
the vendored wrapper, checks the post-call state (return value **and** any tensors mutated in place)
against golden, then times it.

```
python bench/kernels.py                          # every captured kernel, all sizes with captures
python bench/kernels.py --kernel compute_wave_step
python bench/kernels.py --phase backward         # only backward kernels
python bench/kernels.py --phase forward --check-only
```

Each call is labelled with an `int_sig` (its integer args — `ws, W, S, …`) and a tag like
`(noise:1 masked:3 tensors)` indicating how many output tensors carry an atomic-noise floor and how
many have don't-care (uninitialized) regions. See §7.

### `bench/check.py` — the correctness gate

Runs forward + backward + every captured kernel in **check-only** mode across all sizes, prints a
summary, and **exits non-zero** if anything drifts beyond tolerance. This is the gate the optimizer
must pass before a timing number is trustworthy.

```
================ SUMMARY ================
  forward    PASS
  backward   PASS
  kernels    PASS
  ALL PASS
```

---

## 6. Capturing the golden (`capture/capture.py`)

The golden is **not** fetched from any external library — it is computed by running the **pristine
vendored kernels** on the committed fixtures. `capture/capture.py` imports only `torch` + `kbench`.

For each size it loads `fixtures/<label>.pt` (the static problem data + input parameters), runs the
vendored forward and backward, and writes `data/<label>/`:

- `whole.pt` — meta + static + inputs (from the fixture) + the forward intermediates + the golden
  `loss` / `grad_theta` / `grad_col`.
- `kernels/_pool.pt` + `kernels/<wrapper>__NNN.pt` — per-kernel snapshots (a shared content-addressed
  buffer pool + tiny per-call ref files), captured by monkeypatching the `kbench` wrappers during the
  vendored forward/backward. Only written for the `--kernels-only-on` label (default `small`).

Flags:

| flag | meaning |
|------|---------|
| `--sizes small,medium,large` | which fixture labels to snapshot (default: all in `fixtures/`) |
| `--no-kernels` / `--kernels-only-on LABEL` | control per-kernel snapshotting |
| `--per-wrapper K` | distinct waves captured per kernel (default 4) |

Run it **on the pristine kernels** to establish the baseline; re-running after an edit would move the
baseline to the edited output. (The solver settings — `pi_iters`, `neumann_terms`, etc. — are baked
into each fixture's `meta`.)

### Sizes (reference)

`C` = rows in the captured batch; `S = 119` columns throughout.

| label  | C | fwd / bwd median | data |
|--------|----|------------------|------|
| small  | ~225k  | 38 ms / 84 ms  | ~2.7 GB (incl. kernels) |
| medium | ~310k  | 38 ms / 97 ms  | ~310 MB |
| large  | ~1.50M | 92 ms / 273 ms | ~1.5 GB |

`fixtures/` (~160 MB) is committed; `data/` is git-ignored and rebuilt by `capture/`.

---

## 7. How correctness is judged

A rewritten kernel is **correct** iff every compared tensor matches golden within tolerance at the
meaningful positions:

- **Tolerance**: pass if `max_abs ≤ atol` **or** `max_rel ≤ rtol` (default `2e-3`). These kernels are
  float32 — match the reference, don't chase bit-identity.
- **inf/NaN-safe**: the log-space tensors hold `-inf`; positions where got == ref exactly (including
  matching `±inf`) and positions where both are NaN are agreements.
- **don't-care positions** (per-kernel only): some kernels `torch.empty` their outputs and write only
  *active* rows. Those uninitialized positions are detected at capture (by sentinel-filling
  `torch.empty` across two runs) and **excluded** from the comparison.
- **noise floor** (per-kernel only): kernels that accumulate with atomics differ run-to-run; the
  captured per-tensor noise widens the effective tolerance to `max(tol, 4 × noise)`.

**The whole-pass forward/backward gates are the authoritative signal** — fully initialized,
deterministic-to-tolerance. The per-kernel checks are a tighter localized aid; trust them, but if a
per-kernel check and the whole-pass disagree, believe the whole-pass.

---

## 8. Recommended iteration loop

1. Pick a target. Backward (`wave_backward_uniform_fused`) is the biggest lever; in the forward,
   `compute_wave_step` runs most often.
2. Edit the kernel in `kbench/core/kernels/`.
3. `python bench/check.py` — if it fails, you broke correctness; fix before looking at speed.
4. `python bench/forward.py` / `bench/backward.py` for the aggregate effect; `bench/kernels.py
   --kernel <name>` to confirm the local win.
5. Compare median ms against §6 (or your own baseline run before the edit). Keep the change only if it
   is faster and still PASS; otherwise revert.

---

## 9. Troubleshooting

- **`no fixture at fixtures/<label>.pt`** — pass a `--sizes` label that exists in `fixtures/`.
- **`no captures found in data/`** — run `capture/capture.py` first.
- **`check.py` fails right after a fresh `capture/`** — you captured on already-edited kernels; the
  golden tracked the edit. Restore the pristine kernels and re-capture.
- **A per-kernel check fails but whole-pass passes** — likely a benign difference in a don't-care
  region the harness didn't mask, or atomic noise above the recorded floor; trust the whole-pass, and
  consider raising `--rtol/--atol` for that kernel.
- **OOM at `large`** — use `--size medium` (or `small`).
