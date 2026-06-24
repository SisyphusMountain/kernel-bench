# gpurec vs AleRax — genewise DTL-rate speed comparison

Head-to-head wall-clock of **gpurec** (GPU) and **AleRax v1.4.0** (CPU) fitting per-family `UndatedDTL`
rates on the **same** inputs (same species tree, same gene-tree distributions, `--model-parametrization
PER-FAMILY` = gpurec "genewise"). Two datasets:

- **Archaea** (§5.1): 3,946 families covering ≥4 species (of 5,446).
- **Hogenom** (throughput): the ≥4-species set at three scales (the set AleRax's `UndatedDTL` retains).

> The genewise recipe is now the **gpurec library default**, `gpurec.fit_genewise(...)`. Everything here
> (`reconcile.py`, `run_gpurec_traced.py`) is a thin wrapper over it — one source of truth.

## Results

**Archaea (≥4 species, 3,946 families):**

| tool | hardware | wall-clock (fit) | NLL |
|---|---|---|---|
| AleRax v1.4.0 | 24 CPU cores (i9-13900K, MPI) | **1526 s** | −226,334 nats |
| gpurec genewise | 1× RTX 4090 | **~120 s** | 225,865 nats (= bits×ln2) |

~12× faster, same optimum (gpurec −469 nats). NLLs differ only by bits-vs-nats (`gpurec_bits×ln2 = nats`).

**Hogenom (≥4 species, gpurec, traced recipe, single RTX 4090):**

| subset | #families (≥4 sp) | fit | converged | NLL (bits) | throughput |
|---|---|---|---|---|---|
| Hogenom-512 | 506 | 119 s | 500/506 | 324,262 | 4.3 fam/s |
| Hogenom-1055 | 1,042 | 219 s | 1,037/1,042 | 577,593 | 4.8 fam/s |
| Hogenom-full | 10,869 | 758 s | 10,853/10,869 | 1,906,464 | 14.3 fam/s |

`fit` = optimize wall-clock (AleRax-comparable, no PD cert); converged/NLL are from an optional cold PD
certificate. The AleRax CPU baseline on the identical 10,869-family set is run via
`run_alerax_hogenom_full.sh` (see Reproduce). Throughput rises with scale as batching saturates the GPU.

The full optimization story (every A/B-tested lever) is in [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md).

## Prerequisites

- **GPU** (RTX 4090) for the gpurec side; **AleRax v1.4.0** on `PATH` + `mpiexec` for the AleRax side.
- The gpurec **worktree** (library + `run_cv.py` dataset registry + the Rust preprocess `.so` at
  `crates/gpurec-preprocess/target/release/libgpurec_preprocess.so`). Override paths via env:
  `GPUREC_WT`, `PYTHON`, `PREPROCESS_SO`.
- Data ships with the gpurec repo under `gpurec/tests/data/` (archaea + Hogenom-Core).

## Reproduce

```bash
# gpurec side (GPU, minutes): builds the >=4-species family files, then warmup+warm times all 3 sizes.
./reproduce.sh

# AleRax CPU baseline (>=4 species, full 10,869 families) — LONG (hours). Launch DETACHED so it
# survives the shell (a closed session otherwise kills it mid-run):
nohup NP=24 ./run_alerax_hogenom_full.sh > run_alerax_hogenom_full.log 2>&1 &
```

`make_hogenom_full_families.py` deterministically builds each `[FAMILIES]` file (species = leaf prefix
before the first `_`, the same rule gpurec's Rust preprocess and AleRax both use), so both tools fit the
identical family set. `run_gpurec_hogenom_subsets.sh` runs each size **twice** (first warms the Triton
kernel cache; the second is the reported warm number).

For the **archaea** §5.1 comparison: `./run_alerax.sh` (CPU) and `./run_gpurec.sh` twice (GPU).

## Use gpurec on your own data (coming from AleRax)

`reconcile.py` is a dataset-agnostic CLI over `gpurec.fit_genewise`. Inputs are the same ones AleRax uses
(rooted newick species tree; one `.ale` CCP **or** a newick gene-tree distribution per family, leaves
`<Species>_<id>`):

```bash
python reconcile.py --species-tree tree.newick --gene-trees 'ale_dir/*.ale' --out rates.tsv [--certify]
```

Output `rates.tsv` = `family⇥Dup⇥Loss⇥Trans`, per-family rates relative to speciation (= 2^θ), exactly
AleRax's `model_parameters/<fam>_rates.txt`. Verified on `fix_100`: D=0.065 L=0.306 T=0.274 vs AleRax
0.0651/0.3048/0.2736. `--max-rate` is the box bound (a family that runs to it is non-identifiable);
`--fp64` for tight tolerances. Or call the library directly:

```python
import gpurec
res = gpurec.fit_genewise("tree.newick", "ale_dir/*.ale", certify=True)
res["rates"]      # [F,3] D,L,T relative to speciation;  res["loss_bits"], res["converged"], ...
```

## The recipe (library default `gpurec.fit_genewise`)

Adam warm-up (5 steps, lr=1, grad-clip-norm 10) → box-constrained trust-region Newton on the per-family
3×3 **forward-difference** Hessian (eigen-clamped → PD) → **convergence-based rebatching** (drop families
at |Pg|<1e-3, rebuild over survivors) + **pi-tier escalation** (16→64 for forward-stiff families) +
**warm-adjoint** at neu=16, **memory-gated** automatically (the cache scales as clades×species×dtype; the
gate runs large batches cold and re-enables warm as the active set shrinks — keeps full Hogenom on 24 GB).

`run_gpurec_traced.py` env knobs (all map to `fit_genewise` params): `DATASET` `FAMILIES` `FAMFILE`
`CLADE_BUDGET` `CERT` `PIS` `NEU_OPT` `NEU_CERT` `ADAM` `ADAM_LR` `GRAD_CLIP` `TOL` `MAXIT`.

## Files

| file | role |
|---|---|
| `reproduce.sh` | **single entry point** — gpurec Hogenom throughput (3 sizes) + the AleRax command |
| `reconcile.py` | dataset-agnostic genewise CLI over `gpurec.fit_genewise` (use on your own data) |
| `run_gpurec_hogenom_subsets.sh` | gpurec warmup+warm timing on the 3 Hogenom ≥4-species subsets |
| `make_hogenom_full_families.py` | deterministic ≥4-species `[FAMILIES]` builder (full / subset) |
| `run_alerax_hogenom_full.sh` | AleRax CPU baseline on the Hogenom ≥4-species full set (`NP=` cores) |
| `run_gpurec_traced.py` | thin benchmark wrapper over `gpurec.fit_genewise` (env knobs → params) |
| `make_families.py` / `run_alerax.sh` / `run_gpurec.sh` | the archaea §5.1 comparison (builder + runners) |
| `OPTIMIZATION_PLAN.md` | the full optimization investigation: every lever, A/B-measured |
| `run_gpurec_lbfgs.py` / `run_gpurec_rebatch.py` | rejected L-BFGS-B endgame / pre-optimization reference |
