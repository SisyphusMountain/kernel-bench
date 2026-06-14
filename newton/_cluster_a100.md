# OIST cluster (deigo + saion) — A100 runbook

Verified 2026-06-14. Motivation: the local consumer GPU has weak fp64 + 24 GB; the cluster A100s
have full fp64 + 80 GB, so they're the place to test whether the ill-conditioned-CG ‖g‖ floor is an
fp32-rounding artifact (see `_optimize_findings.md` "conditioning wall").

## Two clusters

- **deigo** = storage cluster. `ssh deigo-ext` → `deigo-login3.oist.jp`. `/bucket` is read-write here.
- **saion** = compute cluster (the GPUs). `ssh saion-ext` → `saion-login1.oist.jp`.
- Both use key-based auth (BatchMode works). Transient `Connection reset by peer` /
  `kex_exchange_identification` happens — just retry.
- Direct **saion→deigo ssh is NOT configured** (host-key verification fails). Don't rely on it.

## Filesystems

| path | deigo | saion login | saion **compute (A100)** | notes |
|---|---|---|---|---|
| `/bucket/SzollosiU/enzo-marsot` | rw | **rw** | **read-only** | persistent storage; has `git/`, `venvs/` |
| `/work/SzollosiU/enzo-marsot` | not visible | rw | rw | saion scratch; temp work/output; has `pip-cache/`, prior `gpurec-<jobid>/` dirs |

- `/bucket` is the persistence target. It is read-only on the **compute** node, so jobs write to
  `/work` and you copy `/work → /bucket` **from the saion login node** (where `/bucket` is rw).
- The cluster repo is **`gpurec`** (Rust `crates/` + python `gpurec` package) — a DIFFERENT codebase
  from the local `kernel-bench`. Recent commits include "Add MAP tutorial and optimization
  diagnostics" (the MAP/Newton work has a counterpart there).

## Getting an A100

```bash
ssh saion-ext
srun -p gpu-a100 -c 16 --mem=128G --gres=gpu:1 --time=HH:MM:SS --pty bash -l   # interactive
# non-interactive (scriptable): replace --pty bash -l with the command, e.g.
srun -p gpu-a100 -c 16 --mem=128G --gres=gpu:1 --time=00:10:00 bash -lc 'hostname; nvidia-smi'
```

- `gpu-a100`: 4 nodes `saion-gpu23..26`, **8× A100-SXM4-80GB each**. Often mostly idle (gpu25/26 were
  0/128) → allocation is instant. `largegpu` partition also has `a100:8`. Other partitions:
  `gpu-v100` (v100:4), `gpu-p100`/`test-gpu` (p100). Check load: `sinfo -p gpu-a100 -o "%n %t %G %C"`,
  `squeue -p gpu-a100`.
- ⚠ **GROUP QUOTA caps concurrency at ONE GPU job** (verified 2026-06-15): the `allstudents` assoc
  GrpTRES is `cpu=16, gres/gpu=1, mem=128G` (`sacctmgr -n -P show assoc user=$USER format=GrpTRES`). So
  a *second* concurrent GPU job **never** starts (reason `AssocGrpGpuLimit`/`AssocGrpMemLimit`) no matter
  how you size `-c`/`--mem`. Run jobs **serially**; don't bother trying two A100s in parallel. fp64 HVP
  on the A100 is ~7 s each (m=250 Lanczos ≈ 30 min) — slower per-call than expected, but 80 GB fits the
  fp64 666x80 HVP that OOMs locally.
- Device: **A100-SXM4-80GB**, capability (8,0), ~85 GB usable, full-rate fp64. (vs local 24 GB.)

## Python / torch environment

- **glibc / wheels**: **login node = glibc 2.17** (RHEL7, GCC 4.8.5), **compute (A100) = glibc 2.28**
  (RHEL8), compute has internet. glibc 2.28 = the `manylinux_2_28` wheel baseline, and the LATEST
  torch 2.12 / numpy 2.4.6 / scipy 1.17.1 / triton 3.7.0 all ship `manylinux_2_28` wheels — so
  **building the env on a compute node installs the latest of everything, no pins, no source build.**
  Building on the LOGIN node forces a numpy source build that fails (GCC 4.8.5 < 9.3). Can't upgrade
  system glibc (no root); 2.28 is the current frontier (nothing on `manylinux_2_34` yet). No
  Apptainer/Singularity.
- ⚠ **`import torch` FAILS on login nodes** (`GLIBC_2.28 not found`); works on compute nodes. Always
  run torch via `srun`.
- micromamba env, Python **3.11**. There is **no system `python3.11` and no conda/micromamba on
  PATH** — micromamba (static binary, no root) provides the interpreter. System `module avail` is old
  (python ≤3.7, cuda ≤11.3), unused.

### Creating the env (one command)

`scripts/setup_cluster_env.sh` does it reproducibly. **Run it from the saion login node** — it
**re-execs itself onto a gpu-a100 compute node** (glibc 2.28) so the latest wheels install, and builds
**self-contained** (`PYTHONNOUSERSITE=1` so torch installs fresh, not from `~/.local`):

```bash
cd <repo>                                       # the /bucket clone or a /work copy
bash scripts/setup_cluster_env.sh [ENV_PREFIX]  # default /work/.../kbench-env; pass a /bucket path
                                                # for a PERSISTENT env (avoids "rebuild every time")
```

It installs micromamba → python 3.11 → torch (cu130) + `requirements.txt` (triton/numpy/scipy, latest)
+ the package, then verifies torch.cuda on the node. Run the env's python with `PYTHONNOUSERSITE=1`
to keep it isolated from `~/.local`.

## Per-experiment workflow

1. **Stage** (saion login): make a work dir under `/work/SzollosiU/enzo-marsot/<name>`; copy the repo
   from `/bucket` (ro source is fine) and reuse/copy a micromamba `env/`. Or `scp` code from local:
   `scp -r local_dir saion-ext:/work/SzollosiU/enzo-marsot/<name>/`.
2. **Run** (A100): `srun -p gpu-a100 -c 16 --mem=128G --gres=gpu:1 --time=... \
   /work/.../env/bin/python script.py` — read inputs from `/work` (or `/bucket` ro), write **outputs
   to `/work`** (compute node can't write `/bucket`).
3. **Persist** (saion login, `/bucket` is rw there): `cp -r /work/.../<name>/results \
   /bucket/SzollosiU/enzo-marsot/<dest>/`. (This is the "scp back to deigo" — `/bucket` IS deigo
   storage mounted rw on the saion login node.) Alt: pull to local then push to deigo:
   `scp -r saion-ext:/work/.../results .` then `scp -r results deigo-ext:/bucket/.../`.
4. **Clean**: `rm -rf /work/SzollosiU/enzo-marsot/<name>`.

## Quick verification commands (all passed 2026-06-14)

```bash
ssh deigo-ext 'ls /bucket/SzollosiU/enzo-marsot/'                      # git venvs
ssh saion-ext 'sinfo -p gpu-a100 -o "%n %t %G %C"'                     # node availability
scp file saion-ext:/work/SzollosiU/enzo-marsot/                       # stage to scratch
ssh saion-ext 'srun -p gpu-a100 -c 16 --mem=128G --gres=gpu:1 --time=00:10:00 \
  /work/.../env/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"'
```
