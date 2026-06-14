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
- Device: **A100-SXM4-80GB**, capability (8,0), ~85 GB usable, full-rate fp64. (vs local 24 GB.)

## Python / torch environment

- **micromamba env**, Python **3.11.15**, **torch 2.12.0+cu130 (CUDA 13.0)**. torch resolves from
  `~/.local/lib/python3.11/site-packages` (user site shadows the env — set `PYTHONNOUSERSITE=1` if
  you need the env's own torch).
- Working interpreter: `/work/SzollosiU/enzo-marsot/gpurec-<jobid>/env/bin/python` (a micromamba env
  staged into `/work`; the prior `gpurec-4630915/` work dir has `env/ micromamba/ bin/ gpurec/`).
- ⚠ **Login nodes have old GLIBC (<2.28): `import torch` FAILS there** (`GLIBC_2.28 not found`). torch
  works only on **compute nodes** (newer OS). Always run torch via `srun` on a compute node.
- `/bucket/.../venvs/gpurec/bin/python` is py3.11 but its `libpython3.11.so` isn't loadable on login —
  use the micromamba env in `/work` instead.
- System `module avail` works but is old (python ≤3.7, cuda ≤11.3) — NOT used; the env self-contains
  CUDA 13 via the pip torch wheel.
- There is **no system `python3.11` and no conda/micromamba on PATH** (only `~/.local` has
  torch+triton, not numpy/scipy). So a plain venv has no base interpreter — use micromamba.

### Creating the env (one command)

`scripts/setup_cluster_env.sh` (in this repo) does it reproducibly: installs micromamba (static
binary, no root) → creates a python 3.11 env → pip-installs torch (cu130) + `requirements.txt`
(triton/numpy/scipy) + the package. Run it on the **saion login node** (pip downloads there):

```bash
cd /work/SzollosiU/enzo-marsot/<workdir>/kernel-bench   # or the /bucket clone
bash scripts/setup_cluster_env.sh [ENV_PREFIX]          # default /work/.../kbench-env (scratch)
# pass a /bucket path for a PERSISTENT env to reuse every session (avoids "venv every time")
```

Then **verify on a compute node** (login fails torch import — GLIBC):
```bash
srun -p gpu-a100 -c 4 --mem=16G --gres=gpu:1 --time=00:05:00 \
  <ENV_PREFIX>/bin/python -c 'import torch,triton,numpy,scipy; print(torch.cuda.is_available())'
```
Deps are declared in `requirements.txt` (torch, triton, numpy, scipy).

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
