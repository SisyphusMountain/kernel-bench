"""Profile whether the genewise family-batches / waves should be LARGER.

The production build uses family_chunk_size=300, clade_budget=80000, max_wave_size=8192, so the
3946 archaea families are split into ~14 GPU batches of <=300 families. Memory headroom is large
(~4/24 GB), so we CAN go bigger -- the question is whether per-eval throughput improves or we are
already saturated. For each config: build, report the batch/wave structure (n batches, families &
clades per batch, waves), then time N warm loss+grad evals over the full 3946 + peak memory. Bigger
helps iff per-eval time drops. A background nvidia-smi sampler records GPU util per eval window.
"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions

DEV = "cuda"; DT = torch.float32
FAMFILE = os.environ.get("FAMFILE", os.path.join(HERE, "archaea_ge4sp.families.txt"))
paths = [ln.split("=", 1)[1].strip() for ln in open(FAMFILE) if ln.startswith("starting_gene_tree")]
TREE = str(DATASETS["archaea"]["species_tree"])
N_EVAL = int(os.environ.get("N_EVAL", "20"))
print(f"[profile] {len(paths)} families  N_EVAL={N_EVAL}  (pi16/neu16, warm-off cold evals)", flush=True)


def sopts(pi=16, neu=16): return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": neu})


def static_clades(st):
    for a in ("wave_layout", "clade_parent", "num_clades", "n_clades"):
        v = getattr(st, a, None)
        if v is None: continue
        if torch.is_tensor(v): return int(v.shape[0])
        if isinstance(v, int): return v
    return -1


def static_waves(st):
    wl = getattr(st, "wave_layout", None)
    if wl is None: return -1
    # wave_layout commonly encodes wave boundaries/offsets; report its length as a proxy
    try:
        if torch.is_tensor(wl): return int(wl.max().item()) + 1 if wl.numel() else 0
    except Exception:
        pass
    return -1


def run_cfg(tag, chunk, budget, wave):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tb = time.perf_counter()
    try:
        m = GeneReconModel(TREE, [str(x) for x in paths], mode="genewise", device=DEV,
                           solver_options=sopts(), family_chunk_size=chunk,
                           clade_budget=budget, max_wave_size=wave)
    except RuntimeError as e:
        print(f"\n=== {tag}: chunk={chunk} budget={budget} wave={wave}  -> BUILD FAILED: {str(e)[:80]}", flush=True)
        torch.cuda.empty_cache(); return
    m.receiver_weights.requires_grad_(False)
    torch.cuda.synchronize(); build_s = time.perf_counter() - tb
    nb = len(m.batch_statics)
    fpb = [len(b) for b in m.family_batches]
    cpb = [static_clades(st) for st in m.batch_statics]
    if tag == "A":  # discover attrs once
        st0 = m.batch_statics[0]
        print(f"  [introspect static0 attrs] {[a for a in dir(st0) if not a.startswith('_')]}", flush=True)
    print(f"\n=== {tag}: chunk={chunk} budget={budget} wave={wave} ===", flush=True)
    print(f"  build={build_s:.1f}s  n_batches={nb}  families/batch: min={min(fpb)} max={max(fpb)} "
          f"mean={sum(fpb)/nb:.0f}", flush=True)
    print(f"  clades/batch: min={min(cpb)} max={max(cpb)} total={sum(cpb)}  "
          f"waves/batch(max-id): {[static_waves(st) for st in m.batch_statics[:4]]}...", flush=True)

    th = torch.zeros(len(paths), 3, device=DEV, dtype=DT)
    def lg():
        lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); return lv, g
    for _ in range(2): lg()
    torch.cuda.synchronize()
    ws = time.time(); t = time.perf_counter()
    for _ in range(N_EVAL): lg()
    torch.cuda.synchronize(); dt = time.perf_counter() - t; we = time.time()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [WINDOW {tag}] eval_start={ws:.3f} eval_end={we:.3f}", flush=True)
    print(f"  >> per-eval = {dt/N_EVAL*1000:.0f} ms   ({N_EVAL} evals = {dt:.1f}s)   peak_mem={peak:.2f} GB", flush=True)
    del m; torch.cuda.empty_cache()


# current production, then progressively larger batches, then larger waves
run_cfg("A", 300,   80_000,   8192)    # production
run_cfg("B", 1000,  250_000,  8192)
run_cfg("C", 4000,  900_000,  8192)    # aim: all 3946 in one batch
run_cfg("D", 4000,  900_000,  32768)   # one batch + 4x larger waves
print("\n[done]", flush=True)
