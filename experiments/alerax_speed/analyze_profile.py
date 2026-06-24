"""Analyze /tmp/profile_steps.json: coarse per-step time distribution + fine convergence-waste."""
import json, sys
J = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/profile_steps.json"))
EV = J["events"]; M = J["meta"]
newton = [e for e in EV if e["phase"] == "newton"]
rebuild = [e for e in EV if e["phase"] == "rebuild"]
resid = [e for e in EV if e["phase"] == "resid"]
build = [e for e in EV if e["phase"] == "build"]
adam = [e for e in EV if e["phase"] == "adam"]
cert = [e for e in EV if e["phase"] == "cert"]

def s(evs, k="dt_ms"): return sum(e[k] for e in evs)
print(f"meta: F={M['F']} opt={M['opt_s']:.0f}s cert={M['cert_s']:.0f}s total={M['total_s']:.0f}s "
      f"conv={M['conv']}/{M['F']} nll={M['nll_nats']:.0f}nats  ({len(newton)} newton iters)")

# ---------- COARSE: per-iteration time distribution ----------
print("\n" + "="*88 + "\nCOARSE: per-iteration wall-time (ms) THROUGHOUT the run (not averaged)\n" + "="*88)
print(f"{'gstep':>5} {'it':>3} {'pi':>3} {'n_active':>8} {'n_conv':>6} {'kind':>12} {'|Pg|max':>9} {'dt_ms':>8}")
for e in newton:
    kind = ("DROP+rebuild" if e["rebuilt"] else "DROP" if e["certverify"] else
            "step+hess" if e["hess"] else "step")
    print(f"{e['gstep']:>5} {e['it']:>3} {e['pi']:>3} {e['n_active']:>8} {e['n_conv']:>6} {kind:>12} "
          f"{e['pgmax']:>9.2e} {e['dt_ms']:>8.0f}")

# categorize newton iterations
cat = {"step": [], "step+hess": [], "DROP": [], "DROP+rebuild": []}
for e in newton:
    k = ("DROP+rebuild" if e["rebuilt"] else "DROP" if e["certverify"] else "step+hess" if e["hess"] else "step")
    cat[k].append(e)
print("\n--- time by iteration kind (newton dt_ms is the FULL iteration incl. cert-verify/resid/rebuild) ---")
for k, evs in cat.items():
    if evs:
        ts = [e["dt_ms"] for e in evs]
        print(f"  {k:>13}: n={len(evs):3d}  total={sum(ts)/1000:6.1f}s  "
              f"min={min(ts):6.0f}  med={sorted(ts)[len(ts)//2]:6.0f}  max={max(ts):6.0f} ms")
print(f"\n  sub-events folded inside DROP iters:  rebuild total={s(rebuild)/1000:.1f}s ({len(rebuild)})  "
      f"resid total={s(resid)/1000:.1f}s ({len(resid)})")
print(f"  whole-run phase totals:  build={s(build)/1000:.1f}s  adam={s(adam)/1000:.1f}s  "
      f"newton(all)={s(newton)/1000:.1f}s  cert={s(cert)/1000:.1f}s")

# time grouped by active-batch-size bucket
print("\n--- newton wall-time grouped by active batch size (where does the time concentrate?) ---")
buckets = [(3000, 9999, ">=3000"), (1000, 2999, "1000-2999"), (100, 999, "100-999"),
           (30, 99, "30-99"), (0, 29, "<30")]
for lo, hi, lbl in buckets:
    evs = [e for e in newton if lo <= e["n_active"] <= hi]
    if evs:
        print(f"  batch {lbl:>10}: {len(evs):3d} iters  {s(evs)/1000:6.1f}s  "
              f"({100*s(evs)/s(newton):4.1f}% of newton time)")

# ---------- FINE: convergence waste ----------
print("\n" + "="*88 + "\nFINE: are we stepping families that already converged (|Pg|<TOL)?\n" + "="*88)
steps = [e for e in newton if not e["certverify"]]   # iters that actually take a Newton step on all active
wasted_fs = sum(e["n_conv"] for e in steps)
total_fs = sum(e["n_active"] for e in steps)
# time-weighted: fraction of each stepping iter's time spent on already-converged families
wasted_time = sum(e["dt_ms"] * (e["n_conv"]/max(e["n_active"],1)) for e in steps) / 1000
print(f"  Newton-stepping iters (excl. drop-check iters): {len(steps)}")
print(f"  family-steps on ALREADY-converged families = {wasted_fs:,} / {total_fs:,} total "
      f"= {100*wasted_fs/max(total_fs,1):.1f}%")
print(f"  time-weighted wasted compute ~= {wasted_time:.1f}s of {s(steps)/1000:.1f}s stepping-time "
      f"({100*wasted_time/max(s(steps),1e-9):.1f}%)")
print("\n  per-iter already-converged lingering (n_conv of n_active), stepping iters only:")
for e in steps:
    if e["n_conv"] > 0:
        bar = "#" * int(40*e["n_conv"]/max(e["n_active"],1))
        print(f"   gstep{e['gstep']:>4} it{e['it']:>3} pi{e['pi']:>3}  {e['n_conv']:>5}/{e['n_active']:<5} "
              f"({100*e['n_conv']/e['n_active']:3.0f}%) {bar}  {e['dt_ms']:.0f}ms")
