"""
rware_compare.py - the fleet comparison: does the demand-geometry
mismatch propagate to fleet-level fulfillment, or wash out?

FOUR STREAMS, one schedule (identical orders, sizes and item counts):
  A  REAL-TRUE   held-out real baskets, intact          (ground truth)
  B  REAL-POOL   real items, re-assembled i.i.d.        (ARTIFACT CONTROL)
  C  SYN-POOL    TabSyn items, same rule                (test)
  D  SYN-POOL    CTGAN items, same rule                 (test - the
                                                         fabrication case)

HEADLINE COMPARISON IS B vs C (and B vs D). A anchors ground truth and
prices the assembly rule: if |A-B| is large, the assembly artifact
dominates and THAT is the finding, not a clean generator verdict.

Every verdict is a FIRE RATE across independent draws x mappings
against a MEASURED bar (95th pct of real-vs-real |differences|),
never a single point estimate.

Reads:  data/v3_compare.csv, data/v3_train_order_ids.csv,
        data/tabsyn/synthetic_tabsyn.csv, data/synthetic_v3.csv
Writes: results/rware/comparison.json, results/rware/runs.jsonl

Usage:  python rware_compare.py [--orders N] [--draws K] [--quick]
"""
import argparse
import json
import os
import time

import numpy as np

import rware_bridge as rb

ap = argparse.ArgumentParser()
ap.add_argument("--orders", type=int, default=200)
ap.add_argument("--draws", type=int, default=10)
ap.add_argument("--maps", type=int, nargs="+", default=[11, 22, 33])
ap.add_argument("--policy", default="nearest")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.orders, args.draws, args.maps = 60, 3, [11]

METRICS = ["steps_per_delivery", "throughput_per_1k",
           "mean_order_completion"]
PRIMARY = "steps_per_delivery"

os.makedirs(rb.RESULTS_DIR, exist_ok=True)
runs_path = os.path.join(rb.RESULTS_DIR, "runs.jsonl")
runs_f = open(runs_path, "w", encoding="utf-8")
t0 = time.time()

print("=" * 70)
print("RWARE FLEET COMPARISON - real vs synthetic order streams")
print("=" * 70)
print(f"  orders/run {args.orders}   draws {args.draws}   "
      f"mappings {args.maps}   policy {args.policy}")
print(f"  fleet: medium layout, {rb.FLEET['n_agents']} agents, "
      f"queue {rb.FLEET['request_queue_size']}")

orders = rb.load_real_orders("data/v3_compare.csv",
                             "data/v3_train_order_ids.csv")
pools = {
    "B_real_pool": rb.real_item_pool(orders),
    "C_tabsyn": rb.load_item_pool("data/tabsyn/synthetic_tabsyn.csv"),
    "D_ctgan": rb.load_item_pool("data/synthetic_v3.csv"),
}
print(f"  real held-out orders: {len(orders):,}")

env0 = rb.make_env(); env0.reset(seed=0)
amaps = {}
for ms in args.maps:
    amaps[ms] = rb.build_aisle_shelf_map(env0, ms)
    p = os.path.join(rb.DATA_DIR, f"aisle_shelf_map_s{ms}.json")
    fp = rb.save_aisle_shelf_map(amaps[ms], ms, p)
    print(f"  map_seed {ms}: fingerprint {fp}")

records = []
deadlocks = []


def do_run(stream, amap, seed, tag):
    r = rb.run_episode(stream, amap, env_seed=seed, policy=args.policy)
    r["tag"] = tag
    runs_f.write(json.dumps(r) + "\n")
    runs_f.flush()
    if not r["valid"]:
        deadlocks.append({k: r[k] for k in
                          ("tag", "reason", "deliveries",
                           "n_requested", "steps")})
    return r


# ══ the experiment ═══════════════════════════════════════
# For each mapping x draw: one schedule, four streams built from it,
# plus a SECOND independent real stream that supplies the null.
for ms in args.maps:
    amap = amaps[ms]
    print(f"\n--- mapping {ms} " + "-" * 50)
    for k in range(args.draws):
        rng_s = np.random.RandomState(1000 + k)
        sched = rb.build_schedule(orders, args.orders, rng_s)

        streams = {
            "A_real_true": rb.stream_from_schedule_real(orders, sched),
            "B_real_pool": rb.stream_from_schedule_pool(
                pools["B_real_pool"], sched,
                np.random.RandomState(2000 + k)),
            "C_tabsyn": rb.stream_from_schedule_pool(
                pools["C_tabsyn"], sched,
                np.random.RandomState(2000 + k)),
            "D_ctgan": rb.stream_from_schedule_pool(
                pools["D_ctgan"], sched,
                np.random.RandomState(2000 + k)),
        }
        # matched-size assertion, every draw
        n_items = {n: sum(len(o["aisles"]) for o in s)
                   for n, s in streams.items()}
        assert len(set(n_items.values())) == 1, \
            f"FATAL: item counts diverged {n_items}"

        # TWO nulls, because the two questions differ.
        #
        # NULL_pool is the one the headline uses: an independent
        # redraw from the SAME real pool, same schedule, same env
        # seed. It differs from B in exactly the way C differs from
        # B - the assembly draw - and in no other way, so it measures
        # what "no generator effect" looks like. Using a true-basket
        # null here would price stream-level variation that the B-vs-C
        # contrast does not contain.
        null_pool = rb.stream_from_schedule_pool(
            pools["B_real_pool"], sched, np.random.RandomState(3000 + k))
        # NULL_real2 is stream-level context for the A anchor.
        sched_n = rb.build_schedule(orders, args.orders,
                                    np.random.RandomState(9000 + k))
        null_stream = rb.stream_from_schedule_real(orders, sched_n)

        row = {"map_seed": ms, "draw": k}
        for name, st in streams.items():
            r = do_run(st, amap, k, f"m{ms}_d{k}_{name}")
            row[name] = {m: r[m] for m in METRICS}
            row[name]["valid"] = r["valid"]
        rn = do_run(null_stream, amap, k, f"m{ms}_d{k}_NULLreal")
        row["NULL_real2"] = {m: rn[m] for m in METRICS}
        row["NULL_real2"]["valid"] = rn["valid"]
        rnp = do_run(null_pool, amap, k, f"m{ms}_d{k}_NULLpool")
        row["NULL_pool"] = {m: rnp[m] for m in METRICS}
        row["NULL_pool"]["valid"] = rnp["valid"]
        records.append(row)

        ok = all(row[n]["valid"] for n in
                 ["A_real_true", "B_real_pool", "C_tabsyn",
                  "D_ctgan", "NULL_real2", "NULL_pool"])
        print(f"  draw {k}: " + "  ".join(
            f"{n.split('_')[0]}={row[n][PRIMARY]:.3f}"
            if row[n][PRIMARY] else f"{n.split('_')[0]}=DEADLOCK"
            for n in ["A_real_true", "B_real_pool", "C_tabsyn",
                      "D_ctgan", "NULL_pool"])
            + ("" if ok else "   <-- INVALID RUN PRESENT"))

runs_f.close()


# ══ analysis: measured bar + fire rates ══════════════════
def vals(name, metric):
    return np.array([r[name][metric] for r in records
                     if r[name]["valid"] and r[name][metric] is not None])


def paired(a, b, metric):
    out = []
    for r in records:
        if r[a]["valid"] and r[b]["valid"] \
                and r[a][metric] is not None and r[b][metric] is not None:
            out.append(r[a][metric] - r[b][metric])
    return np.array(out)


summary = {
    "config": vars(args),
    "LIMITATION_mapping": (
        "aisle->shelf map is a modeling abstraction, not a real "
        "warehouse layout; the comparison is valid because the map is "
        "IDENTICAL across streams, not because it is physically real"),
    "LIMITATION_assembly": (
        "CTGAN/TabSyn emit independent item rows with no order_id, so "
        "baskets for B/C/D are assembled i.i.d. from a real size "
        "schedule; basket-size realism is NOT tested and cannot be"),
    "n_records": len(records),
    "deadlocks": deadlocks,
}

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for metric in METRICS:
    # headline bar: B vs an independent redraw of the SAME real pool
    null_diff = np.abs(paired("B_real_pool", "NULL_pool", metric))
    null_stream_diff = np.abs(paired("A_real_true", "NULL_real2", metric))
    if len(null_diff) == 0:
        continue
    bar = float(np.percentile(null_diff, 95))
    block = {"null_mean": float(null_diff.mean()),
             "bar_95pct": bar,
             "n_null": int(len(null_diff)),
             "stream_level_null_mean": (float(null_stream_diff.mean())
                                        if len(null_stream_diff) else None)}
    print(f"\n[{metric}]")
    print(f"  measured bar (95th pct of |B - same-pool redraw|, "
          f"n={len(null_diff)}): {bar:.4f}")
    print(f"  {'stream':<14}{'mean':>10}{'sd':>9}   vs B: "
          f"signed    |diff|   fires")
    for name in ["A_real_true", "B_real_pool", "C_tabsyn", "D_ctgan"]:
        v = vals(name, metric)
        if len(v) == 0:
            continue
        signed = paired(name, "B_real_pool", metric)
        d = np.abs(signed)
        fires = int((d > bar).sum())
        rate = fires / len(d) if len(d) else float("nan")
        # SIGN matters for the write-up: the question is not only
        # whether synthetic differs from real, but whether it makes
        # the fleet look FASTER or SLOWER than it really is.
        star = "" if name == "B_real_pool" else \
            f"  {signed.mean():>+8.4f}  {d.mean():>8.4f}   " \
            f"{fires}/{len(d)} ({100*rate:.0f}%)"
        print(f"  {name:<14}{v.mean():>10.4f}{v.std():>9.4f}{star}")
        block[name] = {
            "mean": float(v.mean()), "sd": float(v.std()),
            "n": int(len(v)),
            "signed_diff_vs_B_mean": (float(signed.mean())
                                      if len(signed) else None),
            "abs_diff_vs_B_mean": float(d.mean()) if len(d) else None,
            "fires_vs_B": fires, "n_draws": int(len(d)),
            "fire_rate_vs_B": float(rate) if len(d) else None,
        }
    summary[metric] = block

# per-mapping fire rates for the primary metric - a verdict that only
# holds under one mapping is a mapping artifact, not a finding
print(f"\n[{PRIMARY}] fire rate BY MAPPING (mapping-robustness)")
by_map = {}
for ms in args.maps:
    sub = [r for r in records if r["map_seed"] == ms]
    nd = np.abs([r["B_real_pool"][PRIMARY] - r["NULL_pool"][PRIMARY]
                 for r in sub if r["B_real_pool"]["valid"]
                 and r["NULL_pool"]["valid"]])
    if len(nd) == 0:
        continue
    bar_m = float(np.percentile(nd, 95))
    line = {"bar": bar_m}
    for name in ["A_real_true", "C_tabsyn", "D_ctgan"]:
        d = np.abs([r[name][PRIMARY] - r["B_real_pool"][PRIMARY]
                    for r in sub if r[name]["valid"]
                    and r["B_real_pool"]["valid"]])
        line[name] = {"fires": int((d > bar_m).sum()),
                      "n": int(len(d)),
                      "mean_abs_diff": float(np.mean(d)) if len(d) else None}
    by_map[ms] = line
    print(f"  map {ms} (bar {bar_m:.4f}): " + "  ".join(
        f"{n.split('_')[0]} {line[n]['fires']}/{line[n]['n']}"
        for n in ["A_real_true", "C_tabsyn", "D_ctgan"]))
summary["by_mapping_primary"] = by_map

if deadlocks:
    print(f"\n  !! {len(deadlocks)} DEADLOCKED RUN(S) - excluded from "
          f"metrics:")
    for d in deadlocks[:10]:
        print(f"     {d['tag']}: {d['reason']} "
              f"{d['deliveries']}/{d['n_requested']}")
else:
    print("\n  no deadlocks: every run completed its full stream")
summary["elapsed_sec"] = round(time.time() - t0, 1)

with open(os.path.join(rb.RESULTS_DIR, "comparison.json"), "w",
          encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\n-> {os.path.join(rb.RESULTS_DIR, 'comparison.json')}")
print(f"-> {runs_path}")
print(f"   ({summary['elapsed_sec']}s)")
