"""
rware_fixtures.py - known-answer gates for the fleet harness.

NOTHING in the comparison is trustworthy until all five pass. A
detector that has never fired on a known-bad case is not a detector.

  F1 BIJECTION        aisle->shelf map is total, injective,
                      reproducible, and IDENTICAL across conditions
  F2 ORDER MATCHING   all four streams carry the same orders, the
                      same sizes and the same item count
  F3 REPEATABILITY    same stream + same seed reproduces exactly;
                      same stream across seeds gives the noise floor
  F4 DISCRIMINATION   a deliberately different stream is detected
                      above that floor (and a same-distribution
                      redraw is NOT - the false-positive check)
  F5 DEADLOCK         a frozen fleet and a starved budget are both
                      reported as deadlock, never as metrics

Reads:  data/v3_compare.csv, data/v3_train_order_ids.csv,
        data/tabsyn/synthetic_tabsyn.csv, data/synthetic_v3.csv
Writes: results/rware/fixtures.json
"""
import json
import os
import time

import numpy as np

import rware_bridge as rb
from rware.warehouse import Action

N_ORDERS = 120          # fixture size - kept small; full runs are larger
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
METRIC = "steps_per_delivery"

os.makedirs(rb.RESULTS_DIR, exist_ok=True)
res = {"LIMITATION": (
    "aisle->shelf map is a modeling abstraction, not a real warehouse "
    "layout; validity rests on it being IDENTICAL across conditions, "
    "not on physical realism")}
t_start = time.time()

print("=" * 68)
print("RWARE HARNESS FIXTURES")
print("=" * 68)

env0 = rb.make_env()
env0.reset(seed=0)
amap = rb.build_aisle_shelf_map(env0, 11)
orders = rb.load_real_orders("data/v3_compare.csv",
                             "data/v3_train_order_ids.csv")

# ══ F1 BIJECTION ═════════════════════════════════════════
print("\n[F1] BIJECTION GATE")
fp = rb.map_fingerprint(amap)
env_b = rb.make_env(); env_b.reset(seed=12345)
fp_again = rb.map_fingerprint(rb.build_aisle_shelf_map(env_b, 11))
f1 = {
    "total": len(amap) == rb.N_AISLES,
    "injective": len(set(amap.values())) == rb.N_AISLES,
    "in_range": all(0 <= v < len(env0.shelfs) for v in amap.values()),
    "reproducible_across_env": fp == fp_again,
    "fingerprint": fp,
}
f1["PASS"] = all(v for k, v in f1.items() if isinstance(v, bool))
for k, v in f1.items():
    print(f"     {k:<26} {v}")
res["F1_bijection"] = f1

# ══ F2 ORDER-COUNT MATCHING ══════════════════════════════
print("\n[F2] ORDER-COUNT MATCHING (the four streams)")
sched = rb.build_schedule(orders, N_ORDERS, np.random.RandomState(7))
pool_real = rb.real_item_pool(orders)
pool_tab = rb.load_item_pool("data/tabsyn/synthetic_tabsyn.csv")
pool_ctg = rb.load_item_pool("data/synthetic_v3.csv")

streams = {
    "A_real_true": rb.stream_from_schedule_real(orders, sched),
    "B_real_pool": rb.stream_from_schedule_pool(
        pool_real, sched, np.random.RandomState(7)),
    "C_tabsyn": rb.stream_from_schedule_pool(
        pool_tab, sched, np.random.RandomState(7)),
    "D_ctgan": rb.stream_from_schedule_pool(
        pool_ctg, sched, np.random.RandomState(7)),
}
sig = {}
for nm, st in streams.items():
    sizes = [len(o["aisles"]) for o in st]
    sig[nm] = {
        "n_orders": len(st),
        "n_items": int(sum(sizes)),
        "size_vector_hash": hash(tuple(sizes)),
        "n_small": sum(o["grp"] == "small" for o in st),
        "n_large": sum(o["grp"] == "large" for o in st),
        "dup_rate": round(
            1 - sum(len(set(o["aisles"])) for o in st) / sum(sizes), 4),
        "distinct_aisles": len({a for o in st for a in o["aisles"]}),
    }
    print(f"     {nm:<12} orders {sig[nm]['n_orders']:>4}  items "
          f"{sig[nm]['n_items']:>5}  small/large "
          f"{sig[nm]['n_small']}/{sig[nm]['n_large']}  "
          f"dup {sig[nm]['dup_rate']:.3f}  "
          f"aisles {sig[nm]['distinct_aisles']}")
ref = sig["A_real_true"]
f2_ok = all(
    sig[n]["n_orders"] == ref["n_orders"]
    and sig[n]["n_items"] == ref["n_items"]
    and sig[n]["size_vector_hash"] == ref["size_vector_hash"]
    and sig[n]["n_small"] == ref["n_small"]
    and sig[n]["n_large"] == ref["n_large"]
    for n in sig)
print(f"     -> identical orders/sizes/items across all four: "
      f"{'PASS' if f2_ok else 'FAIL'}")
print("     (dup_rate differs by design: A keeps real within-basket")
print("      correlation, B/C/D are i.i.d. draws - that gap IS the")
print("      assembly artifact stream B measures)")
res["F2_matching"] = {"per_stream": sig, "PASS": bool(f2_ok)}

# ══ F3 REPEATABILITY ═════════════════════════════════════
print("\n[F3] REPEATABILITY")
sA = streams["A_real_true"]
r1 = rb.run_episode(sA, amap, env_seed=0)
r2 = rb.run_episode(sA, amap, env_seed=0)
exact = (r1["steps"] == r2["steps"]
         and r1["deliveries"] == r2["deliveries"])
print(f"     same stream, same seed: {r1['steps']} vs {r2['steps']} "
      f"steps -> {'IDENTICAL' if exact else 'DIVERGED'}")

floor_vals = []
for s in SEEDS:
    r = rb.run_episode(sA, amap, env_seed=s)
    assert r["valid"], f"fixture run invalid at seed {s}: {r['reason']}"
    floor_vals.append(r[METRIC])
floor_vals = np.array(floor_vals)
print(f"     same stream, {len(SEEDS)} seeds: mean "
      f"{floor_vals.mean():.4f}  sd {floor_vals.std():.4f}  "
      f"range {np.ptp(floor_vals):.4f}")
res["F3_repeatability"] = {
    "exact_same_seed": bool(exact),
    "seed_values": [float(v) for v in floor_vals],
    "mean": float(floor_vals.mean()),
    "sd": float(floor_vals.std()),
    "PASS": bool(exact),
}

# ══ F4 DISCRIMINATION ════════════════════════════════════
# The bar is MEASURED, never guessed: pairs of independent
# same-distribution real streams, bar = 95th pct of their |difference|.
#
# FIXTURE REDESIGN (v2). v1 crushed all demand onto 5 aisles and fired
# only 62% - but it made the fleet FASTER (6.46 vs 6.88), because that
# fault moves two things at once and they nearly cancel:
#   locality UP      - the same 5 shelves are re-fetched constantly
#   concurrency DOWN - only 5 distinct shelves can fill an 8-slot queue
# It landed on a cancellation point, testing the harness at its least
# informative operating point.
#
# v2 moves TRAVEL DISTANCE with concurrency held fixed: demand is
# redirected onto the 40 aisles physically FARTHEST from the goal, and
# separately the 40 NEAREST. 40 aisles keeps the 8-slot queue
# saturated, so only geometry moves. That is precisely the effect the
# study is about - if the harness cannot see this, it cannot see the
# research question either.
print("\n[F4] DISCRIMINATION (measured bar, not a guessed threshold)")

# true loaded-travel cost of every aisle: BFS shelf -> nearest goal on
# highways only, which is what a carrying agent actually pays
dist = {}
for a, si in amap.items():
    sh = env0.shelfs[si]
    best = None
    for gx, gy in env0.goals:
        p = rb.plan_path(env0, (sh.x, sh.y), env0.agents[0].dir,
                         (int(gx), int(gy)), carrying=True)
        if p is not None and (best is None or len(p) < best):
            best = len(p)
    dist[a] = best if best is not None else 10 ** 6
ranked = sorted(dist, key=lambda a: dist[a])
near40, far40 = ranked[:40], ranked[-40:]
print(f"     loaded travel cost to goal: near40 mean "
      f"{np.mean([dist[a] for a in near40]):.1f} steps, far40 mean "
      f"{np.mean([dist[a] for a in far40]):.1f} steps")

null_diffs = []
for k in range(10):
    sch_a = rb.build_schedule(orders, N_ORDERS,
                              np.random.RandomState(100 + k))
    sch_b = rb.build_schedule(orders, N_ORDERS,
                              np.random.RandomState(500 + k))
    ra = rb.run_episode(rb.stream_from_schedule_real(orders, sch_a),
                        amap, env_seed=k)
    rbb = rb.run_episode(rb.stream_from_schedule_real(orders, sch_b),
                         amap, env_seed=k)
    assert ra["valid"] and rbb["valid"], "null draw invalid"
    null_diffs.append(abs(ra[METRIC] - rbb[METRIC]))
null_diffs = np.array(null_diffs)
bar = float(np.percentile(null_diffs, 95))
print("     null (real vs real, same distribution, 10 pairs):")
print(f"       mean |diff| {null_diffs.mean():.4f}   "
      f"95th pct BAR {bar:.4f}")

# baseline = the SAME stream at the SAME seeds already run in F3
base = dict(zip(SEEDS, floor_vals))


def planted_from(pool_aisles, seed):
    rp = np.random.RandomState(seed)
    return [{"order_id": o["order_id"], "grp": o["grp"],
             "aisles": [int(a) for a in
                        rp.choice(pool_aisles, len(o["aisles"]))]}
            for o in sA]


faults = {}
for label, pool_a, tag in (("far40", far40, "farthest from goal"),
                           ("near40", near40, "nearest to goal"),
                           ("five_aisle", [1, 2, 3, 4, 5],
                            "v1 fixture, kept as diagnostic")):
    st = planted_from(pool_a, 3)
    vals_f, fires, signed = [], 0, []
    for s in SEEDS:
        r = rb.run_episode(st, amap, env_seed=s)
        if not r["valid"]:
            continue
        vals_f.append(r[METRIC])
        signed.append(r[METRIC] - base[s])
        fires += abs(r[METRIC] - base[s]) > bar
    n = max(len(vals_f), 1)
    faults[label] = {
        "mean": float(np.mean(vals_f)) if vals_f else None,
        "mean_signed_diff": float(np.mean(signed)) if signed else None,
        "fires": int(fires), "n": len(vals_f),
        "fire_rate": float(fires / n),
        "note": tag,
    }
    print(f"     {label:<11} ({tag}): {np.mean(vals_f):.4f} vs real "
          f"{np.mean(list(base.values())):.4f}  signed "
          f"{np.mean(signed):+.4f}  -> fires {fires}/{len(vals_f)} "
          f"({100 * fires / n:.0f}%)")

# false positive: an independent same-distribution real redraw must NOT
# fire against the same baseline
sched2 = rb.build_schedule(orders, N_ORDERS, np.random.RandomState(77))
fp_fires, fp_n = 0, 0
for s in SEEDS:
    rq = rb.run_episode(rb.stream_from_schedule_real(orders, sched2),
                        amap, env_seed=s)
    if rq["valid"]:
        fp_n += 1
        fp_fires += abs(rq[METRIC] - base[s]) > bar
print(f"     same-distribution redraw: fires {fp_fires}/{fp_n} "
      f"({100 * fp_fires / max(fp_n, 1):.0f}%)  <- must stay low")

# the gate: BOTH geometry faults must be caught, in OPPOSITE directions
# (far = slower, near = faster), with a low false-positive rate. The
# 5-aisle diagnostic is reported but does NOT gate.
f4_ok = (faults["far40"]["fire_rate"] >= 0.75
         and faults["near40"]["fire_rate"] >= 0.75
         and faults["far40"]["mean_signed_diff"] > 0
         and faults["near40"]["mean_signed_diff"] < 0
         and fp_fires / max(fp_n, 1) <= 0.25)
print(f"     -> {'PASS' if f4_ok else 'FAIL'}")
res["F4_discrimination"] = {
    "metric": METRIC,
    "null_diffs": [float(v) for v in null_diffs],
    "bar_95pct": bar,
    "baseline_mean": float(np.mean(list(base.values()))),
    "aisle_travel_cost": {
        "near40_mean": float(np.mean([dist[a] for a in near40])),
        "far40_mean": float(np.mean([dist[a] for a in far40]))},
    "faults": faults,
    "false_positive_rate": float(fp_fires / max(fp_n, 1)),
    "fixture_note": (
        "v1 used a 5-aisle fault which fired only 62%: it moves "
        "locality UP and concurrency DOWN and the two nearly cancel "
        "(the fleet got FASTER). v2 moves travel distance with "
        "concurrency held fixed at 40 aisles."),
    "PASS": bool(f4_ok),
}

# ══ F5 DEADLOCK ══════════════════════════════════════════
print("\n[F5] DEADLOCK DETECTION")
# (a) frozen fleet - robots permanently stuck, the exact failure the
#     abandoned A* congestion sim hit
_real_act = rb.ScriptedFleet.act
rb.ScriptedFleet.act = lambda self: [Action.NOOP.value] * self.env.n_agents
frozen = rb.run_episode(sA, amap, env_seed=0, inactivity_limit=300)
rb.ScriptedFleet.act = _real_act
print(f"     frozen fleet   -> deadlock={frozen['deadlock']} "
      f"valid={frozen['valid']} reason={frozen['reason']} "
      f"deliveries={frozen['deliveries']}")
# (b) starved step budget
starved = rb.run_episode(sA, amap, env_seed=0, step_budget=200)
print(f"     starved budget -> deadlock={starved['deadlock']} "
      f"valid={starved['valid']} reason={starved['reason']} "
      f"deliveries={starved['deliveries']}/{starved['n_requested']}")
# (c) a healthy run must NOT be flagged
healthy = rb.run_episode(sA, amap, env_seed=0)
print(f"     healthy run    -> deadlock={healthy['deadlock']} "
      f"valid={healthy['valid']} "
      f"({healthy['deliveries']}/{healthy['n_requested']})")
f5_ok = (frozen["deadlock"] and not frozen["valid"]
         and starved["deadlock"] and not starved["valid"]
         and not healthy["deadlock"] and healthy["valid"])
print(f"     -> {'PASS' if f5_ok else 'FAIL'}")
res["F5_deadlock"] = {
    "frozen": {k: frozen[k] for k in
               ("deadlock", "valid", "reason", "deliveries")},
    "starved": {k: starved[k] for k in
                ("deadlock", "valid", "reason", "deliveries")},
    "healthy": {k: healthy[k] for k in
                ("deadlock", "valid", "reason", "deliveries")},
    "PASS": bool(f5_ok),
}

# ══ VERDICT ══════════════════════════════════════════════
gates = {k: res[k]["PASS"] for k in res if k.startswith("F")}
res["ALL_PASS"] = bool(all(gates.values()))
res["elapsed_sec"] = round(time.time() - t_start, 1)
print("\n" + "=" * 68)
for k, v in gates.items():
    print(f"  {k:<20} {'PASS' if v else 'FAIL'}")
print(f"  {'ALL GATES':<20} "
      f"{'PASS' if res['ALL_PASS'] else 'FAIL'}   "
      f"({res['elapsed_sec']}s)")
print("=" * 68)

with open(os.path.join(rb.RESULTS_DIR, "fixtures.json"), "w",
          encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"-> {os.path.join(rb.RESULTS_DIR, 'fixtures.json')}")
