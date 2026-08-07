"""rwseason_compare.py -- SEASONALLY-CONDITIONED DEMAND THROUGH THE FLEET SIM.

This is the brief's clause "these differently conditioned data patterns will be
fed to the simulator to test the performance of the fleet", for the DATE
conditioning axis (Dunnhumby weeks 35-40 vs baseline).

THE ORDER OF QUESTIONS MATTERS
------------------------------
Q1 (anchor, REAL data only)  does the real seasonal shift move fleet
   performance at all? This holds regardless of how good any generator is,
   and it is the analogue of `test_real_fleet_effect.py` on the Instacart
   side: establish the real effect FIRST, then ask whether synthetic
   reproduces it. If reality shows no fleet-level seasonal effect, then a
   synthetic miss on the seasonal shift has no operational consequence -- and
   that is itself a publishable result.
Q2 (generator)  does the conditional TabSyn output reproduce whatever Q1 says?

STREAMS  (one shared basket-size schedule; only the CATEGORY MIX varies)
    A_real   real Dunnhumby baskets, intact              (ground truth)
    B_pool   the same real items, re-assembled i.i.d.    (ASSEMBLY CONTROL)
    C_syn    conditional-TabSyn items, same rule         (test)
each run twice: once filled from WINDOW items, once from BASELINE items.
The contrast is (window - baseline) WITHIN each source, so the shared
schedule cancels basket-size effects -- the same size control the TVD
analysis uses.

WHY B EXISTS: `data/dunnhumby/synthetic_season.csv` has no basket_id (TabSyn
emits independent item rows and does not model basket membership), so an
assembly rule is unavoidable. B prices that rule. If |A-B| dominates, THAT is
the finding -- exactly as it was on the Instacart bench, where basket
structure moved latency 87% of draws while item content did not.

LAYOUT: Dunnhumby has 300 categories and the medium layout has only 144
storage cells, so this uses shelf_columns=7, column_height=8, shelf_rows=3
-> 320 shelves, permitting a genuine BIJECTION (asserted). `rware_bridge`
reads its layout from module constants at call time, so those constants are
REBOUND here rather than edited -- rware_bridge.py itself is READ-ONLY and is
not modified.

STATED LIMITATION (carried over, and it applies with full force): the
category->shelf map is a MODELING ABSTRACTION, not a real store layout.
Validity rests on the map being IDENTICAL across real and synthetic, not on
physical realism -- hence multiple map seeds, since travel geometry is
relabelling-sensitive.

READS  : data/dunnhumby/dj_items.csv
         data/dunnhumby/synthetic_season.csv
         results/dunnhumby/tabsyn_prep_report.json
         results/dunnhumby/signal_search.json
WRITES : results/rwseason/{fixtures,comparison}.json, runs.jsonl
         data/rwseason/cat_shelf_map_s*.json                      (all NEW)

Usage:
    python rwseason_compare.py --fixtures
    python rwseason_compare.py
    python rwseason_compare.py --quick
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import rware_bridge as rb

ap = argparse.ArgumentParser()
ap.add_argument("--orders", type=int, default=250)
ap.add_argument("--draws", type=int, default=10)
ap.add_argument("--maps", type=int, nargs="+", default=[11, 22])
ap.add_argument("--fixtures", action="store_true")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.orders, args.draws, args.maps = 50, 2, [11]

RESULTS_DIR = os.path.join("results", "rwseason")
DATA_DIR = os.path.join("data", "rwseason")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

METRICS = ["steps_per_delivery", "throughput_per_1k", "mean_order_completion"]
PRIMARY = "steps_per_delivery"
PERIODS = ["window", "baseline"]
SOURCES = ["A_real", "B_pool", "C_syn"]
MAX_BASKET = 40      # cap: the 8-slot queue makes very long baskets pure
                     # serialisation; capped identically in EVERY condition


def hr(c="="):
    print(c * 74)


# ── rebind the fleet layout (rware_bridge is READ-ONLY; these are its
#    module constants, read at call time by make_env/build_aisle_shelf_map) ──
rb.FLEET = dict(shelf_columns=7, shelf_rows=3, column_height=8,
                n_agents=8, request_queue_size=8)
N_CAT = 300
rb.N_AISLES = N_CAT

hr()
print("RWSEASON -- seasonally-conditioned demand through the fleet simulator")
hr()

_env = rb.make_env()
_env.reset(seed=0)
print(f"  layout {_env.grid_size}   shelves {len(_env.shelfs)}   "
      f"agents {rb.FLEET['n_agents']}   queue {rb.FLEET['request_queue_size']}")
assert len(_env.shelfs) >= N_CAT, "layout too small for a bijection"


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
prep = json.load(open(os.path.join("results", "dunnhumby",
                                   "tabsyn_prep_report.json")))
gate = json.load(open(os.path.join("results", "dunnhumby",
                                   "signal_search.json")))
W0, W1 = prep["frozen_window"]
LO, HI = prep["usable_weeks"]
TARGET = prep["target_classes"]            # {'baseline':0,'window':1}
INV_TARGET = {v: k for k, v in TARGET.items()}
print(f"  frozen window weeks {W0}..{W1}   usable weeks {LO}..{HI}")
assert gate["discovery"]["frozen_window"] == [W0, W1], "window disagrees with gate"

items = pd.read_csv(os.path.join("data", "dunnhumby", "dj_items.csv"),
                    usecols=["basket_id", "category", "week_of_year"])
items = items[items["week_of_year"].between(LO, HI)]

# the category vocabulary MUST be the one the model was trained on
syn = pd.read_csv(os.path.join("data", "dunnhumby", "synthetic_season.csv"),
                  usecols=["category", "season_period"])
CATS = sorted(set(syn["category"]))
assert len(CATS) == N_CAT, f"expected {N_CAT} synthetic categories, got {len(CATS)}"
unmapped = set(syn["category"]) - set(items["category"])
assert not unmapped, f"synthetic invented categories: {sorted(unmapped)[:5]}"
CODE = {c: i + 1 for i, c in enumerate(CATS)}     # 1..300, like aisle_id
items = items[items["category"].isin(set(CATS))].copy()
items["code"] = items["category"].map(CODE).astype(int)
items["period"] = np.where(items["week_of_year"].between(W0, W1),
                           "window", "baseline")

# synthetic period labels are integer CODES (season_period is TabSyn's
# binclass target) -- the same trap that would have scored empty pools in
# dunnhumby_conditional_test.py
syn["period"] = syn["season_period"].map(INV_TARGET)
assert syn["period"].notna().all(), "unmapped synthetic season_period codes"
syn["code"] = syn["category"].map(CODE).astype(int)

# real baskets, capped
g = items.groupby("basket_id")
baskets = []
for bid, gg in g:
    per = gg["period"].iloc[0]
    assert (gg["period"] == per).all(), f"basket {bid} spans periods"
    baskets.append({"order_id": int(bid), "grp": per,
                    "aisles": [int(c) for c in gg["code"].values[:MAX_BASKET]]})
real_by_period = {p: [b for b in baskets if b["grp"] == p] for p in PERIODS}
print(f"  real baskets: window {len(real_by_period['window']):,}   "
      f"baseline {len(real_by_period['baseline']):,}")

pools = {
    ("B_pool", p): items.loc[items["period"] == p, "code"].to_numpy()
    for p in PERIODS}
pools.update({
    ("C_syn", p): syn.loc[syn["period"] == p, "code"].to_numpy()
    for p in PERIODS})
for k, v in pools.items():
    assert len(v) > 0, f"empty pool {k}"
    print(f"  pool {k[0]:<8} {k[1]:<9} {len(v):>10,} items   "
          f"{len(np.unique(v)):>3} distinct categories")


def build_map(map_seed):
    rng = np.random.RandomState(map_seed)
    chosen = rng.permutation(len(_env.shelfs))[:N_CAT]
    m = {int(c): int(s) for c, s in zip(range(1, N_CAT + 1), chosen)}
    assert len(m) == N_CAT and len(set(m.values())) == N_CAT, "not a bijection"
    return m


def build_schedule(rng, n):
    """ONE size schedule shared by every source and BOTH periods, drawn from
    real baskets pooled across periods, so basket size cannot differ between
    the conditions being contrasted."""
    idx = rng.choice(len(baskets), n, replace=False)
    return [{"order_id": baskets[i]["order_id"],
             "size": len(baskets[i]["aisles"])} for i in idx]


def stream_real(period, sched, rng):
    """Stream A: intact real baskets of the given period, matched to the
    schedule's SIZES (nearest available basket of that size)."""
    pool = real_by_period[period]
    by_size = {}
    for b in pool:
        by_size.setdefault(len(b["aisles"]), []).append(b)
    sizes = sorted(by_size)
    out = []
    for s in sched:
        want = s["size"]
        near = min(sizes, key=lambda x: (abs(x - want), x))
        b = by_size[near][rng.randint(len(by_size[near]))]
        out.append({"order_id": s["order_id"], "grp": period,
                    "aisles": list(b["aisles"])})
    return out


def stream_pool(key, sched, rng):
    """Streams B/C: schedule sizes filled i.i.d. WITH replacement from the
    source's item pool (with replacement, so within-basket category repeats
    are preserved -- drawing distinct categories would emit zero repeats and
    distort exactly the axis under test)."""
    p = pools[key]
    return [{"order_id": s["order_id"], "grp": key[1],
             "aisles": [int(x) for x in rng.choice(p, s["size"])]}
            for s in sched]


def build_streams(sched, k):
    st = {}
    for period in PERIODS:
        st[("A_real", period)] = stream_real(
            period, sched, np.random.RandomState(4000 + k))
        st[("B_pool", period)] = stream_pool(
            ("B_pool", period), sched, np.random.RandomState(2000 + k))
        st[("C_syn", period)] = stream_pool(
            ("C_syn", period), sched, np.random.RandomState(2000 + k))
    n_items = {kk: sum(len(o["aisles"]) for o in v) for kk, v in st.items()}
    assert len(set(n_items.values())) == 1, f"item counts diverged: {n_items}"
    return st


# ══════════════════════════════════════════════════════════════════
# POWER FIXTURE
# ══════════════════════════════════════════════════════════════════
def run_fixtures():
    hr()
    print("POWER FIXTURE -- measured sensitivity of the window-baseline contrast")
    hr()
    amap = build_map(args.maps[0])
    dist = {}
    for a, si in amap.items():
        sh = _env.shelfs[si]
        best = None
        for gx, gy in _env.goals:
            p = rb.plan_path(_env, (sh.x, sh.y), _env.agents[0].dir,
                             (int(gx), int(gy)), carrying=True)
            if p is not None and (best is None or len(p) < best):
                best = len(p)
        dist[a] = best if best is not None else 10 ** 6
    ranked = sorted(dist, key=lambda a: dist[a])
    near, far = ranked[:60], ranked[-60:]
    print(f"  loaded travel cost: near60 {np.mean([dist[a] for a in near]):.1f} "
          f"steps   far60 {np.mean([dist[a] for a in far]):.1f} steps")

    n_fx = max(4, args.draws // 2)
    null, planted, base = [], [], []
    for k in range(n_fx):
        sched = build_schedule(np.random.RandomState(1000 + k), args.orders)
        rngB = np.random.RandomState(2000 + k)
        s1 = stream_pool(("B_pool", "baseline"), sched, rngB)
        s2 = stream_pool(("B_pool", "baseline"), sched,
                         np.random.RandomState(3000 + k))
        r1 = rb.run_episode(s1, amap, env_seed=k)
        r2 = rb.run_episode(s2, amap, env_seed=k)
        if not (r1["valid"] and r2["valid"]):
            continue
        null.append(abs(r1[PRIMARY] - r2[PRIMARY]))
        # planted: same schedule, but every item routed to the 60 FARTHEST
        rp = np.random.RandomState(7 + k)
        sf = [{"order_id": o["order_id"], "grp": "planted",
               "aisles": [int(x) for x in rp.choice(far, len(o["aisles"]))]}
              for o in s1]
        rf = rb.run_episode(sf, amap, env_seed=k)
        if rf["valid"]:
            base.append(r1[PRIMARY])
            planted.append(rf[PRIMARY])
        print(f"    draw {k}: null |diff| {null[-1]:.4f}   "
              f"planted {rf[PRIMARY]:.3f} vs base {r1[PRIMARY]:.3f}")
    if not null or not planted:
        print("  FIXTURE FAILED: no valid draws")
        return False, {}
    bar = float(np.percentile(null, 95))
    d = np.array(planted) - np.array(base)
    fires = int(np.sum(np.abs(d) > bar))
    ok = fires >= max(1, int(0.75 * len(d)))
    print(f"\n  null mean {np.mean(null):.4f}   95th-pct BAR {bar:.4f}")
    print(f"  planted shift mean {d.mean():+.4f} steps/delivery   "
          f"fires {fires}/{len(d)}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    out = {"map_seed": args.maps[0], "orders": args.orders, "n_draws": n_fx,
           "near60_cost": float(np.mean([dist[a] for a in near])),
           "far60_cost": float(np.mean([dist[a] for a in far])),
           "null_abs_diffs": [float(x) for x in null], "bar_95pct": bar,
           "planted_shift_mean": float(d.mean()),
           "planted_fires": fires, "planted_n": len(d), "PASS": bool(ok),
           "INTERPRETATION": "a null result means 'no detectable effect at a "
                             f"measured sensitivity of {d.mean():+.3f} "
                             "steps/delivery', never 'no effect'"}
    json.dump(out, open(os.path.join(RESULTS_DIR, "fixtures.json"), "w"), indent=2)
    print(f"  -> {os.path.join(RESULTS_DIR, 'fixtures.json')}")
    return ok, out


if args.fixtures:
    ok, _ = run_fixtures()
    raise SystemExit(0 if ok else 1)

fx_path = os.path.join(RESULTS_DIR, "fixtures.json")
if not os.path.isfile(fx_path) or not json.load(open(fx_path)).get("PASS"):
    print(f"\n  REFUSING TO RUN: run `--fixtures` first and have it PASS. "
          f"A null contrast is uninterpretable without measured sensitivity.")
    raise SystemExit(1)
fx = json.load(open(fx_path))
# The bar is a steps/delivery quantity measured at a given stream scale.
# Applying a bar measured at 50 orders to a 250-order comparison is the
# mismatched-basis defect this project has already been burnt by twice.
if fx.get("orders") != args.orders:
    print(f"\n  REFUSING TO RUN: fixture bar was measured at "
          f"{fx.get('orders')} orders/run but the comparison is configured for "
          f"{args.orders}. Re-run `--fixtures --orders {args.orders}`.")
    raise SystemExit(1)
print(f"\n  power fixture PASSED: sensitivity {fx['planted_shift_mean']:+.4f} "
      f"steps/delivery vs bar {fx['bar_95pct']:.4f}")


# ══════════════════════════════════════════════════════════════════
# THE COMPARISON
# ══════════════════════════════════════════════════════════════════
t0 = time.time()
runs_f = open(os.path.join(RESULTS_DIR, "runs.jsonl"), "w", encoding="utf-8")
records, deadlocks = [], []
amaps = {}
for ms in args.maps:
    amaps[ms] = build_map(ms)
    fp = rb.map_fingerprint(amaps[ms])
    json.dump({"map_seed": ms, "fingerprint": fp,
               "map": {str(k): v for k, v in amaps[ms].items()}},
              open(os.path.join(DATA_DIR, f"cat_shelf_map_s{ms}.json"), "w"))
    print(f"  map_seed {ms}: fingerprint {fp}")

total = len(args.maps) * args.draws * len(SOURCES) * len(PERIODS)
print(f"\n  {total} episodes planned")
done = 0
for ms in args.maps:
    amap = amaps[ms]
    print(f"\n--- mapping {ms} " + "-" * 52)
    for k in range(args.draws):
        sched = build_schedule(np.random.RandomState(1000 + k), args.orders)
        st = build_streams(sched, k)
        row = {"map_seed": ms, "draw": k}
        for src in SOURCES:
            for per in PERIODS:
                r = rb.run_episode(st[(src, per)], amap, env_seed=k)
                r["tag"] = f"m{ms}_d{k}_{src}_{per}"
                runs_f.write(json.dumps(r) + "\n")
                runs_f.flush()
                if not r["valid"]:
                    deadlocks.append(r["tag"])
                row[f"{src}|{per}"] = {m: r[m] for m in METRICS}
                row[f"{src}|{per}"]["valid"] = r["valid"]
                done += 1
        records.append(row)
        print(f"  d{k}: " + "  ".join(
            f"{s}({p[0]})=" + (f"{row[f'{s}|{p}'][PRIMARY]:.3f}"
                               if row[f"{s}|{p}"][PRIMARY] else "DEAD")
            for s in SOURCES for p in PERIODS) + f"   [{done}/{total}]")
runs_f.close()


# ══════════════════════════════════════════════════════════════════
hr()
print("RESULTS -- (window - baseline) per source")
hr()
bar = fx["bar_95pct"]
res = {}
print(f"  {'source':<10}{'mean w-b':>12}{'sd':>9}{'fires':>10}   (bar "
      f"{bar:.4f})")
for src in SOURCES:
    d = [r[f"{src}|window"][PRIMARY] - r[f"{src}|baseline"][PRIMARY]
         for r in records
         if r[f"{src}|window"]["valid"] and r[f"{src}|baseline"]["valid"]]
    d = np.array(d, dtype=float)
    fires = int(np.sum(np.abs(d) > bar))
    res[src] = {"mean_window_minus_baseline": float(d.mean()) if len(d) else None,
                "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                "draws": [float(x) for x in d],
                "fires": fires, "n": len(d),
                "fire_rate": fires / len(d) if len(d) else None,
                "effect": bool(fires > len(d) // 2)}
    print(f"  {src:<10}{d.mean():>12.4f}{d.std(ddof=1):>9.4f}"
          f"{fires:>7}/{len(d):<3}")

real_effect = res["B_pool"]["effect"]
syn_effect = res["C_syn"]["effect"]
if real_effect and syn_effect:
    verdict = "BOTH show a fleet-level seasonal effect"
elif real_effect and not syn_effect:
    verdict = "REAL shows a fleet-level seasonal effect, SYNTHETIC MISSES it"
elif not real_effect and syn_effect:
    verdict = "SYNTHETIC FABRICATES a fleet-level seasonal effect reality lacks"
else:
    verdict = ("NEITHER shows a detectable fleet-level seasonal effect at a "
               f"measured sensitivity of {fx['planted_shift_mean']:+.3f} "
               "steps/delivery")
print(f"\n  VERDICT: {verdict}")
print(f"  assembly artifact |A - B| context: A {res['A_real']['mean_window_minus_baseline']:+.4f} "
      f"vs B {res['B_pool']['mean_window_minus_baseline']:+.4f}")

json.dump({
    "config": {"orders": args.orders, "draws": args.draws, "maps": args.maps,
               "layout": rb.FLEET, "n_categories": N_CAT,
               "frozen_window": [W0, W1], "usable_weeks": [LO, HI],
               "max_basket": MAX_BASKET, "n_episodes": done,
               "elapsed_s": time.time() - t0},
    "power_fixture": {k: fx[k] for k in
                      ("bar_95pct", "planted_shift_mean", "planted_fires",
                       "planted_n")},
    "results": res,
    "VERDICT": verdict,
    "deadlocks": deadlocks,
    "LIMITATION": "the category->shelf map is a modeling abstraction, not a "
                  "real store layout; validity rests on it being IDENTICAL "
                  "across real and synthetic, hence multiple map seeds",
}, open(os.path.join(RESULTS_DIR, "comparison.json"), "w"), indent=2)
print(f"\n  episodes {done}   deadlocks {len(deadlocks)}   "
      f"elapsed {(time.time() - t0)/60:.1f} min")
print(f"  -> {os.path.join(RESULTS_DIR, 'comparison.json')}")
