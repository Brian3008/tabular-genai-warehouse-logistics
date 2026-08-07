"""rwstyle_compare.py -- FLEET MANAGEMENT STYLES PER DEMAND PATTERN.

Closes two open items at once:
  * Hadfield's standing suggestion -- "explore fleet management styles per
    demand pattern" -- which was time-boxed out of the Jul 29 RWARE run
    (`rware_findings.md`, item 6: "NOT run ... remains future work").
  * The brief's clause "these differently conditioned data patterns will be
    fed to the simulator to test the performance of the fleet". Basket size
    is the conditioning axis CTGAN was actually trained on
    (`order_size_grp`), so a small-heavy vs large-heavy demand stream IS a
    conditioned pattern.

THE DESIGN
----------
Three crossed axes, one shared schedule per cell:
    policy      nearest | random     (both ALREADY implemented at
                                      rware_bridge.py:357-374, never exercised
                                      -- the Jul 29 run was `nearest` only)
    demand mix  small-heavy (80/20) | large-heavy (20/80)
    stream      A real-true | B real-pool (assembly control)
                C TabSyn | D CTGAN | NULL (independent real redraw)

TWO QUESTIONS, TWO CONTRASTS
  Q_style  is one policy better, and DOES THE BEST POLICY DEPEND ON THE
           DEMAND MIX? That is an INTERACTION:
               I = (nearest - random | large-heavy)
                 - (nearest - random | small-heavy)
  Q_fidelity do the synthetic streams reproduce whatever the real streams say
           about Q_style? (B vs C and B vs D, per cell.)

POWER FIRST -- a null is uninterpretable without measured sensitivity.
`rware_findings.md` is citable only because F4 established +-1.8
steps/delivery. This script therefore REFUSES to report a comparison until its
own fixture passes. All planted faults are CONCURRENCY-PRESERVING map
relabellings -- demand is untouched, only physical placement moves -- and they
are CHOSEN FROM A CLOSED-FORM power calculation (expected loaded travel per
pick) rather than guessed. Two earlier attempts were rejected on evidence:
  v1  routed large orders onto the 40 farthest aisles -> moved distance AND
      concurrency at once, the exact defect that archived the original RWARE
      F4, and it made episodes crawl behind the 8-slot queue;
  v2  a size-skew-only relabelling -> moved expected travel just +2.04
      steps/pick and shifted steps/delivery by +0.24 against a 0.31 null bar,
      i.e. underpowered -- caught analytically, not after hours of episodes.
v3 brackets the geometry effect with `adversarial` (most-demanded aisle to the
farthest shelf, +6.7 steps/pick) and `benign` (-7.5), and keeps `size_skew`
for the INTERACTION because it is the only one that moves the large-minus-
small differential. PASS is gated on the geometry pair firing in BOTH
directions; the interaction's sensitivity is reported whatever it is, since a
difference of differences carries ~4x the variance and a weak interaction bar
is a stated limitation rather than a reason to suppress the result.

Every verdict is a FIRE RATE over draws x mappings against a MEASURED bar.

READS  : data/v3_compare.csv, data/v3_train_order_ids.csv,
         data/tabsyn/synthetic_tabsyn.csv, data/synthetic_v3.csv
WRITES : results/rwstyle/fixtures.json, comparison.json, runs.jsonl
         (NEW folder -- rware_* and results/rware/ are READ-ONLY and untouched)

Usage:
    python rwstyle_compare.py --fixtures     # power fixture, must pass first
    python rwstyle_compare.py                # the comparison
    python rwstyle_compare.py --quick        # smoke test
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

import rware_bridge as rb   # import-safe: module level is constants only

ap = argparse.ArgumentParser()
ap.add_argument("--orders", type=int, default=300)
ap.add_argument("--draws", type=int, default=10)
ap.add_argument("--maps", type=int, nargs="+", default=[11, 22])
ap.add_argument("--fixtures", action="store_true")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.orders, args.draws, args.maps = 60, 2, [11]

RESULTS_DIR = os.path.join("results", "rwstyle")
DATA_DIR = os.path.join("data", "rwstyle")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

METRICS = ["steps_per_delivery", "throughput_per_1k", "mean_order_completion"]
PRIMARY = "steps_per_delivery"
POLICIES = ["nearest", "random"]
MIXES = {"small_heavy": 0.80, "large_heavy": 0.20}   # fraction of SMALL orders
STREAMS = ["A_real_true", "B_real_pool", "C_tabsyn", "D_ctgan", "NULL_pool"]


def hr(c="="):
    print(c * 74)


# ══════════════════════════════════════════════════════════════════
# demand-mix schedule
# ══════════════════════════════════════════════════════════════════
def build_schedule_mixed(orders, n_orders, rng, frac_small):
    """`rb.build_schedule` samples orders uniformly, so the small/large mix is
    whatever the held-out pool happens to be. Here the mix is the treatment,
    so it is set explicitly. Everything else -- order ids, per-order sizes,
    item counts -- still comes from REAL orders, and the resulting schedule is
    reused verbatim by every stream in the cell."""
    small = [i for i, o in enumerate(orders) if o["grp"] == "small"]
    large = [i for i, o in enumerate(orders) if o["grp"] == "large"]
    n_s = int(round(n_orders * frac_small))
    n_l = n_orders - n_s
    assert n_s <= len(small) and n_l <= len(large), "pool too small for mix"
    idx = np.concatenate([rng.choice(small, n_s, replace=False),
                          rng.choice(large, n_l, replace=False)])
    rng.shuffle(idx)
    return [{"order_id": orders[i]["order_id"], "grp": orders[i]["grp"],
             "size": len(orders[i]["aisles"])} for i in idx]


def loaded_travel_cost(env0, amap):
    """BFS shelf -> nearest goal, CARRYING (highways only) -- the cost a
    loaded agent actually pays. Copied from rware_fixtures.py:167-179;
    that module executes at import time so it cannot be imported."""
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
    return dist


# ══════════════════════════════════════════════════════════════════
# setup
# ══════════════════════════════════════════════════════════════════
hr()
print("RWSTYLE -- FLEET MANAGEMENT STYLES PER DEMAND PATTERN")
hr()
print(f"  orders/run {args.orders}   draws {args.draws}   maps {args.maps}")
print(f"  policies {POLICIES}   mixes "
      f"{ {k: f'{int(v*100)}% small' for k, v in MIXES.items()} }")

orders = rb.load_real_orders("data/v3_compare.csv",
                            "data/v3_train_order_ids.csv")
n_small = sum(o["grp"] == "small" for o in orders)
print(f"  real held-out orders {len(orders):,}  "
      f"(small {n_small:,} / large {len(orders) - n_small:,})")

pools = {
    "B_real_pool": rb.real_item_pool(orders),
    "C_tabsyn": rb.load_item_pool("data/tabsyn/synthetic_tabsyn.csv"),
    "D_ctgan": rb.load_item_pool("data/synthetic_v3.csv"),
}

env0 = rb.make_env()
env0.reset(seed=0)
amaps = {}
for ms in args.maps:
    amaps[ms] = rb.build_aisle_shelf_map(env0, ms)
    fp = rb.save_aisle_shelf_map(amaps[ms], ms,
                                 os.path.join(DATA_DIR, f"aisle_shelf_map_s{ms}.json"))
    print(f"  map_seed {ms}: fingerprint {fp}")

t0 = time.time()


def streams_for(sched, k):
    """All five streams off one schedule. Item counts identical by
    construction; asserted every draw."""
    s = {"A_real_true": rb.stream_from_schedule_real(orders, sched)}
    for nm in ("B_real_pool", "C_tabsyn", "D_ctgan"):
        s[nm] = rb.stream_from_schedule_pool(
            pools[nm], sched, np.random.RandomState(2000 + k))
    s["NULL_pool"] = rb.stream_from_schedule_pool(
        pools["B_real_pool"], sched, np.random.RandomState(3000 + k))
    n_items = {n: sum(len(o["aisles"]) for o in v) for n, v in s.items()}
    assert len(set(n_items.values())) == 1, f"item counts diverged: {n_items}"
    return s


# ══════════════════════════════════════════════════════════════════
# POWER FIXTURE
# ══════════════════════════════════════════════════════════════════
def _demand_freq(orders, amap, grp=None):
    f = {a: 0 for a in amap}
    for o in orders:
        if grp and o["grp"] != grp:
            continue
        for a in o["aisles"]:
            f[a] = f.get(a, 0) + 1
    tot = sum(f.values()) or 1
    return {a: f[a] / tot for a in amap}


def _relabel(amap, order_aisles, costs_sorted):
    """Assign the i-th aisle of `order_aisles` to the i-th shelf of
    `costs_sorted`. Always a bijection over the SAME shelf set, so demand and
    request concurrency are bit-identical -- only physical placement moves."""
    cost_to_shelves = {}
    for a, s in amap.items():
        cost_to_shelves.setdefault(s, None)
    shelves_by_cost = [s for s, _ in costs_sorted]
    out = {a: s for a, s in zip(order_aisles, shelves_by_cost)}
    assert len(out) == len(amap), "planted map not total"
    assert len(set(out.values())) == len(amap), "planted map not injective"
    assert set(out.values()) == set(amap.values()), "planted map changed shelf set"
    return out


def planted_maps(amap, dist, orders):
    """Three CONCURRENCY-PRESERVING faults, each a pure relabelling.

    v1 of this fixture routed large orders onto the 40 farthest aisles. That
    moved travel distance AND concurrency at once (48 large baskets contending
    over 40 shelves behind an 8-slot queue) -- the same defect that got the
    original RWARE F4 archived, and it made episodes crawl.

    Fault choice is NOT guessed. Expected loaded travel per pick,
    sum_a freq(a)*cost(map(a)), is computed in closed form first (reported in
    fixtures.json) and the faults are chosen because that number says they have
    power. A first attempt at a size-skew-only fault moved expected cost by
    +2.04 steps/pick and empirically shifted steps/delivery by only +0.24
    against a 0.31 null bar -- i.e. underpowered, discovered analytically
    rather than after hours of episodes.

      adversarial  most-demanded aisle -> farthest shelf   (+6.7 steps/pick)
      benign       most-demanded aisle -> nearest shelf    (-7.5 steps/pick)
        -> these two bracket the MIX / geometry main effect, the same
           far/near bracketing the original F4 used.
      size_skew    aisles skewed to LARGE baskets -> farthest shelves
        -> the only one that moves the large-minus-small DIFFERENTIAL
           (+1.38 vs the baseline map's -0.08), so it is the lever for the
           INTERACTION.
    """
    by_cost_desc = sorted(((s, dist[a]) for a, s in amap.items()),
                          key=lambda t: -t[1])
    by_cost_asc = sorted(((s, dist[a]) for a, s in amap.items()),
                         key=lambda t: t[1])
    fa = _demand_freq(orders, amap)
    fl = _demand_freq(orders, amap, "large")
    fs = _demand_freq(orders, amap, "small")

    by_demand = sorted(amap, key=lambda a: -fa[a])
    skew = {a: ((fl[a] - fs[a]) / (fl[a] + fs[a])) if (fl[a] + fs[a]) else 0.0
            for a in amap}
    by_skew = sorted(amap, key=lambda a: -skew[a])

    maps = {
        "adversarial": _relabel(amap, by_demand, by_cost_desc),
        "benign": _relabel(amap, by_demand, by_cost_asc),
        "size_skew": _relabel(amap, by_skew, by_cost_desc),
    }
    inv_cost = {}
    for a, s in amap.items():
        inv_cost[s] = dist[a]

    def expected(fr, m):
        return sum(fr[a] * inv_cost[m[a]] for a in amap)

    analytic = {"baseline": {"all": expected(fa, amap),
                             "small": expected(fs, amap),
                             "large": expected(fl, amap)}}
    for k, m in maps.items():
        analytic[k] = {"all": expected(fa, m), "small": expected(fs, m),
                       "large": expected(fl, m)}
    for k in analytic:
        analytic[k]["large_minus_small"] = (analytic[k]["large"]
                                            - analytic[k]["small"])
    return maps, analytic


def run_fixtures():
    hr()
    print("POWER FIXTURE -- measured sensitivity for the mix effect AND the")
    print("                policy x mix interaction")
    hr()
    ms = args.maps[0]
    amap = amaps[ms]
    dist = loaded_travel_cost(env0, amap)
    maps, analytic = planted_maps(amap, dist, orders)

    print(f"  loaded travel cost per shelf: min {min(dist.values())} "
          f"max {max(dist.values())} mean {np.mean(list(dist.values())):.2f}")
    print(f"\n  CLOSED-FORM expected loaded travel per pick "
          f"(sum_a freq(a)*cost(map(a))) --")
    print(f"  the faults were chosen from THIS table, not guessed:")
    print(f"    {'map':<14}{'all':>9}{'small':>9}{'large':>9}{'lg-sm':>9}{'shift':>9}")
    b = analytic["baseline"]["all"]
    for k in ("baseline", "adversarial", "benign", "size_skew"):
        r = analytic[k]
        print(f"    {k:<14}{r['all']:>9.3f}{r['small']:>9.3f}{r['large']:>9.3f}"
              f"{r['large_minus_small']:>9.3f}"
              f"{(r['all'] - b) if k != 'baseline' else 0:>+9.3f}")

    n_fx = max(4, args.draws // 2)

    def cells(the_map, k):
        out = {}
        for mix, frac in MIXES.items():
            sched = build_schedule_mixed(
                orders, args.orders, np.random.RandomState(1000 + k), frac)
            st = rb.stream_from_schedule_real(orders, sched)
            for pol in POLICIES:
                r = rb.run_episode(st, the_map, env_seed=k, policy=pol)
                if not r["valid"]:
                    return None
                out[(mix, pol)] = r[PRIMARY]
        return out

    def overall(c):
        return float(np.mean(list(c.values())))

    def mix_effect(c):
        return float(np.mean([c[("large_heavy", p)] - c[("small_heavy", p)]
                              for p in POLICIES]))

    def policy_effect(c):
        return float(np.mean([c[(m, "nearest")] - c[(m, "random")]
                              for m in MIXES]))

    def interaction(c):
        return ((c[("large_heavy", "nearest")] - c[("large_heavy", "random")])
                - (c[("small_heavy", "nearest")] - c[("small_heavy", "random")]))

    print(f"\n  BASELINE map, {n_fx} draws")
    base = []
    for k in range(n_fx):
        c = cells(amap, k)
        if c is None:
            continue
        base.append((k, c))
        print(f"    draw {k}: overall {overall(c):.3f}  mix {mix_effect(c):+.4f}"
              f"  policy {policy_effect(c):+.4f}  interaction {interaction(c):+.4f}")

    print(f"\n  NULL: independent re-draws under the SAME map")
    n_over, n_mix, n_int = [], [], []
    for k in range(n_fx):
        a, b2 = cells(amap, 100 + k), cells(amap, 500 + k)
        if a is None or b2 is None:
            continue
        n_over.append(abs(overall(a) - overall(b2)))
        n_mix.append(abs(mix_effect(a) - mix_effect(b2)))
        n_int.append(abs(interaction(a) - interaction(b2)))
        print(f"    pair {k}: |d overall| {n_over[-1]:.4f}  "
              f"|d mix| {n_mix[-1]:.4f}  |d interaction| {n_int[-1]:.4f}")
    if not n_over or not base:
        print("  FIXTURE FAILED: no valid draws")
        return False, {}
    bar_over = float(np.percentile(n_over, 95))
    bar_mix = float(np.percentile(n_mix, 95))
    bar_int = float(np.percentile(n_int, 95))
    print(f"    BAR overall {bar_over:.4f}   mix {bar_mix:.4f}   "
          f"interaction {bar_int:.4f}")

    faults = {}
    for name, m in maps.items():
        d_over, d_int = [], []
        print(f"\n  PLANTED '{name}'")
        for k, cb in base:
            cp = cells(m, k)
            if cp is None:
                continue
            d_over.append(overall(cp) - overall(cb))
            d_int.append(interaction(cp) - interaction(cb))
            print(f"    draw {k}: d overall {d_over[-1]:+.4f}   "
                  f"d interaction {d_int[-1]:+.4f}")
        f_over = int(sum(abs(x) > bar_over for x in d_over))
        f_int = int(sum(abs(x) > bar_int for x in d_int))
        faults[name] = {
            "d_overall": [float(x) for x in d_over],
            "d_interaction": [float(x) for x in d_int],
            "mean_d_overall": float(np.mean(d_over)) if d_over else None,
            "mean_d_interaction": float(np.mean(d_int)) if d_int else None,
            "fires_overall": f_over, "fires_interaction": f_int,
            "n": len(d_over)}
        print(f"    -> overall fires {f_over}/{len(d_over)}  "
              f"(mean {np.mean(d_over):+.4f}, bar {bar_over:.4f})   "
              f"interaction fires {f_int}/{len(d_int)}")

    fp = int(sum(x > bar_over for x in n_over))
    # PASS = the harness resolves the GEOMETRY effect it exists to detect, in
    # BOTH directions.
    #
    # NOTE ON FALSE POSITIVES -- an earlier version of this gate additionally
    # required fp == 0 and was MATHEMATICALLY UNSATISFIABLE: the bar is the
    # 95th percentile OF THE SAME null sample, and for n<=20 numpy's
    # interpolated 95th percentile lies strictly below the maximum, so at
    # least one null value always exceeds it. Testing a sample against a
    # threshold derived from that same sample is circular. A genuine
    # false-positive control needs an INDEPENDENT redraw set (as the Jul 29
    # RWARE F4 had, 0/8). It is reported here for transparency but cannot gate
    # the result; a 95th-percentile bar admits ~5% by definition.
    #
    # The interaction is a difference of differences (~4x the variance); its
    # sensitivity is reported whatever it is, and a weak interaction bar is a
    # stated limitation, not grounds to suppress the geometry result.
    adv, ben = faults["adversarial"], faults["benign"]
    ok = (adv["fires_overall"] >= max(1, int(0.75 * adv["n"]))
          and ben["fires_overall"] >= max(1, int(0.75 * ben["n"]))
          and adv["mean_d_overall"] > 0 > ben["mean_d_overall"])
    exp_fp = 0.05 * len(n_over)
    print(f"\n  null exceedances of a same-sample 95th-pct bar: "
          f"{fp}/{len(n_over)} (expected ~{exp_fp:.1f}; >=1 is structural at "
          f"this n, NOT a failure)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    out = {
        "purpose": "measured sensitivity: geometry/mix main effect and the "
                   "policy x mix interaction",
        "map_seed": ms, "orders": args.orders, "n_draws": n_fx,
        "SCALE_NOTE": f"bars are steps/delivery at {args.orders} orders/run; "
                      "valid only for a comparison at the same scale (enforced)",
        "fault_design": "pure map relabellings -- demand and request "
                        "concurrency bit-identical, only placement moves",
        "closed_form_expected_loaded_travel": analytic,
        "null_abs_overall": [float(x) for x in n_over],
        "null_abs_mix": [float(x) for x in n_mix],
        "null_abs_interaction": [float(x) for x in n_int],
        "bar_overall": bar_over, "bar_mix": bar_mix,
        "bar_interaction": bar_int, "bar_95pct": bar_int,
        "faults": faults,
        "null_false_positives_overall": fp,
        "sensitivity_overall": float(np.mean(
            [abs(adv["mean_d_overall"]), abs(ben["mean_d_overall"])])),
        "sensitivity_interaction": faults["size_skew"]["mean_d_interaction"],
        "PASS": bool(ok),
        "INTERPRETATION": "report any null as 'no detectable effect at a "
                          "measured sensitivity of X steps/delivery'. The "
                          "overall/mix/interaction bars are DIFFERENT BASES "
                          "and must never be interchanged.",
    }
    json.dump(out, open(os.path.join(RESULTS_DIR, "fixtures.json"), "w"),
              indent=2)
    print(f"  -> {os.path.join(RESULTS_DIR, 'fixtures.json')}")
    return ok, out


if args.fixtures:
    ok, _ = run_fixtures()
    raise SystemExit(0 if ok else 1)

fx_path = os.path.join(RESULTS_DIR, "fixtures.json")
if not os.path.isfile(fx_path):
    print(f"\n  REFUSING TO RUN: {fx_path} missing.\n"
          f"  Run `python rwstyle_compare.py --fixtures` first -- a null "
          f"interaction is uninterpretable without measured sensitivity.")
    raise SystemExit(1)
fx = json.load(open(fx_path))
if not fx.get("PASS"):
    print(f"\n  REFUSING TO RUN: power fixture did not pass.")
    raise SystemExit(1)
# The fixture bar is a steps/delivery quantity measured at a given stream
# scale. Applying a bar measured at 60 orders to a 300-order comparison would
# be exactly the mismatched-basis defect this project has already been burnt
# by twice (DCR bases, marginal-error bases).
if fx.get("orders") != args.orders:
    print(f"\n  REFUSING TO RUN: fixture bar was measured at "
          f"{fx.get('orders')} orders/run but the comparison is configured "
          f"for {args.orders}. Bars are scale-dependent -- re-run "
          f"`--fixtures --orders {args.orders}`.")
    raise SystemExit(1)
print(f"\n  power fixture PASSED")
print(f"    geometry sensitivity  {fx['sensitivity_overall']:+.4f} "
      f"steps/delivery  vs bar {fx['bar_overall']:.4f}")
print(f"    interaction sensitivity {fx['sensitivity_interaction']:+.4f} "
      f"vs bar {fx['bar_interaction']:.4f}")


# ══════════════════════════════════════════════════════════════════
# THE COMPARISON
# ══════════════════════════════════════════════════════════════════
runs_f = open(os.path.join(RESULTS_DIR, "runs.jsonl"), "w", encoding="utf-8")
records, deadlocks = [], []
total = len(args.maps) * args.draws * len(MIXES) * len(POLICIES) * len(STREAMS)
print(f"\n  {total} episodes planned")

done = 0
for ms in args.maps:
    amap = amaps[ms]
    print(f"\n--- mapping {ms} " + "-" * 52)
    for k in range(args.draws):
        for mix, frac in MIXES.items():
            sched = build_schedule_mixed(
                orders, args.orders, np.random.RandomState(1000 + k), frac)
            st = streams_for(sched, k)
            for pol in POLICIES:
                row = {"map_seed": ms, "draw": k, "mix": mix, "policy": pol}
                for nm in STREAMS:
                    r = rb.run_episode(st[nm], amap, env_seed=k, policy=pol)
                    r["tag"] = f"m{ms}_d{k}_{mix}_{pol}_{nm}"
                    runs_f.write(json.dumps(r) + "\n")
                    runs_f.flush()
                    if not r["valid"]:
                        deadlocks.append(r["tag"])
                    row[nm] = {m: r[m] for m in METRICS}
                    row[nm]["valid"] = r["valid"]
                    done += 1
                records.append(row)
                print(f"  d{k} {mix:<12} {pol:<8} " + "  ".join(
                    f"{n.split('_')[0]}="
                    + (f"{row[n][PRIMARY]:.3f}" if row[n][PRIMARY] else "DEAD")
                    for n in STREAMS) + f"   [{done}/{total}]")
runs_f.close()


# ══════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════
def cell(mix, pol, stream, metric):
    return np.array([r[stream][metric] for r in records
                     if r["mix"] == mix and r["policy"] == pol
                     and r[stream]["valid"]], dtype=float)


hr()
print("RESULTS")
hr()

# -- measured bar for a policy contrast: |NULL - B| within a cell -------
null_diffs = []
for r in records:
    if r["B_real_pool"]["valid"] and r["NULL_pool"]["valid"]:
        null_diffs.append(abs(r["B_real_pool"][PRIMARY]
                              - r["NULL_pool"][PRIMARY]))
bar = float(np.percentile(null_diffs, 95)) if null_diffs else float("nan")
print(f"\n  POOLED bar (|B - independent redraw|, n={len(null_diffs)}): "
      f"{bar:.4f}   null mean {np.mean(null_diffs):.4f}")

# Per-cell bars too. On the Jul 29 run the pooled bar was inflated by one
# noisier mapping, making it LOOSER than two of the three per-mapping bars --
# quoting only the pooled figure would have understated how often things fire.
per_cell_bar = {}
for mix, pol in itertools.product(MIXES, POLICIES):
    d = [abs(r["B_real_pool"][PRIMARY] - r["NULL_pool"][PRIMARY])
         for r in records if r["mix"] == mix and r["policy"] == pol
         and r["B_real_pool"]["valid"] and r["NULL_pool"]["valid"]]
    if d:
        per_cell_bar[f"{mix}|{pol}"] = float(np.percentile(d, 95))
        print(f"    per-cell bar {mix}|{pol:<8}: "
              f"{per_cell_bar[f'{mix}|{pol}']:.4f}  (n={len(d)})")

# -- Q_style: policy effect per mix, per stream -------------------------
print(f"\n  Q_STYLE -- policy effect (nearest - random), {PRIMARY}")
print(f"  {'stream':<14}{'small_heavy':>14}{'large_heavy':>14}"
      f"{'INTERACTION':>14}{'fires':>10}")
style = {}
for nm in STREAMS:
    per_mix, per_draw_I = {}, []
    for mix in MIXES:
        a, b = cell(mix, "nearest", nm, PRIMARY), cell(mix, "random", nm, PRIMARY)
        n = min(len(a), len(b))
        per_mix[mix] = float(np.mean(a[:n] - b[:n])) if n else float("nan")
    # paired interaction per draw x mapping
    for ms in args.maps:
        for k in range(args.draws):
            def g(mix, pol):
                v = [r[nm][PRIMARY] for r in records
                     if r["map_seed"] == ms and r["draw"] == k
                     and r["mix"] == mix and r["policy"] == pol
                     and r[nm]["valid"]]
                return v[0] if v else None
            vals = {(m, p): g(m, p) for m in MIXES for p in POLICIES}
            if any(v is None for v in vals.values()):
                continue
            per_draw_I.append(
                (vals[("large_heavy", "nearest")] - vals[("large_heavy", "random")])
                - (vals[("small_heavy", "nearest")] - vals[("small_heavy", "random")]))
    I = float(np.mean(per_draw_I)) if per_draw_I else float("nan")
    fires = int(sum(abs(x) > fx["bar_95pct"] for x in per_draw_I))
    style[nm] = {"policy_effect_by_mix": per_mix,
                 "interaction_mean": I,
                 "interaction_draws": [float(x) for x in per_draw_I],
                 "fires_vs_fixture_bar": fires, "n_draws": len(per_draw_I),
                 "fire_rate": fires / len(per_draw_I) if per_draw_I else None}
    print(f"  {nm:<14}{per_mix['small_heavy']:>14.4f}"
          f"{per_mix['large_heavy']:>14.4f}{I:>14.4f}"
          f"{fires:>7}/{len(per_draw_I):<3}")

# -- Q_fidelity: synthetic vs the B control, within each cell -----------
print(f"\n  Q_FIDELITY -- synthetic vs B control, per cell ({PRIMARY})")
print(f"  {'cell':<26}{'A-B':>10}{'C-B':>10}{'D-B':>10}"
      f"{'  fires(C/D) vs bar'}")
fidelity = {}
for mix, pol in itertools.product(MIXES, POLICIES):
    key = f"{mix}|{pol}"
    B = cell(mix, pol, "B_real_pool", PRIMARY)
    row = {}
    for nm in ("A_real_true", "C_tabsyn", "D_ctgan"):
        X = cell(mix, pol, nm, PRIMARY)
        n = min(len(X), len(B))
        d = X[:n] - B[:n]
        row[nm] = {"mean_diff": float(np.mean(d)) if n else None,
                   "fires": int(np.sum(np.abs(d) > bar)), "n": int(n),
                   "fire_rate": float(np.mean(np.abs(d) > bar)) if n else None}
    fidelity[key] = row
    print(f"  {key:<26}{row['A_real_true']['mean_diff']:>10.4f}"
          f"{row['C_tabsyn']['mean_diff']:>10.4f}"
          f"{row['D_ctgan']['mean_diff']:>10.4f}"
          f"   {row['C_tabsyn']['fires']}/{row['C_tabsyn']['n']}"
          f"  {row['D_ctgan']['fires']}/{row['D_ctgan']['n']}")

elapsed = time.time() - t0
out = {
    "config": {"orders": args.orders, "draws": args.draws,
               "maps": args.maps, "policies": POLICIES,
               "mixes": MIXES, "streams": STREAMS,
               "n_episodes": done, "elapsed_s": elapsed},
    "power_fixture": {
        "bar_overall": fx["bar_overall"], "bar_mix": fx["bar_mix"],
        "bar_interaction": fx["bar_interaction"],
        "sensitivity_overall": fx["sensitivity_overall"],
        "sensitivity_interaction": fx["sensitivity_interaction"],
        "closed_form_expected_loaded_travel":
            fx["closed_form_expected_loaded_travel"],
        "faults": {k: {"fires_overall": v["fires_overall"],
                       "n": v["n"], "mean_d_overall": v["mean_d_overall"]}
                   for k, v in fx["faults"].items()}},
    "measured_bar_stream_contrast_POOLED": bar,
    "measured_bar_stream_contrast_PER_CELL": per_cell_bar,
    "BAR_NOTE": "pooled and per-cell bars are DIFFERENT BASES; quote which "
                "one a fire rate came from, never mix them",
    "null_mean": float(np.mean(null_diffs)) if null_diffs else None,
    "Q_style": style,
    "Q_fidelity": fidelity,
    "deadlocks": deadlocks,
    "HOW_TO_CITE": "Report fire rates against the measured bars. A null "
                   "interaction means 'no detectable interaction at a "
                   "measured sensitivity of "
                   f"{fx['sensitivity_interaction']:+.3f} steps/delivery', "
                   "NEVER 'no interaction'. The overall/mix/interaction bars "
                   "are DIFFERENT BASES; say which one a fire rate used.",
}
json.dump(out, open(os.path.join(RESULTS_DIR, "comparison.json"), "w"), indent=2)
print(f"\n  episodes {done}   deadlocks {len(deadlocks)}   "
      f"elapsed {elapsed / 60:.1f} min")
print(f"  -> {os.path.join(RESULTS_DIR, 'comparison.json')}")
