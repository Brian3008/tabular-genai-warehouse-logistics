"""
dunnhumby_conditional_test.py - THE CONDITIONAL VERDICT.

Does the trained TabSyn reproduce the real seasonal category shift,
faithfully miss it, or fabricate a different one?

PRE-REGISTERED BEFORE ANY SYNTHETIC OUTPUT WAS SEEN
---------------------------------------------------
Scored against the STRICT week-block bar (ordinary week-to-week
variation), never the generous basket-clustered bar. Three outcomes,
distinguished by TWO measurements - magnitude AND direction:

  REPRODUCES   synthetic TVD fires above the week-block bar on a
               majority of draws AND its category-shift vector agrees
               with the real shift above a MEASURED agreement bar
  MISSES       synthetic TVD fires on a minority - the shift is absent
  FABRICATES   fires above the bar BUT shift agreement is at or below
               the agreement bar: right magnitude, wrong shift

BASIS DISCIPLINE - the one thing that must not be fudged
--------------------------------------------------------
The gate's 0.08276 week-block bar was measured with BASKET-CLUSTERED
sampling. Synthetic rows have NO basket_id (TabSyn emits independent
rows), so that bar cannot be applied to synthetic without mixing
bases - the defect that forced the DCR and marginal-error corrections
earlier in this project.

So every number here is recomputed on ONE common basis:
  item-level, matched on basket_size_grp, at an IDENTICAL per-group
  item budget for every dataset and every window.
The item-level week-block bar is reported alongside the gate's
basket-clustered one, and they are never quoted interchangeably.

KNOWN-ANSWER GATE
-----------------
Before scoring synthetic, this script reproduces the gate's own
year-2 held-out numbers using the gate's own imported functions. If it
cannot reproduce signal_search.json, the scoring pipeline is not the
one that produced the gate and nothing downstream is trustworthy.

Reads:  data/dunnhumby/dj_items.csv
        tabsyn_repo/data/dunnhumby_season/train.csv   (what the model saw)
        data/dunnhumby/synthetic_season.csv
        results/dunnhumby/signal_search.json
Writes: results/dunnhumby/conditional_test.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import dunnhumby_signal_search as ss

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS = Path("results/dunnhumby")
DATA = Path("data/dunnhumby")
TRAIN_CSV = Path("tabsyn_repo/data/dunnhumby_season/train.csv")
SYNTH_CSV = DATA / "synthetic_season.csv"
N_DRAWS = 10
SEED = 20260730

out = {}
print("=" * 68)
print("DUNNHUMBY CONDITIONAL TEST - does TabSyn reproduce the shift?")
print("=" * 68)

gate = json.load(open(RESULTS / "signal_search.json", encoding="utf-8"))
w0, w1 = gate["discovery"]["frozen_window"]
lo, hi = gate["usable_weeks"]
WIN = set(range(w0, w1 + 1))
L = w1 - w0 + 1
print(f"\nfrozen window weeks {w0}..{w1}   usable {lo}..{hi}")


# ══ [1] KNOWN-ANSWER GATE ════════════════════════════════
print("\n" + "=" * 68)
print("[1] KNOWN-ANSWER GATE - reproduce the gate's own year-2 numbers")
print("=" * 68)
_, p1, p2, K_gate, common, cutoff, dropped, dips = ss.load_panels()
ia = p2.select_weeks(WIN)
ib = p2.select_weeks(set(common) - WIN)
rng = np.random.RandomState(ss.SEED + 1)
raw2 = ss.raw_measure(p2, ia, ib, K_gate, rng, ss.N_NULL_DRAWS,
                      ss.N_TEST_DRAWS)
rng = np.random.RandomState(ss.SEED + 2)
ctl2 = ss.controlled_measure(p2, ia, ib, K_gate, rng, ss.N_NULL_DRAWS,
                             ss.N_TEST_DRAWS)
ref = gate["confirmation"]["year2_HELD_OUT"]
checks = {
    "raw_observed": (raw2["observed_mean"], ref["raw"]["observed_mean"]),
    "raw_bar": (raw2["bar"], ref["raw"]["bar"]),
    "ctl_observed": (ctl2["observed_mean"],
                     ref["size_controlled"]["observed_mean"]),
    "ctl_bar": (ctl2["bar"], ref["size_controlled"]["bar"]),
}
# The WEEK-BLOCK pair is the bar this whole test is scored against, so
# it must be reproduced too - not just the basket-clustered numbers.
L = w1 - w0 + 1
rngw = np.random.RandomState(ss.SEED + 3)
vals = {}
for s in range(min(common), max(common) - L + 2):
    w = set(range(s, s + L))
    if not w <= set(common):
        continue
    jw = p2.select_weeks(w)
    jb = p2.select_weeks(set(common) - w)
    if len(jw) < 200 or len(jb) < 200:
        continue
    N = int(ss.BUDGET_FRAC * min(p2.sizes[jw].sum(), p2.sizes[jb].sum()))
    ts = []
    for _ in range(3):
        a = ss._items_capped(p2, jw, N, rngw)
        b = ss._items_capped(p2, jb, N, rngw)
        if a is not None and b is not None:
            ts.append(ss.tvd(a, b, K_gate))
    if ts:
        vals[s] = float(np.mean(ts))
wb_ref = gate["week_block_null"]["year2_HELD_OUT"]
checks["weekblock_frozen_tvd"] = (vals[w0], wb_ref["window_tvd"])
checks["weekblock_bar"] = (
    float(np.percentile([v for s, v in vals.items() if s != w0], 95)),
    wb_ref["bar_95pct_other_windows"])

# and the category-shift Spearman, which the direction test depends on
_real = pd.read_csv(DATA / "dj_items.csv")
_real = _real[(_real["week_of_year"] >= lo) & (_real["week_of_year"] <= hi)]
_sh = []
for _y in (1, 2):
    _d = _real[_real["year"] == _y]
    _a = _d[_d["week_of_year"].isin(WIN)]["category"].value_counts(
        normalize=True)
    _b = _d[~_d["week_of_year"].isin(WIN)]["category"].value_counts(
        normalize=True)
    _i = sorted(set(_a.index) | set(_b.index))
    _sh.append((_a.reindex(_i, fill_value=0)
                - _b.reindex(_i, fill_value=0)).rename(f"y{_y}"))
_D = pd.concat(_sh, axis=1)
checks["category_shift_spearman"] = (
    float(_D["y1"].corr(_D["y2"], method="spearman")),
    gate["category_shift"]["spearman_year1_vs_year2"])

worst = 0.0
for k, (got, want) in checks.items():
    d = abs(got - want)
    worst = max(worst, d)
    print(f"    {k:<24} recomputed {got:.9f}  recorded {want:.9f}  "
          f"|diff| {d:.2e}")
ka_pass = worst <= 1e-9
print(f"    -> worst |diff| {worst:.2e}  "
      f"{'PASS' if ka_pass else 'FAIL'} (spec <= 1e-9)")
assert ka_pass, "FATAL: cannot reproduce the gate; pipeline mismatch"
out["known_answer_gate"] = {
    "worst_abs_diff": worst, "PASS": True,
    "reproduced": {k: {"recomputed": g, "recorded": w}
                   for k, (g, w) in checks.items()}}

if "--gate-only" in sys.argv:
    json.dump(out, open(RESULTS / "conditional_known_answer.json", "w"),
              indent=2)
    print(f"\n--gate-only: stopping before the verdict, as instructed.")
    print(f"-> {RESULTS / 'conditional_known_answer.json'}")
    sys.exit(0)


# ══ [2] ONE COMMON BASIS ═════════════════════════════════
print("\n" + "=" * 68)
print("[2] COMMON BASIS - item-level, matched on basket_size_grp")
print("=" * 68)

real = pd.read_csv(DATA / "dj_items.csv")
real = real[(real["week_of_year"] >= lo) & (real["week_of_year"] <= hi)]
bs = real.groupby("basket_id")["category"].transform("size")
prep = json.load(open(RESULTS / "tabsyn_prep_report.json",
                      encoding="utf-8"))
q33, q67 = prep["basket_size_tertiles"]
real["basket_size_grp"] = np.where(bs <= q33, "small",
                                   np.where(bs >= q67, "large", "mid"))
real["season_period"] = np.where(real["week_of_year"].isin(WIN),
                                 "window", "baseline")
print(f"    real usable population {len(real):,} items "
      f"(tertiles reused from prep: {q33:.0f}/{q67:.0f})")

train = pd.read_csv(TRAIN_CSV)
print(f"    real TRAINING population {len(train):,} items "
      f"(what the model actually saw - the basis of record)")
synth = pd.read_csv(SYNTH_CSV)
print(f"    synthetic {len(synth):,} items")
for c in ("category", "basket_size_grp", "season_period"):
    assert c in synth.columns, f"FATAL: synthetic missing {c}"

# season_period is TabSyn's binclass TARGET, so it comes back as the
# integer CODE (0/1), not the label. Real data uses the labels. Left
# unmapped, every synthetic pool would silently come back EMPTY and
# the test would score noise. Map codes -> labels via the prep report's
# own mapping, then assert the label sets match.
_tgt = json.load(open(RESULTS / "tabsyn_prep_report.json",
                      encoding="utf-8"))["target_classes"]
_inv = {int(v): k for k, v in _tgt.items()}
if not synth["season_period"].astype(str).isin(_inv.values()).all():
    synth["season_period"] = (pd.to_numeric(synth["season_period"],
                                            errors="coerce")
                              .map(_inv))
    print(f"    mapped synthetic season_period codes -> labels {_inv}")
assert set(synth["season_period"].dropna().unique()) == {"window",
                                                         "baseline"}, \
    f"FATAL: bad season_period labels {set(synth['season_period'].unique())}"
assert synth["season_period"].notna().all(), \
    "FATAL: unmapped season_period values in synthetic"
print(f"    synthetic season_period "
      f"{synth['season_period'].value_counts(normalize=True).round(4).to_dict()}")

# category vocabulary fixed by the REAL training population
cats = sorted(set(train["category"].astype(str)))
code = {c: i for i, c in enumerate(cats)}
K = len(cats)
unseen = set(synth["category"].astype(str)) - set(cats)
print(f"    categories: real train {K}   synthetic "
      f"{synth['category'].nunique()}   invented by model: "
      f"{len(unseen)}")
cov = synth["category"].astype(str).isin(cats).mean()
print(f"    synthetic rows in a real category: {100*cov:.3f}%")
out["category_coverage"] = {
    "K_real_train": K,
    "n_synth_categories": int(synth["category"].nunique()),
    "n_invented": len(unseen),
    "frac_synth_rows_in_real_categories": float(cov),
    "coverage_of_real_categories":
        float(len(set(synth["category"].astype(str)) & set(cats)) / K)}

GRPS = ["small", "mid", "large"]


def as_pools(df):
    d = df.copy()
    d["code"] = d["category"].astype(str).map(code)
    d = d[d["code"].notna()]
    d["code"] = d["code"].astype(int)
    return {(p, g): d[(d["season_period"] == p)
                      & (d["basket_size_grp"] == g)]["code"].to_numpy()
            for p in ("window", "baseline") for g in GRPS}


pools = {"real_full": as_pools(real), "real_train": as_pools(train),
         "synth": as_pools(synth)}

# week-block pools: every contiguous L-week window on real_full
blocks = {}
for s in range(lo, hi - L + 2):
    w = set(range(s, s + L))
    if not w <= set(common):
        continue
    d = real.copy()
    d["season_period"] = np.where(d["week_of_year"].isin(w),
                                  "window", "baseline")
    blocks[s] = as_pools(d)
print(f"    {len(blocks)} contiguous {L}-week blocks for the "
      f"item-level week-block bar")

# ONE budget per size-group: the global minimum across every dataset,
# every side and every week block. Guarantees identical n everywhere.
plan = {}
for g in GRPS:
    mins = []
    for src in list(pools.values()) + list(blocks.values()):
        for p in ("window", "baseline"):
            mins.append(len(src[(p, g)]))
    plan[g] = min(mins) // 2
plan = {g: k for g, k in plan.items() if k > 0}
print(f"    common per-group item budget: {plan}  "
      f"(total {sum(plan.values()):,} items per side)")
out["common_basis"] = {"per_group_budget": plan,
                       "items_per_side": int(sum(plan.values())),
                       "n_week_blocks": len(blocks)}


def draw(src, rng):
    a = np.concatenate([src[("window", g)][
        rng.choice(len(src[("window", g)]), k, replace=False)]
        for g, k in plan.items()])
    b = np.concatenate([src[("baseline", g)][
        rng.choice(len(src[("baseline", g)]), k, replace=False)]
        for g, k in plan.items()])
    return a, b


def shift_vec(src, rng, reps=5):
    """Mean signed per-category share difference, window - baseline."""
    acc = np.zeros(K)
    for _ in range(reps):
        a, b = draw(src, rng)
        pa = np.bincount(a, minlength=K) / len(a)
        pb = np.bincount(b, minlength=K) / len(b)
        acc += pa - pb
    return acc / reps


def tvd_draws(src, rng, n=N_DRAWS):
    vals = []
    for _ in range(n):
        a, b = draw(src, rng)
        vals.append(ss.tvd(a, b, K))
    return np.array(vals)


# ══ [3] ITEM-LEVEL WEEK-BLOCK BAR ════════════════════════
print("\n" + "=" * 68)
print("[3] ITEM-LEVEL WEEK-BLOCK BAR (same basis as synthetic)")
print("=" * 68)
rng = np.random.RandomState(SEED)
block_tvd = {s: float(np.mean(tvd_draws(blocks[s], rng, 3)))
             for s in blocks}
frozen_v = block_tvd[w0]
others = [v for s, v in block_tvd.items() if s != w0]
bar_item = float(np.percentile(others, 95))
rank = int(sum(v >= frozen_v for v in block_tvd.values()))
print(f"    frozen window {frozen_v:.5f}   95th pct of the other "
      f"{len(others)} blocks {bar_item:.5f}")
print(f"    frozen window rank {rank}/{len(block_tvd)}")
print(f"    (the gate's basket-clustered bar was "
      f"{gate['week_block_null']['year2_HELD_OUT']['bar_95pct_other_windows']:.5f}"
      f" - DIFFERENT BASIS, not interchangeable)")
out["week_block_bar_item_level"] = {
    "bar_95pct": bar_item, "frozen_window_tvd": frozen_v,
    "rank": rank, "n_blocks": len(block_tvd),
    "gate_basket_clustered_bar":
        gate["week_block_null"]["year2_HELD_OUT"][
            "bar_95pct_other_windows"],
    "NOTE": "item-level basis; NOT interchangeable with the gate bar"}

# ══ [4] REAL vs SYNTHETIC, same basis ════════════════════
print("\n" + "=" * 68)
print("[4] REAL vs SYNTHETIC on the common basis")
print("=" * 68)
res = {}
for name in ("real_train", "real_full", "synth"):
    rng = np.random.RandomState(SEED + 7)
    v = tvd_draws(pools[name], rng)
    fires = int((v > bar_item).sum())
    res[name] = {"tvd_mean": float(v.mean()), "tvd_sd": float(v.std()),
                 "tvd_draws": [float(x) for x in v],
                 "fires_vs_week_block_bar": fires,
                 "n_draws": len(v),
                 "fire_rate": fires / len(v)}
    print(f"    {name:<11} TVD {v.mean():.5f} +/- {v.std():.5f}   "
          f"fires {fires}/{len(v)} ({100*fires/len(v):.0f}%) vs bar "
          f"{bar_item:.5f}")
out["tvd"] = res

# ══ [5] DIRECTION - measured agreement bar ═══════════════
print("\n" + "=" * 68)
print("[5] DIRECTION - does the shift point the same way?")
print("=" * 68)
rng = np.random.RandomState(SEED + 11)
sv_real = shift_vec(pools["real_train"], rng)
sv_synth = shift_vec(pools["synth"], rng)
rho_synth = float(spearmanr(sv_real, sv_synth).statistic)

# MEASURED agreement bar: how well does an ARBITRARY week block's
# shift agree with the frozen window's? That is agreement by accident.
null_rho = []
for s in blocks:
    if s == w0:
        continue
    null_rho.append(float(spearmanr(
        sv_real, shift_vec(blocks[s], rng, 3)).statistic))
bar_rho = float(np.percentile(null_rho, 95))
print(f"    synthetic vs real shift agreement (Spearman): "
      f"{rho_synth:.4f}")
print(f"    MEASURED agreement bar (95th pct of {len(null_rho)} "
      f"arbitrary week blocks): {bar_rho:.4f}")
print(f"    real year1-vs-year2 agreement (the ceiling a faithful")
print(f"      generator could reach): "
      f"{gate['category_shift']['spearman_year1_vs_year2']:.4f}")
out["direction"] = {
    "spearman_synth_vs_real": rho_synth,
    "agreement_bar_95pct_arbitrary_blocks": bar_rho,
    "null_rho_mean": float(np.mean(null_rho)),
    "real_year1_vs_year2_ceiling":
        gate["category_shift"]["spearman_year1_vs_year2"],
    "exceeds_agreement_bar": bool(rho_synth > bar_rho)}

# top synthetic movers, for the write-up
top_r = [cats[i] for i in np.argsort(-sv_real)[:10]]
top_s = [cats[i] for i in np.argsort(-sv_synth)[:10]]
print(f"\n    real top risers:      {top_r[:5]}")
print(f"    synthetic top risers: {top_s[:5]}")
print(f"    overlap in top-10: "
      f"{len(set(top_r) & set(top_s))}/10")
out["direction"]["real_top10_risers"] = top_r
out["direction"]["synth_top10_risers"] = top_s
out["direction"]["top10_overlap"] = len(set(top_r) & set(top_s))

# ══ [6] VERDICT ══════════════════════════════════════════
mag = res["synth"]["fire_rate"] > 0.5
dirn = rho_synth > bar_rho
verdict = ("REPRODUCES" if (mag and dirn)
           else "FABRICATES" if (mag and not dirn)
           else "MISSES")
print("\n" + "=" * 68)
print("VERDICT (pre-registered criteria)")
print("=" * 68)
print(f"  magnitude: fires above the item-level week-block bar on a "
      f"majority?  {mag}")
print(f"             ({res['synth']['fires_vs_week_block_bar']}/"
      f"{res['synth']['n_draws']} draws)")
print(f"  direction: shift agreement above the measured bar?          "
      f"{dirn}")
print(f"             ({rho_synth:.4f} vs bar {bar_rho:.4f})")
print(f"\n  -> {verdict}")
out["verdict"] = {"magnitude_fires_majority": mag,
                  "direction_exceeds_bar": dirn,
                  "VERDICT": verdict}

json.dump(out, open(RESULTS / "conditional_test.json", "w"), indent=2)
print(f"\n-> {RESULTS / 'conditional_test.json'}")
