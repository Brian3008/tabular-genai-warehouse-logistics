"""1A -- CONDITIONING EFFECTIVENESS: are annual windows distinct FROM EACH
OTHER, and does a profile learned on one window TRANSFER to another?

THE GAP THIS CLOSES
-------------------
`dunnhumby_signal_search.py` (the gate) established that ONE frozen window
(weeks 35-40) differs from its baseline, replicated out-of-sample. It never
tested:
  Q1  whether DIFFERENT windows are distinguishable FROM EACH OTHER, or
      whether any two 6-week blocks differ just as much (which would make
      "seasonal" an overclaim);
  Q2  whether a category-shift profile learned on one window PREDICTS another
      window's shift -- i.e. "does a model for one period work on the other".

Both are answerable from data already on disk. NOTHING is retrained.

METHOD -- identical machinery to the gate, imported not copied
--------------------------------------------------------------
`raw_measure` / `controlled_measure` / `Panel` / `load_panels` are IMPORTED from
`dunnhumby_signal_search` so the detector cannot drift from the one that was
fixture-verified. That module is import-safe: its only module-level side effect
is `os.makedirs(results/dunnhumby, exist_ok=True)` on a directory that exists,
and its entry point is guarded (`if __name__ == "__main__"` at line 589).

All nulls are BASKET-CLUSTERED (items within a trip are correlated -- an
item-level null fired 10/10 on two halves of the same year during the gate's
development). Bars are MEASURED as the 95th percentile of the null. Every
verdict is a FIRE RATE over 10 independent draws, never a point estimate.

WINDOWS: five NON-OVERLAPPING 6-week blocks inside the measured usable range,
anchored so the frozen window is one of them:
    W17_22, W23_28, W29_34, W35_40 (= the gate's frozen window), W41_46

GATES (all must pass before any verdict is computed)
----------------------------------------------------
G1  known-answer: re-running the gate's own frozen-window measurement through
    the imported functions reproduces `signal_search.json` to <= 1e-9.
G2  null fixture: two halves of the SAME window must NOT fire.
G3  planted fixture: a synthetic category shift MUST fire.

READS  : data/dunnhumby/dj_items.csv
         results/dunnhumby/signal_search.json      (known-answer reference)
WRITES : results/dunnhumby/window_discrimination.json          (new)
         results/dunnhumby/window_shift_profiles.csv           (new)
Nothing existing is modified.

Run:  .venv/Scripts/python.exe dunnhumby_window_discrimination.py
      .venv/Scripts/python.exe dunnhumby_window_discrimination.py --gate-only
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import dunnhumby_signal_search as ss

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_DIR = os.path.join("results", "dunnhumby")
OUT_JSON = os.path.join(RESULTS_DIR, "window_discrimination.json")
OUT_CSV = os.path.join(RESULTS_DIR, "window_shift_profiles.csv")
GATE_JSON = os.path.join(RESULTS_DIR, "signal_search.json")

SEED = 20260731
N_NULL = ss.N_NULL_DRAWS      # 40
N_TEST = ss.N_TEST_DRAWS      # 10
TOL = 1e-9

WINDOWS = {
    "W17_22": set(range(17, 23)),
    "W23_28": set(range(23, 29)),
    "W29_34": set(range(29, 35)),
    "W35_40": set(range(35, 41)),   # the gate's frozen window
    "W41_46": set(range(41, 47)),
}
FROZEN = "W35_40"


def hr(c="="):
    print(c * 70)


# ══════════════════════════════════════════════════════════════════
# SHIFT PROFILE  (size-matched, basket-clustered)
# ══════════════════════════════════════════════════════════════════
def shift_profile(panel, idx_a, idx_b, K, rng, reps=5):
    """Mean signed per-category share difference, A - B, over size-matched
    basket draws. Same definition as `shift_vec` in
    dunnhumby_conditional_test.py:295, but drawn basket-clustered and
    size-matched via the gate's own `_match_plan`."""
    plan = ss._match_plan(panel, idx_a, idx_b)
    if not plan:
        return None
    ba, bb = panel.bins[idx_a], panel.bins[idx_b]
    pools_a = {s: idx_a[ba == s] for s in plan}
    pools_b = {s: idx_b[bb == s] for s in plan}

    def draw(pools):
        picks = [pools[s][rng.choice(len(pools[s]), k, replace=False)]
                 for s, k in plan.items()]
        return panel.items_of(np.concatenate(picks))

    acc = np.zeros(K)
    for _ in range(reps):
        a, b = draw(pools_a), draw(pools_b)
        acc += (np.bincount(a, minlength=K) / len(a)
                - np.bincount(b, minlength=K) / len(b))
    return acc / reps


# ══════════════════════════════════════════════════════════════════
# GATES
# ══════════════════════════════════════════════════════════════════
def gate_known_answer(p1, p2, K, common):
    """Reproduce the gate's frozen-window numbers through the imported
    functions. The gate re-seeds a FRESH RandomState per measurement
    (signal_search.py:456,459), so this is bit-reproducible."""
    hr()
    print("G1  KNOWN-ANSWER GATE -- reproduce signal_search.json")
    hr()
    if not os.path.isfile(GATE_JSON):
        print(f"  FAIL: {GATE_JSON} not found")
        return False, {}
    gate = json.load(open(GATE_JSON))
    WIN = WINDOWS[FROZEN]
    rec_win = gate["discovery"]["frozen_window"]
    if set(rec_win) != {min(WIN), max(WIN)}:
        print(f"  FAIL: gate frozen_window {rec_win} != "
              f"{[min(WIN), max(WIN)]}")
        return False, {}

    checks, worst = {}, 0.0
    for label, panel in (("year1_IN_SAMPLE", p1), ("year2_HELD_OUT", p2)):
        ia = panel.select_weeks(WIN)
        ib = panel.select_weeks(set(common) - WIN)
        rng = np.random.RandomState(ss.SEED + 1)
        raw = ss.raw_measure(panel, ia, ib, K, rng, N_NULL, N_TEST)
        rng = np.random.RandomState(ss.SEED + 2)
        ctl = ss.controlled_measure(panel, ia, ib, K, rng, N_NULL, N_TEST)
        rec = gate["confirmation"][label]
        for nm, got, want in (
                (f"{label}.raw.observed_mean", raw["observed_mean"],
                 rec["raw"]["observed_mean"]),
                (f"{label}.raw.bar", raw["bar"], rec["raw"]["bar"]),
                (f"{label}.ctl.observed_mean", ctl["observed_mean"],
                 rec["size_controlled"]["observed_mean"]),
                (f"{label}.ctl.bar", ctl["bar"], rec["size_controlled"]["bar"])):
            d = abs(got - want)
            worst = max(worst, d)
            checks[nm] = {"recomputed": got, "recorded": want, "abs_diff": d}
            print(f"  {'ok ' if d <= TOL else 'FAIL'} {nm:<38} "
                  f"{got:.9f} vs {want:.9f}  d={d:.2e}")
    ok = worst <= TOL
    print(f"\n  worst |diff| = {worst:.2e}  (spec <= {TOL:.0e})  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok, {"worst_abs_diff": worst, "PASS": ok, "checks": checks}


def gate_fixtures(panel, K):
    """G2 null: halves of the same window must not fire.
       G3 planted: an injected category shift must fire."""
    hr()
    print("G2/G3  DETECTOR FIXTURES on the pairwise (window-vs-window) usage")
    hr()
    out = {}
    idx = panel.select_weeks(WINDOWS[FROZEN])

    # -- G2: same-window split ---------------------------------------
    rng = np.random.RandomState(SEED + 11)
    order = rng.permutation(idx)
    h = len(order) // 2
    a, b = np.sort(order[:h]), np.sort(order[h:])
    r_raw = ss.raw_measure(panel, a, b, K, np.random.RandomState(SEED + 12),
                           N_NULL, N_TEST)
    r_ctl = ss.controlled_measure(panel, a, b, K,
                                  np.random.RandomState(SEED + 13),
                                  N_NULL, N_TEST)
    g2 = (r_raw["fires"] <= N_TEST // 2) and (r_ctl["fires"] <= N_TEST // 2)
    print(f"  G2 same-window halves   raw {r_raw['fires']}/{r_raw['n_draws']}"
          f"  controlled {r_ctl['fires']}/{r_ctl['n_draws']}"
          f"   -> {'PASS' if g2 else 'FAIL'} (must NOT fire)")
    out["G2_same_window_null"] = {
        "raw_fires": r_raw["fires"], "raw_rate": r_raw["fire_rate"],
        "ctl_fires": r_ctl["fires"], "ctl_rate": r_ctl["fire_rate"],
        "PASS": bool(g2)}

    # -- G3: planted shift -------------------------------------------
    # Move 30% of one half's items into category 0. A real detector must see it.
    rng = np.random.RandomState(SEED + 14)
    codes = panel.codes.copy()
    mask = np.zeros(len(codes), bool)
    for i in b:
        mask[panel.starts[i]:panel.ends[i]] = True
    hit = np.flatnonzero(mask)
    codes[rng.choice(hit, int(0.30 * len(hit)), replace=False)] = 0
    planted = panel.copy_with_codes(codes)
    p_raw = ss.raw_measure(planted, a, b, K, np.random.RandomState(SEED + 15),
                           N_NULL, N_TEST)
    p_ctl = ss.controlled_measure(planted, a, b, K,
                                  np.random.RandomState(SEED + 16),
                                  N_NULL, N_TEST)
    g3 = (p_raw["fires"] > N_TEST // 2) and (p_ctl["fires"] > N_TEST // 2)
    print(f"  G3 planted 30% shift    raw {p_raw['fires']}/{p_raw['n_draws']}"
          f"  controlled {p_ctl['fires']}/{p_ctl['n_draws']}"
          f"   -> {'PASS' if g3 else 'FAIL'} (MUST fire)")
    out["G3_planted_shift"] = {
        "raw_fires": p_raw["fires"], "raw_rate": p_raw["fire_rate"],
        "ctl_fires": p_ctl["fires"], "ctl_rate": p_ctl["fire_rate"],
        "PASS": bool(g3)}
    out["PASS"] = bool(g2 and g3)
    return bool(g2 and g3), out


# ══════════════════════════════════════════════════════════════════
# Q1  PAIRWISE WINDOW DISCRIMINATION
# ══════════════════════════════════════════════════════════════════
def q1_pairwise(panels, K):
    hr()
    print("Q1  ARE THE WINDOWS DISTINCT FROM EACH OTHER?")
    hr()
    names = list(WINDOWS)
    res = {}
    for ylab, panel in panels:
        print(f"\n  --- {ylab} ---")
        print(f"  {'pair':<18}{'raw TVD':>10}{'bar':>9}{'fires':>8}"
              f"{'ctl TVD':>10}{'bar':>9}{'fires':>8}")
        res[ylab] = {}
        for i, j in itertools.combinations(range(len(names)), 2):
            na, nb = names[i], names[j]
            ia = panel.select_weeks(WINDOWS[na])
            ib = panel.select_weeks(WINDOWS[nb])
            raw = ss.raw_measure(panel, ia, ib, K,
                                 np.random.RandomState(SEED + 100 + 7 * i + j),
                                 N_NULL, N_TEST)
            ctl = ss.controlled_measure(panel, ia, ib, K,
                                        np.random.RandomState(SEED + 200 + 7 * i + j),
                                        N_NULL, N_TEST)
            key = f"{na}_vs_{nb}"
            res[ylab][key] = {
                "involves_frozen": FROZEN in (na, nb),
                "raw": raw, "size_controlled": ctl}
            print(f"  {key:<18}{raw['observed_mean']:>10.5f}{raw['bar']:>9.5f}"
                  f"{raw['fires']:>6}/{raw['n_draws']:<2}"
                  f"{ctl['observed_mean']:>10.5f}{ctl['bar']:>9.5f}"
                  f"{ctl['fires']:>6}/{ctl['n_draws']:<2}")
    return res


# ══════════════════════════════════════════════════════════════════
# Q2  TRANSFER
# ══════════════════════════════════════════════════════════════════
# BASIS NOTE -- two earlier bases were tried and BOTH are contaminated:
#   (a) profile = window - "all other usable weeks". Every window sits inside
#       every other window's baseline, which ANTI-correlates the profiles by
#       construction (measured cross-window rho -0.15).
#   (b) profile = window - "a fixed disjoint baseline (weeks 16,47-50)". Now
#       all five profiles share the same 'mid-year vs boundary-weeks' contrast,
#       which CO-correlates them. Its own chance floor measured +0.65, ABOVE
#       the cross-year value -- proof the basis, not the signal, dominated.
# The basis used below is symmetric: each window is centred on the MEAN across
# the five windows, so no window is privileged. Centring forces the vectors to
# sum to zero, which induces a structural negative correlation of about
# -1/(n_windows-1); that bias is not assumed, it is MEASURED by a
# label-permutation null that goes through the identical pipeline.


def _common_bin_plan(panel, idx_by_window, halves=1):
    """k baskets per size-bin, drawable from EVERY window.
    halves=2 reserves two disjoint draws per bin (for the ceiling split)."""
    bins_by_w = {w: panel.bins[ix] for w, ix in idx_by_window.items()}
    allbins = sorted(set.intersection(*[set(np.unique(b)) for b in bins_by_w.values()]))
    plan = {}
    for s in allbins:
        k = min(int((b == s).sum()) // halves for b in bins_by_w.values())
        if k > 0:
            plan[int(s)] = k
    return plan


def _dist(panel, idx, plan, rng, K, reps=5):
    """Size-matched category distribution of a basket group."""
    bins = panel.bins[idx]
    pools = {s: idx[bins == s] for s in plan if (bins == s).any()}
    if len(pools) != len(plan):
        return None
    acc = np.zeros(K)
    for _ in range(reps):
        picks = [pools[s][rng.choice(len(pools[s]), k, replace=False)]
                 for s, k in plan.items()]
        items = panel.items_of(np.concatenate(picks))
        acc += np.bincount(items, minlength=K) / len(items)
    return acc / reps


def _centred_profiles(panel, idx_by_window, plan, rng, K, reps=5):
    """p_w minus the mean over all windows -- symmetric, no privileged base."""
    d = {w: _dist(panel, ix, plan, rng, K, reps) for w, ix in idx_by_window.items()}
    if any(v is None for v in d.values()):
        return None
    m = np.mean(list(d.values()), axis=0)
    return {w: d[w] - m for w in d}


def q2_transfer(panels, K, common, cats):
    hr()
    print("Q2  DOES A PROFILE FROM ONE WINDOW TRANSFER TO ANOTHER?")
    hr()
    names = list(WINDOWS)
    idx_by = {ylab: {w: panel.select_weeks(WINDOWS[w]) for w in names}
              for ylab, panel in panels}

    # one common size-bin plan per year, shared by every window AND by the
    # permutation null, so nothing differs but the labelling
    prof, plans = {}, {}
    for ylab, panel in panels:
        plans[ylab] = _common_bin_plan(panel, idx_by[ylab])
        p = _centred_profiles(panel, idx_by[ylab], plans[ylab],
                              np.random.RandomState(SEED + 300), K)
        for w in names:
            prof[(ylab, w)] = p[w]
        print(f"  {ylab}: {len(plans[ylab])} size bins, "
              f"{sum(plans[ylab].values()):,} baskets drawn per window")

    # ---- CEILING: same window, two disjoint halves, same year -----------
    ylab0, panel0 = panels[0]
    rng = np.random.RandomState(SEED + 400)
    halfA, halfB = {}, {}
    for w in names:
        ix = idx_by[ylab0][w]
        o = rng.permutation(ix)
        halfA[w], halfB[w] = np.sort(o[:len(o) // 2]), np.sort(o[len(o) // 2:])
    # budget must be feasible in BOTH halves of EVERY window, so derive it
    # from the half-groups themselves rather than from full-window counts
    plan_h = _common_bin_plan(
        panel0, {**{f"{w}_A": halfA[w] for w in names},
                 **{f"{w}_B": halfB[w] for w in names}})
    pA = _centred_profiles(panel0, halfA, plan_h,
                           np.random.RandomState(SEED + 401), K)
    pB = _centred_profiles(panel0, halfB, plan_h,
                           np.random.RandomState(SEED + 402), K)
    rel = ({w: float(spearmanr(pA[w], pB[w]).statistic) for w in names}
           if pA and pB else {})

    # ---- REPLICATION: same window, year 1 vs year 2 ---------------------
    cross_year = {w: float(spearmanr(prof[("year1", w)],
                                     prof[("year2", w)]).statistic)
                  for w in names}

    # ---- OBSERVED: different windows, same year -------------------------
    cross_win = {}
    for ylab, _ in panels:
        for a, b in itertools.combinations(names, 2):
            cross_win[f"{ylab}:{a}_vs_{b}"] = float(
                spearmanr(prof[(ylab, a)], prof[(ylab, b)]).statistic)

    # ---- FLOOR: LABEL-PERMUTATION null ----------------------------------
    # Within each size bin, randomly re-partition the pooled baskets of all
    # five windows into five pseudo-windows of identical composition, then run
    # the IDENTICAL centring and correlation. This destroys only the temporal
    # labelling, so it prices both sampling noise and the sum-to-zero bias.
    floor = []
    for t in range(20):
        rng = np.random.RandomState(SEED + 800 + t)
        pseudo = {w: [] for w in names}
        for s in plans[ylab0]:
            pooled = np.concatenate([idx_by[ylab0][w][panel0.bins[idx_by[ylab0][w]] == s]
                                     for w in names])
            pooled = rng.permutation(pooled)
            per = len(pooled) // len(names)
            for i, w in enumerate(names):
                pseudo[w].append(pooled[i * per:(i + 1) * per])
        pseudo = {w: np.sort(np.concatenate(v)) for w, v in pseudo.items()}
        pp = _centred_profiles(panel0, pseudo, plans[ylab0],
                               np.random.RandomState(SEED + 850 + t), K)
        if pp is None:
            continue
        floor += [float(spearmanr(pp[a], pp[b]).statistic)
                  for a, b in itertools.combinations(names, 2)]
    floor = np.array(floor)

    fl = float(floor.mean()) if len(floor) else float("nan")
    cw = np.array(list(cross_win.values()))
    cy = float(np.mean(list(cross_year.values())))
    rl = float(np.mean(list(rel.values()))) if rel else float("nan")
    denom = cy - fl
    ti = float((cw.mean() - fl) / denom) if denom else float("nan")

    print(f"\n  {'window':<10}{'self-reliability':>19}{'cross-YEAR (y1 vs y2)':>24}")
    for w in names:
        star = "  <- frozen" if w == FROZEN else ""
        print(f"  {w:<10}{rel.get(w, float('nan')):>19.4f}"
              f"{cross_year[w]:>24.4f}{star}")

    print(f"\n  BASIS: each window centred on the mean of all five windows "
          f"(symmetric).")
    print(f"    FLOOR   label-permutation null (n={len(floor)} pairs) "
          f": {fl:+.4f}  (sd {floor.std(ddof=1) if len(floor) > 1 else 0:.4f})")
    print(f"    CEILING same window, disjoint halves                "
          f": {rl:+.4f}")
    print(f"    same window, year1 vs year2 (replication)           "
          f": {cy:+.4f}")
    print(f"    DIFFERENT windows, same year (transfer)             "
          f": {cw.mean():+.4f}  (sd {cw.std(ddof=1):.4f}, n={len(cw)})")
    print(f"\n    TRANSFER INDEX (transfer - floor)/(replication - floor) "
          f"= {ti:.3f}")
    print(f"      1.0 = transfers as well as it replicates;  0.0 = no better "
          f"than a permuted label")

    froz = {k: v for k, v in cross_win.items() if FROZEN in k}
    quiet = {k: v for k, v in cross_win.items() if FROZEN not in k}
    print(f"\n    pairs involving {FROZEN}: mean "
          f"{np.mean(list(froz.values())):+.4f} (n={len(froz)})")
    print(f"    pairs not involving it : mean "
          f"{np.mean(list(quiet.values())):+.4f} (n={len(quiet)})")

    rows = []
    for (ylab, w), v in prof.items():
        for c, val in zip(cats, v):
            rows.append({"year": ylab, "window": w, "category": c,
                         "centred_shift_pp": val * 100})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    return {"basis": "each window centred on the mean across all five windows; "
                     "floor measured by label permutation through the same "
                     "pipeline",
            "self_reliability": rel,
            "cross_year_same_window": cross_year,
            "cross_window_same_year": cross_win,
            "floor_draws": [float(x) for x in floor],
            "FLOOR_label_permutation": fl,
            "CEILING_self_reliability": rl,
            "replication_cross_year": cy,
            "transfer_cross_window": float(cw.mean()),
            "transfer_cross_window_sd": float(cw.std(ddof=1)),
            "TRANSFER_INDEX": ti,
            "TRANSFER_INDEX_definition":
                "(transfer - floor)/(replication - floor); 1.0 = transfers as "
                "well as it replicates, 0.0 = chance",
            "mean_pairs_involving_frozen": float(np.mean(list(froz.values()))),
            "mean_pairs_not_involving_frozen": float(np.mean(list(quiet.values()))),
            }


# ══════════════════════════════════════════════════════════════════
def main():
    hr()
    print("1A  CONDITIONING EFFECTIVENESS -- WINDOW DISCRIMINATION & TRANSFER")
    hr()
    df, p1, p2, K, common, cutoff, dropped, dips = ss.load_panels()
    cats = sorted(pd.Categorical(
        pd.read_csv(os.path.join("data", "dunnhumby", "dj_items.csv"),
                    usecols=["category"])["category"]).categories)

    bad = {w: sorted(v - set(common)) for w, v in WINDOWS.items()
           if not v <= set(common)}
    if bad:
        print(f"\n  FAIL: windows fall outside the measured usable range: {bad}")
        sys.exit(1)
    print(f"\n  windows (all inside usable weeks {min(common)}..{max(common)}): "
          f"{', '.join(f'{w}' for w in WINDOWS)}")

    g1_ok, g1 = gate_known_answer(p1, p2, K, common)
    g23_ok, g23 = gate_fixtures(p1, K)
    if not (g1_ok and g23_ok):
        print("\n  GATES FAILED -- no verdict computed.")
        json.dump({"gates": {"known_answer": g1, "fixtures": g23},
                   "VERDICT": "GATES FAILED"},
                  open(OUT_JSON, "w"), indent=2)
        sys.exit(1)
    print("\n  ALL GATES PASSED\n")
    if "--gate-only" in sys.argv:
        print("  --gate-only: stopping before the verdict.")
        return

    panels = [("year1", p1), ("year2", p2)]
    q1 = q1_pairwise(panels, K)
    q2 = q2_transfer(panels, K, common, cats)

    # ---- verdicts -------------------------------------------------
    hr()
    print("VERDICT")
    hr()
    n_pairs = len(q1["year2"])
    fired_both = [k for k in q1["year2"]
                  if q1["year1"][k]["size_controlled"]["effect"]
                  and q1["year2"][k]["size_controlled"]["effect"]]
    froz_pairs = [k for k in q1["year2"] if q1["year2"][k]["involves_frozen"]]
    froz_fired = [k for k in froz_pairs
                  if q1["year2"][k]["size_controlled"]["effect"]]
    quiet_pairs = [k for k in q1["year2"] if not q1["year2"][k]["involves_frozen"]]
    quiet_fired = [k for k in quiet_pairs
                   if q1["year2"][k]["size_controlled"]["effect"]]

    print(f"  Q1 pairwise discrimination (size-controlled, BOTH years):")
    print(f"     {len(fired_both)}/{n_pairs} window pairs are distinguishable "
          f"in both years")
    print(f"     pairs involving {FROZEN}: {len(froz_fired)}/{len(froz_pairs)} fire")
    print(f"     pairs not involving it : {len(quiet_fired)}/{len(quiet_pairs)} fire")

    ti = q2["TRANSFER_INDEX"]
    print(f"\n  Q2 transfer  (windows centred on their common mean; floor "
          f"measured by\n      label permutation through the identical "
          f"pipeline):")
    print(f"     FLOOR   (permuted labels)                    "
          f": {q2['FLOOR_label_permutation']:+.4f}")
    print(f"     CEILING (same window, disjoint halves)       "
          f": {q2['CEILING_self_reliability']:+.4f}")
    print(f"     replication (same window, year1 vs year2)    "
          f": {q2['replication_cross_year']:+.4f}")
    print(f"     transfer   (different windows, same year)    "
          f": {q2['transfer_cross_window']:+.4f}")
    print(f"     TRANSFER INDEX                               : {ti:.3f}")
    if ti < 0.25:
        tv = "does NOT transfer"
    elif ti < 0.75:
        tv = "transfers only PARTIALLY"
    else:
        tv = "DOES transfer"
    transfers = bool(ti >= 0.75)
    print(f"     -> a profile learned on one window {tv} to another")

    out = {
        "purpose": "conditioning effectiveness: window-vs-window discrimination "
                   "and cross-window transfer",
        "retrained_anything": False,
        "windows": {k: [min(v), max(v)] for k, v in WINDOWS.items()},
        "frozen_window": FROZEN,
        "usable_weeks": [int(min(common)), int(max(common))],
        "n_categories": int(K),
        "seed": SEED, "n_null_draws": N_NULL, "n_test_draws": N_TEST,
        "percentile": ss.PCTL,
        "gates": {"known_answer": g1, "fixtures": g23},
        "Q1_pairwise": q1,
        "Q2_transfer": q2,
        "verdict": {
            "n_pairs": n_pairs,
            "pairs_distinguishable_both_years": len(fired_both),
            "frozen_pairs_fired": f"{len(froz_fired)}/{len(froz_pairs)}",
            "nonfrozen_pairs_fired": f"{len(quiet_fired)}/{len(quiet_pairs)}",
            "BASIS": q2["basis"],
            "floor_label_permutation": q2["FLOOR_label_permutation"],
            "ceiling_self_reliability": q2["CEILING_self_reliability"],
            "replication_cross_year": q2["replication_cross_year"],
            "transfer_cross_window": q2["transfer_cross_window"],
            "TRANSFER_INDEX": ti,
            "transfer_verdict": tv,
            "profile_transfers_across_windows": bool(transfers),
        },
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=float)
    print(f"\n  -> {OUT_JSON}")
    print(f"  -> {OUT_CSV}")


if __name__ == "__main__":
    main()
