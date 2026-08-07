"""LOCATION CONDITIONING, tested with the best available proxy: STORE IDENTITY.

THE GAP THIS CLOSES
-------------------
The brief asks for conditioning on "date and location", so that the generator can
produce "order patterns specific to parts of the world". Location was recorded
as NOT ATTEMPTED because no public order dataset carries geography -- which is
true of coordinates, but Dunnhumby does carry **`store_id` (582 stores)**, and a
store IS a place. Different stores serve different neighbourhoods, so if location
carries any demand signal at all it must show up as between-store differences in
category mix.

This does NOT recover geography: stores cannot be placed on a map, ranked by
distance, or grouped into regions, so nothing here supports a claim about
"parts of the world". What it CAN settle is the prior question -- **is there a
location signal to condition on at all, and does a profile fitted at one
location transfer to another?** That is exactly the pair of questions
`dunnhumby_window_discrimination.py` asks of the seasonal axis, so the same
machinery answers both and the two axes become directly comparable.

METHOD -- identical to the seasonal test, imported not copied
-------------------------------------------------------------
Detector (`raw_measure`, `controlled_measure`, `Panel`) comes from
`dunnhumby_signal_search.py`; the centred-profile and permutation-null helpers
come from `dunnhumby_window_discrimination.py`. Both are import-safe (guarded
entry points, only an idempotent makedirs at module level).

THE BASKET-SIZE CONTROL IS LOAD-BEARING HERE. Stores differ in format -- a
convenience store and a superstore have very different basket sizes -- so a raw
category-mix difference between stores is exactly the confound fixture F3 was
built for. A store difference that dies under the size control is a format
difference, not a location signal.

STORES ARE SELECTED BY VOLUME, NOT BY EFFECT. The five highest-basket-count
stores are taken before any measurement, so there is no selection on the
outcome (the defect that made the original seasonal window need a frozen,
year-1-only discovery step).

READS  : data/dunnhumby/dj_items.csv
         results/dunnhumby/signal_search.json   (usable weeks; known-answer ref)
WRITES : results/dunnhumby/store_discrimination.json          (new)
         results/dunnhumby/store_shift_profiles.csv           (new)

Usage:
  python dunnhumby_store_discrimination.py [--gate-only] [--n-stores 5]
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import dunnhumby_signal_search as ss
import dunnhumby_window_discrimination as wd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_DIR = os.path.join("results", "dunnhumby")
OUT_JSON = os.path.join(RESULTS_DIR, "store_discrimination.json")
OUT_CSV = os.path.join(RESULTS_DIR, "store_shift_profiles.csv")

SEED = 20260801
N_NULL = ss.N_NULL_DRAWS
N_TEST = ss.N_TEST_DRAWS


def hr(c="="):
    print(c * 70)


class StorePanel(ss.Panel):
    """`ss.Panel` keyed by store instead of week. Panel.__init__ sorts by
    basket_id and takes one row per basket via `starts`; the store label is
    taken the same way so it stays aligned with `sizes`/`bins`."""

    def __init__(self, df, K):
        super().__init__(df, K)
        d = df.sort_values("basket_id", kind="mergesort")
        self.store = d["store_id"].to_numpy()[self.starts]

    def select_store(self, s):
        return np.flatnonzero(self.store == s)


def load(n_stores):
    df = pd.read_csv(os.path.join("data", "dunnhumby", "dj_items.csv"))
    gate = json.load(open(os.path.join(RESULTS_DIR, "signal_search.json")))
    lo, hi = gate["usable_weeks"]
    df = df[df["week_of_year"].between(lo, hi)]
    cats = pd.Categorical(df["category"])
    df = df.assign(code=cats.codes)
    K = len(cats.categories)
    print(f"  {len(df):,} items   {df['basket_id'].nunique():,} baskets   "
          f"{K} categories   usable weeks {lo}..{hi}")

    counts = df.groupby("store_id")["basket_id"].nunique().sort_values(
        ascending=False)
    stores = [int(s) for s in counts.index[:n_stores]]
    print(f"  {len(counts)} stores; taking the {n_stores} highest-volume "
          f"(selected on VOLUME, before any measurement):")
    for s in stores:
        sub = df[df["store_id"] == s]
        print(f"    store {s:<8} {counts[s]:>6,} baskets  "
              f"{len(sub):>8,} items  mean basket "
              f"{len(sub) / counts[s]:.2f}")
    panel = StorePanel(df, K)
    return df, panel, K, stores, list(cats.categories), (lo, hi)


def gates(panel, K, stores):
    hr()
    print("GATES")
    hr()
    idx = panel.select_store(stores[0])
    rng = np.random.RandomState(SEED + 1)
    o = rng.permutation(idx)
    a, b = np.sort(o[:len(o) // 2]), np.sort(o[len(o) // 2:])
    r_raw = ss.raw_measure(panel, a, b, K, np.random.RandomState(SEED + 2),
                           N_NULL, N_TEST)
    r_ctl = ss.controlled_measure(panel, a, b, K,
                                  np.random.RandomState(SEED + 3), N_NULL, N_TEST)
    g1 = r_raw["fires"] <= N_TEST // 2 and r_ctl["fires"] <= N_TEST // 2
    print(f"  G1 same-store halves    raw {r_raw['fires']}/{r_raw['n_draws']}"
          f"  controlled {r_ctl['fires']}/{r_ctl['n_draws']}"
          f"   -> {'PASS' if g1 else 'FAIL'} (must NOT fire)")

    rng = np.random.RandomState(SEED + 4)
    codes = panel.codes.copy()
    mask = np.zeros(len(codes), bool)
    for i in b:
        mask[panel.starts[i]:panel.ends[i]] = True
    hit = np.flatnonzero(mask)
    codes[rng.choice(hit, int(0.30 * len(hit)), replace=False)] = 0
    planted = panel.copy_with_codes(codes)
    p_raw = ss.raw_measure(planted, a, b, K, np.random.RandomState(SEED + 5),
                           N_NULL, N_TEST)
    p_ctl = ss.controlled_measure(planted, a, b, K,
                                  np.random.RandomState(SEED + 6), N_NULL, N_TEST)
    g2 = p_raw["fires"] > N_TEST // 2 and p_ctl["fires"] > N_TEST // 2
    print(f"  G2 planted 30% shift    raw {p_raw['fires']}/{p_raw['n_draws']}"
          f"  controlled {p_ctl['fires']}/{p_ctl['n_draws']}"
          f"   -> {'PASS' if g2 else 'FAIL'} (MUST fire)")
    return g1 and g2, {
        "G1_same_store_null": {"raw_fires": r_raw["fires"],
                               "ctl_fires": r_ctl["fires"], "PASS": bool(g1)},
        "G2_planted_shift": {"raw_fires": p_raw["fires"],
                             "ctl_fires": p_ctl["fires"], "PASS": bool(g2)},
        "PASS": bool(g1 and g2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stores", type=int, default=5)
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()

    hr()
    print("LOCATION CONDITIONING via STORE IDENTITY -- discrimination & transfer")
    hr()
    df, panel, K, stores, cats, weeks = load(a.n_stores)

    ok, gate_rec = gates(panel, K, stores)
    if not ok:
        print("\n  GATES FAILED -- no verdict computed.")
        json.dump({"gates": gate_rec, "VERDICT": "GATES FAILED"},
                  open(OUT_JSON, "w"), indent=2)
        sys.exit(1)
    print("\n  ALL GATES PASSED")
    if a.gate_only:
        return

    # ── Q1 pairwise discrimination ────────────────────────────────
    hr()
    print("Q1  ARE STORES DISTINGUISHABLE FROM EACH OTHER?")
    hr()
    print(f"  {'pair':<22}{'raw TVD':>10}{'bar':>9}{'fires':>8}"
          f"{'ctl TVD':>10}{'bar':>9}{'fires':>8}")
    q1 = {}
    for i, j in itertools.combinations(range(len(stores)), 2):
        ia = panel.select_store(stores[i])
        ib = panel.select_store(stores[j])
        raw = ss.raw_measure(panel, ia, ib, K,
                             np.random.RandomState(SEED + 100 + 7 * i + j),
                             N_NULL, N_TEST)
        ctl = ss.controlled_measure(panel, ia, ib, K,
                                    np.random.RandomState(SEED + 200 + 7 * i + j),
                                    N_NULL, N_TEST)
        if raw is None or ctl is None:
            continue
        key = f"{stores[i]}_vs_{stores[j]}"
        q1[key] = {"raw": raw, "size_controlled": ctl}
        print(f"  {key:<22}{raw['observed_mean']:>10.5f}{raw['bar']:>9.5f}"
              f"{raw['fires']:>6}/{raw['n_draws']:<2}"
              f"{ctl['observed_mean']:>10.5f}{ctl['bar']:>9.5f}"
              f"{ctl['fires']:>6}/{ctl['n_draws']:<2}")

    n_fire_raw = sum(v["raw"]["effect"] for v in q1.values())
    n_fire_ctl = sum(v["size_controlled"]["effect"] for v in q1.values())
    print(f"\n  raw: {n_fire_raw}/{len(q1)} pairs distinguishable   "
          f"size-controlled: {n_fire_ctl}/{len(q1)}")
    if n_fire_raw > n_fire_ctl:
        print(f"  -> {n_fire_raw - n_fire_ctl} pair(s) are a STORE-FORMAT "
              f"(basket-size) difference, not a location signal")

    # ── Q2 transfer, same basis as the seasonal test ──────────────
    hr()
    print("Q2  DOES A PROFILE FROM ONE STORE TRANSFER TO ANOTHER?")
    hr()
    idx_by = {s: panel.select_store(s) for s in stores}
    plan = wd._common_bin_plan(panel, idx_by)
    prof = wd._centred_profiles(panel, idx_by, plan,
                                np.random.RandomState(SEED + 300), K)
    print(f"  {len(plan)} size bins, {sum(plan.values()):,} baskets per store")

    rng = np.random.RandomState(SEED + 400)
    hA, hB = {}, {}
    for s in stores:
        o = rng.permutation(idx_by[s])
        hA[s], hB[s] = np.sort(o[:len(o) // 2]), np.sort(o[len(o) // 2:])
    plan_h = wd._common_bin_plan(panel, {**{f"{s}A": hA[s] for s in stores},
                                         **{f"{s}B": hB[s] for s in stores}})
    pA = wd._centred_profiles(panel, hA, plan_h,
                              np.random.RandomState(SEED + 401), K)
    pB = wd._centred_profiles(panel, hB, plan_h,
                              np.random.RandomState(SEED + 402), K)
    rel = {s: float(spearmanr(pA[s], pB[s]).statistic) for s in stores} \
        if pA and pB else {}

    cross = {f"{x}_vs_{y}": float(spearmanr(prof[x], prof[y]).statistic)
             for x, y in itertools.combinations(stores, 2)}

    floor = []
    for t in range(20):
        r = np.random.RandomState(SEED + 800 + t)
        pseudo = {s: [] for s in stores}
        for b in plan:
            pooled = np.concatenate([idx_by[s][panel.bins[idx_by[s]] == b]
                                     for s in stores])
            pooled = r.permutation(pooled)
            per = len(pooled) // len(stores)
            for n, s in enumerate(stores):
                pseudo[s].append(pooled[n * per:(n + 1) * per])
        pseudo = {s: np.sort(np.concatenate(v)) for s, v in pseudo.items()}
        pp = wd._centred_profiles(panel, pseudo, plan,
                                  np.random.RandomState(SEED + 850 + t), K)
        if pp is None:
            continue
        floor += [float(spearmanr(pp[x], pp[y]).statistic)
                  for x, y in itertools.combinations(stores, 2)]
    floor = np.array(floor)

    fl, rl = float(floor.mean()), (float(np.mean(list(rel.values())))
                                   if rel else float("nan"))
    cv = np.array(list(cross.values()))
    ti = float((cv.mean() - fl) / (rl - fl)) if (rl - fl) else float("nan")

    print(f"\n  {'store':<10}{'self-reliability':>19}")
    for s in stores:
        print(f"  {s:<10}{rel.get(s, float('nan')):>19.4f}")
    print(f"\n    FLOOR   label-permutation null (n={len(floor)} pairs) "
          f": {fl:+.4f}")
    print(f"    CEILING same store, disjoint halves                "
          f": {rl:+.4f}")
    print(f"    transfer BETWEEN stores                            "
          f": {cv.mean():+.4f}  (sd {cv.std(ddof=1):.4f}, n={len(cv)})")
    print(f"\n    TRANSFER INDEX = {ti:.3f}   "
          f"(1.0 = transfers as well as it replicates, 0.0 = chance)")

    # ── Q3 cross-YEAR replication, the analogue of the seasonal test's
    #    strongest evidence. Without it a sceptic can say stores differ only
    #    because different households happen to shop there; a profile that
    #    reproduces itself a year later is a stable property of the location.
    hr()
    print("Q3  DOES A STORE'S PROFILE REPLICATE A YEAR LATER?")
    hr()
    yr = df.sort_values("basket_id", kind="mergesort")["year"].to_numpy()[
        panel.starts]
    cross_year = {}
    idx_y = {}
    for s in stores:
        for y in (1, 2):
            idx_y[(s, y)] = np.flatnonzero((panel.store == s) & (yr == y))
    if all(len(v) > 50 for v in idx_y.values()):
        plan_y = wd._common_bin_plan(
            panel, {f"{s}_{y}": idx_y[(s, y)] for s in stores for y in (1, 2)})
        py = {}
        for y in (1, 2):
            py[y] = wd._centred_profiles(
                panel, {s: idx_y[(s, y)] for s in stores}, plan_y,
                np.random.RandomState(SEED + 500 + y), K)
        if all(v is not None for v in py.values()):
            cross_year = {s: float(spearmanr(py[1][s], py[2][s]).statistic)
                          for s in stores}
            print(f"  {sum(plan_y.values()):,} baskets per store-year")
            print(f"  {'store':<10}{'year1 vs year2':>17}")
            for s in stores:
                print(f"  {s:<10}{cross_year[s]:>17.4f}")
            cy = float(np.mean(list(cross_year.values())))
            print(f"\n    mean cross-year replication {cy:+.4f}   "
                  f"vs FLOOR {fl:+.4f}   vs BETWEEN-store {cv.mean():+.4f}")
            print(f"    -> a store's profile {'IS' if cy > 0.5 * rl else 'is NOT'} "
                  f"stable across years, while remaining distinct from other "
                  f"stores")
    else:
        print("  skipped: a store-year cell has too few baskets")

    verdict = ("location signal EXISTS and is store-specific"
               if n_fire_ctl > len(q1) // 2 and ti < 0.5 else
               "location signal EXISTS and largely TRANSFERS between stores"
               if n_fire_ctl > len(q1) // 2 else
               "NO location signal above the measured bar")
    print(f"\n  VERDICT: {verdict}")

    rows = [{"store": s, "category": c, "centred_shift_pp": v * 100}
            for s in stores for c, v in zip(cats, prof[s])]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    json.dump({
        "purpose": "location conditioning tested with store identity as the "
                   "available proxy; NOT geography -- stores cannot be placed "
                   "on a map, so this settles whether a location signal exists "
                   "and transfers, not anything about 'parts of the world'",
        "usable_weeks": list(weeks), "n_categories": int(K),
        "stores": stores, "store_selection": "highest basket count, chosen "
                                             "before any measurement",
        "seed": SEED, "gates": gate_rec,
        "Q1_pairwise": q1,
        "Q1_summary": {"pairs": len(q1), "fire_raw": int(n_fire_raw),
                       "fire_size_controlled": int(n_fire_ctl)},
        "Q2_transfer": {
            "self_reliability": rel, "cross_store": cross,
            "floor_draws": [float(x) for x in floor],
            "FLOOR_label_permutation": fl,
            "CEILING_self_reliability": rl,
            "transfer_between_stores": float(cv.mean()),
            "TRANSFER_INDEX": ti},
        "Q3_cross_year_replication": {
            "per_store": cross_year,
            "mean": (float(np.mean(list(cross_year.values())))
                     if cross_year else None),
            "NOTE": "the analogue of the seasonal test's cross-year "
                    "replication; a profile that reproduces itself a year "
                    "later is a stable property of the location, not an "
                    "artifact of which households shopped that year"},
        "VERDICT": verdict,
        "COMPARABILITY": "the size-controlled TVDs here are on the same "
                         "detector and basis as dunnhumby_window_discrimination"
                         ".py, so the location and seasonal axes can be "
                         "compared directly by magnitude",
    }, open(OUT_JSON, "w"), indent=2, default=float)
    print(f"\n  -> {OUT_JSON}\n  -> {OUT_CSV}")


if __name__ == "__main__":
    main()
