"""Does the seasonal signal survive INSIDE a single store?

WHY THIS IS NECESSARY (and only became obvious after the location result)
-------------------------------------------------------------------------
`dunnhumby_store_discrimination.py` found that stores differ in category mix by
**~2.2x more** than seasonal windows do. That makes STORE COMPOSITION a live
confound for the seasonal gate: if the set of stores trading (or their relative
volumes) drifts between the frozen window and the baseline weeks, part of the
"seasonal" TVD is really a change in WHICH SHOPS the panel visited, not a change
in what people bought. The panel-ramp cutoff fixed the household-count ramp; it
did not address store mix.

The test is direct: hold the store FIXED and re-measure window-vs-baseline
inside it. A seasonal effect that survives within a single site cannot be store
composition.

Two things are reported:
  1. how much store mix actually drifts between window and baseline weeks
     (if it barely drifts, the confound was never live);
  2. the within-store seasonal fire rate, per store, raw and size-controlled.

Same detector as everything else, imported from `dunnhumby_signal_search.py`;
basket-clustered nulls, measured bars, 10-draw fire rates.

READS  : data/dunnhumby/dj_items.csv, results/dunnhumby/signal_search.json
WRITES : results/dunnhumby/seasonal_within_store.json    (new)

Usage:  python dunnhumby_seasonal_within_store.py [--n-stores 6]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import dunnhumby_signal_search as ss
from dunnhumby_store_discrimination import StorePanel

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_DIR = os.path.join("results", "dunnhumby")
OUT = os.path.join(RESULTS_DIR, "seasonal_within_store.json")
SEED = 20260802
N_NULL, N_TEST = ss.N_NULL_DRAWS, ss.N_TEST_DRAWS


def hr(c="="):
    print(c * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stores", type=int, default=6)
    a = ap.parse_args()

    hr()
    print("DOES THE SEASONAL SIGNAL SURVIVE INSIDE A SINGLE STORE?")
    hr()
    gate = json.load(open(os.path.join(RESULTS_DIR, "signal_search.json")))
    w0, w1 = gate["discovery"]["frozen_window"]
    lo, hi = gate["usable_weeks"]
    WIN = set(range(w0, w1 + 1))

    df = pd.read_csv(os.path.join("data", "dunnhumby", "dj_items.csv"))
    df = df[df["week_of_year"].between(lo, hi)]
    cats = pd.Categorical(df["category"])
    df = df.assign(code=cats.codes)
    K = len(cats.categories)
    print(f"  usable weeks {lo}..{hi}   frozen window {w0}..{w1}   "
          f"{K} categories")

    # ── 1. how much does STORE MIX drift between window and baseline? ──
    hr("-")
    print("[1] STORE-MIX DRIFT between window and baseline weeks")
    b = df.drop_duplicates("basket_id")[["basket_id", "store_id",
                                         "week_of_year"]]
    b = b.assign(period=np.where(b["week_of_year"].isin(WIN),
                                 "window", "baseline"))
    mix = (b.groupby(["period", "store_id"]).size()
           .unstack(0).fillna(0.0))
    mix = mix / mix.sum()
    drift = float(0.5 * (mix["window"] - mix["baseline"]).abs().sum())
    print(f"    stores present: {len(mix)}")
    print(f"    store-mix TVD (window vs baseline) = {drift:.5f}")
    top = (mix["window"] - mix["baseline"]).abs().sort_values(ascending=False)
    print(f"    largest single-store shifts (pp):")
    for s in top.index[:5]:
        print(f"      store {int(s):<8} {mix['baseline'][s]*100:>6.2f}% -> "
              f"{mix['window'][s]*100:>6.2f}%")
    print(f"    for scale: the seasonal CATEGORY TVD being explained is "
          f"~0.086, and between-store category TVD is ~0.17")

    # ── 1b. DECOMPOSITION: how much CATEGORY shift does the store-mix
    #        drift actually INDUCE? The 0.049 store-mix TVD and the 0.086
    #        category TVD are over DIFFERENT supports (516 stores vs 302
    #        categories) and are not comparable as numbers. What matters is
    #        the category shift attributable purely to store composition:
    #        take BASELINE-period baskets only (so season is held fixed) and
    #        reweight the stores to the WINDOW's store distribution.
    hr("-")
    print("[1b] CATEGORY SHIFT INDUCED BY STORE-MIX DRIFT ALONE")
    base_items = df[~df["week_of_year"].isin(WIN)]
    per_store = (base_items.groupby(["store_id", "code"]).size()
                 .unstack(fill_value=0))
    per_store = per_store.reindex(columns=range(K), fill_value=0)
    tot = per_store.sum(axis=1)
    keep = tot[tot >= 200].index                      # need a stable p_s
    P = (per_store.loc[keep].div(tot.loc[keep], axis=0)).to_numpy()
    w_b = mix["baseline"].reindex(keep).fillna(0.0).to_numpy()
    w_w = mix["window"].reindex(keep).fillna(0.0).to_numpy()
    cov_b, cov_w = w_b.sum(), w_w.sum()
    w_b, w_w = w_b / w_b.sum(), w_w / w_w.sum()
    cat_b, cat_w = w_b @ P, w_w @ P
    induced = float(0.5 * np.abs(cat_b - cat_w).sum())
    seasonal = gate["confirmation"]["year2_HELD_OUT"]["raw"]["observed_mean"]
    print(f"    stores with >=200 baseline items: {len(keep)} of {len(mix)}  "
          f"(covering {cov_b:.1%} of baseline / {cov_w:.1%} of window baskets)")
    print(f"    category TVD induced by store-mix drift ALONE : {induced:.5f}")
    print(f"    seasonal category TVD to be explained (gate)  : {seasonal:.5f}")
    print(f"    -> store composition accounts for at most "
          f"{induced / seasonal * 100:.1f}% of the seasonal effect")

    # ── 2. within-store seasonal effect ────────────────────────────────
    hr("-")
    print("[2] WITHIN-STORE seasonal effect (store held FIXED)")
    counts = df.groupby("store_id")["basket_id"].nunique().sort_values(
        ascending=False)
    stores = [int(s) for s in counts.index[:a.n_stores]]
    panel = StorePanel(df, K)

    print(f"  {'store':<9}{'n_win':>8}{'n_base':>8}{'raw TVD':>10}{'bar':>9}"
          f"{'fires':>8}{'ctl TVD':>10}{'bar':>9}{'fires':>8}")
    res, fired_raw, fired_ctl, tested = {}, 0, 0, 0
    for s in stores:
        idx = panel.select_store(s)
        woy = panel.woy[idx]
        ia = idx[np.isin(woy, list(WIN))]
        ib = idx[~np.isin(woy, list(WIN))]
        if len(ia) < 150 or len(ib) < 150:
            print(f"  {s:<9}{len(ia):>8}{len(ib):>8}   (too few baskets)")
            continue
        raw = ss.raw_measure(panel, ia, ib, K,
                             np.random.RandomState(SEED + s % 1000),
                             N_NULL, N_TEST)
        ctl = ss.controlled_measure(panel, ia, ib, K,
                                    np.random.RandomState(SEED + 7 + s % 1000),
                                    N_NULL, N_TEST)
        if raw is None or ctl is None:
            print(f"  {s:<9}{len(ia):>8}{len(ib):>8}   (budget too small)")
            continue
        tested += 1
        fired_raw += bool(raw["effect"])
        fired_ctl += bool(ctl["effect"])
        res[str(s)] = {"n_window_baskets": int(len(ia)),
                       "n_baseline_baskets": int(len(ib)),
                       "raw": raw, "size_controlled": ctl}
        print(f"  {s:<9}{len(ia):>8}{len(ib):>8}"
              f"{raw['observed_mean']:>10.5f}{raw['bar']:>9.5f}"
              f"{raw['fires']:>6}/{raw['n_draws']:<2}"
              f"{ctl['observed_mean']:>10.5f}{ctl['bar']:>9.5f}"
              f"{ctl['fires']:>6}/{ctl['n_draws']:<2}")

    hr()
    print("VERDICT")
    hr()
    print(f"  store-mix drift between window and baseline: TVD {drift:.5f}")
    print(f"  within-store seasonal effect fires in "
          f"{fired_ctl}/{tested} stores (size-controlled), "
          f"{fired_raw}/{tested} raw")
    # The decomposition is the decisive evidence; the within-store test is a
    # weak corroborator because per-store n is 1-2 orders of magnitude smaller
    # than the pooled gate, so its bars are ~5x looser.
    share = induced / seasonal
    if share < 0.25 and tested and fired_ctl > tested // 2:
        verdict = (f"seasonal signal is NOT store-composition drift: store mix "
                   f"induces only {share:.1%} of the seasonal category shift, "
                   f"and the effect still fires in {fired_ctl}/{tested} "
                   f"individual stores")
    elif share < 0.25:
        verdict = (f"store composition induces only {share:.1%} of the seasonal "
                   f"shift, so it is not the explanation; the within-store test "
                   f"({fired_ctl}/{tested}) is underpowered and neither "
                   f"confirms nor refutes on its own")
    else:
        verdict = (f"WARNING: store composition could account for {share:.1%} "
                   f"of the seasonal category shift -- the gate magnitude must "
                   f"be treated as partly a store-mix artifact")
    print(f"  -> {verdict}")
    print(f"\n  NOTE: within-store n is far smaller than the pooled gate "
          f"(thousands vs ~100k baskets), so the bars here are LOOSER and\n"
          f"  these TVDs are NOT comparable in magnitude with the gate's "
          f"0.086 -- read the FIRE RATES, not the numbers.")

    json.dump({
        "purpose": "rule out store-composition drift as an explanation for the "
                   "seasonal gate result -- motivated by the finding that "
                   "between-store category TVD (~0.17) is ~2.2x the seasonal "
                   "effect (~0.077)",
        "frozen_window": [w0, w1], "usable_weeks": [lo, hi],
        "store_mix_drift_tvd": drift,
        "category_tvd_induced_by_store_mix": induced,
        "seasonal_category_tvd_gate": seasonal,
        "share_of_seasonal_explained_by_store_mix": induced / seasonal,
        "n_stores_in_decomposition": int(len(keep)),
        "n_stores_present": int(len(mix)),
        "stores_tested": stores,
        "n_tested": tested,
        "fires_size_controlled": fired_ctl, "fires_raw": fired_raw,
        "per_store": res,
        "VERDICT": verdict,
        "BASIS_WARNING": "within-store sample sizes are orders of magnitude "
                         "smaller than the pooled gate, so bars are looser and "
                         "the TVD magnitudes are NOT interchangeable with the "
                         "gate's 0.08598; cite fire rates only",
    }, open(OUT, "w"), indent=2, default=float)
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
