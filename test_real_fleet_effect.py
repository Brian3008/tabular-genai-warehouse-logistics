"""
test_real_fleet_effect.py
=========================
Does basket size support a fleet claim AT ALL - on REAL
data?

WHY THIS IS THE QUESTION
------------------------
validate_demand_geometry.py found the synthetic data
misrepresents warehouse demand:

    TRAVEL GAP (large basket - small basket)
      real       0.19 aisles
      synthetic  2.36 aisles      -> 1269% of the truth

The model INVENTED a basket-size effect - the same
fabrication association_audit.py measured, showing up
where it does damage.

So the fleet chapter cannot be built on synthetic data.
BUT the real gap is only 0.19 aisles. Tiny. The question
this script answers: is there ANY basket-size fleet
effect in reality, or did CTGAN fabricate an effect from
NOTHING rather than exaggerate a real one?

DESIGN RULES (all lessons from this project's audits)
-----------------------------------------------------
1. REAL DATA ONLY. No synthetic file is read anywhere.
2. SIZE-MATCHED. One N = min(len(small), len(large))//2
   used identically in the null and the observed test.
   (TVD and Gini are sample-size dependent; a mismatched
   floor once produced a false PASS in this project.)
3. THE NULL IS MEASURED, NOT GUESSED. Split a SINGLE
   group into two halves of size N - both halves are the
   same demand type, so their difference is pure
   sampling noise. Repeat many times, alternating which
   group supplies the pool. Bar = 95th percentile.
   No hard-coded thresholds anywhere.
4. CONSISTENT ESTIMATOR NOISE. travel() uses fresh,
   independent permutation randomness for EVERY call, in
   the null and the test alike. (First draft used
   different seed schemes in null vs test - caught in
   audit.)
5. MAJORITY-VOTE INTERPRETATION. A 95th-pct bar fires
   ~5% of the time on the null BY DEFINITION; with three
   metrics that is ~14% chance of a spurious alarm on a
   single draw (caught on fixtures: identical
   distributions produced a Gini "EFFECT"). Fix: repeat
   the observed test N_TEST_DRAWS times with fresh
   subsamples; a metric claims an effect only if it
   fires on a MAJORITY of draws.
6. FIXTURE-VERIFIED. Run with --selftest first:
   Case A (identical distributions) must report NO
   effect; Case B (planted difference) must DETECT it.

Usage:
    python test_real_fleet_effect.py --selftest
    python test_real_fleet_effect.py
"""

import sys
import json
import numpy as np
import pandas as pd

SEED         = 42
N_NULL_DRAWS = 40    # draws to build the null distribution
N_TEST_DRAWS = 10    # repeated observed comparisons
TRAVEL_REPS  = 30    # permutations averaged inside travel()
PCTL         = 95    # bar = this percentile of the null

SMALL_MAX = 10       # small basket:  <= 10 items
LARGE_MIN = 14       # large basket:  >= 14 items
N_AISLES  = 134      # Instacart aisles are 1..134

EVAL_PATH  = "data/v3_compare.csv"
TRAIN_IDS  = "data/v3_train_order_ids.csv"
OUT_PATH   = "data/real_fleet_effect.json"


# ------------------------------------------------------
# metrics
# ------------------------------------------------------
def travel(aisles, rng, reps=TRAVEL_REPS):
    """Expected |aisle distance| between consecutive picks,
    averaged over `reps` random pick orders. `rng` supplies
    fresh permutation randomness on every call (rule 4)."""
    a = np.asarray(aisles, dtype=float)
    return float(np.mean([
        np.mean(np.abs(np.diff(rng.permutation(a))))
        for _ in range(reps)
    ]))


def gini(aisles):
    """Gini of demand concentration across aisles.
    0 = perfectly even, 1 = all demand on one aisle."""
    counts = np.bincount(np.asarray(aisles, dtype=int),
                         minlength=N_AISLES + 1)[1:]
    counts = np.sort(counts.astype(float))
    n = len(counts)
    cum = np.cumsum(counts)
    if cum[-1] == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def tvd(a, b):
    """Total variation distance between two aisle mixes."""
    pa = np.bincount(np.asarray(a, dtype=int),
                     minlength=N_AISLES + 1)[1:].astype(float)
    pb = np.bincount(np.asarray(b, dtype=int),
                     minlength=N_AISLES + 1)[1:].astype(float)
    pa /= pa.sum()
    pb /= pb.sum()
    return float(0.5 * np.abs(pa - pb).sum())


def metric_diffs(a, b, rng):
    """The three demand-geometry differences between two
    samples. Same function used by null and test (rule 4)."""
    return {
        "travel": abs(travel(a, rng) - travel(b, rng)),
        "gini":   abs(gini(a) - gini(b)),
        "tvd":    tvd(a, b),
    }


# ------------------------------------------------------
# core test (used by both real run and fixtures)
# ------------------------------------------------------
def run_test(small, large, rng, label=""):
    small = np.asarray(small)
    large = np.asarray(large)

    N = min(len(small), len(large)) // 2      # rule 2
    assert N >= 500, (
        f"groups too small for a meaningful test (N={N})")

    print(f"\n{label}")
    print(f"  small pool {len(small):>7,}   "
          f"large pool {len(large):>7,}   N = {N:,}")

    # ---- NULL: same-group half-splits (rule 3) ----
    pools = [small, large]
    null = {"travel": [], "gini": [], "tvd": []}
    for i in range(N_NULL_DRAWS):
        pool = pools[i % 2]                   # alternate
        idx = rng.permutation(len(pool))
        h1 = pool[idx[:N]]
        h2 = pool[idx[N:2 * N]]
        d = metric_diffs(h1, h2, rng)
        for k in null:
            null[k].append(d[k])

    bar  = {k: float(np.percentile(v, PCTL))
            for k, v in null.items()}
    mean_null = {k: float(np.mean(v))
                 for k, v in null.items()}

    print(f"  null ({N_NULL_DRAWS} same-group splits), "
          f"bar = {PCTL}th pct:")
    for k in ("travel", "gini", "tvd"):
        print(f"    {k:<7} mean {mean_null[k]:.4f}   "
              f"bar {bar[k]:.4f}")

    # ---- OBSERVED: repeated draws, majority vote (rule 5)
    fires = {k: 0 for k in bar}
    obs_all = {k: [] for k in bar}
    for _ in range(N_TEST_DRAWS):
        s = small[rng.choice(len(small), N, replace=False)]
        l = large[rng.choice(len(large), N, replace=False)]
        d = metric_diffs(s, l, rng)
        for k in bar:
            obs_all[k].append(d[k])
            if d[k] > bar[k]:
                fires[k] += 1

    obs_mean = {k: float(np.mean(v))
                for k, v in obs_all.items()}
    effect = {k: fires[k] > N_TEST_DRAWS // 2
              for k in bar}
    any_effect = any(effect.values())

    print(f"  observed small-vs-large "
          f"({N_TEST_DRAWS} fresh draws):")
    for k in ("travel", "gini", "tvd"):
        pct = 100 * fires[k] // N_TEST_DRAWS
        verdict = "EFFECT" if effect[k] else "no effect"
        print(f"    {k:<7} mean {obs_mean[k]:.4f}   "
              f"fires {pct:>3}%   -> {verdict}")
    print(f"  => ANY basket-size fleet effect: "
          f"{'YES' if any_effect else 'NO'}")

    return {
        "N": int(N),
        "n_small": int(len(small)),
        "n_large": int(len(large)),
        "null_mean": mean_null,
        "bar": bar,
        "observed_mean": obs_mean,
        "fire_rate": {k: fires[k] / N_TEST_DRAWS
                      for k in fires},
        "effect": effect,
        "any_effect": bool(any_effect),
    }


# ------------------------------------------------------
# fixtures (rule 6)
# ------------------------------------------------------
def selftest():
    print("=" * 62)
    print("SELF-TEST ON FIXTURES (must pass before real run)")
    print("=" * 62)
    rng = np.random.RandomState(SEED)

    # a plausible skewed aisle distribution
    p = np.ones(N_AISLES)
    p[23] = 40.0
    p[82] = 40.0
    p = p / p.sum()
    aisles = np.arange(1, N_AISLES + 1)

    # CASE A: identical distributions -> must be NO effect
    A1 = rng.choice(aisles, 16000, p=p)
    A2 = rng.choice(aisles, 24000, p=p)
    ra = run_test(A1, A2, rng,
                  "CASE A: identical distributions "
                  "(truth: NO effect)")

    # CASE B: planted difference -> must DETECT
    p2 = p.copy()
    p2[100:120] *= 6.0
    p2 = p2 / p2.sum()
    B2 = rng.choice(aisles, 24000, p=p2)
    rb = run_test(A1, B2, rng,
                  "CASE B: planted difference "
                  "(truth: REAL effect)")

    ok = (not ra["any_effect"]) and rb["any_effect"]
    print("\n" + "=" * 62)
    print(f"SELF-TEST {'PASSED' if ok else '*** FAILED ***'}")
    print("=" * 62)
    return ok


# ------------------------------------------------------
# real data
# ------------------------------------------------------
def load_real_groups():
    """Mirrors validate_demand_geometry.py exactly:
    same file, same disjointness assert, same grouping
    (row counts per order_id; mid baskets excluded)."""
    df = pd.read_csv(EVAL_PATH)
    assert "aisle_id" in df.columns and \
           "order_id" in df.columns, (
        f"{EVAL_PATH} missing aisle_id/order_id - wrong "
        f"file? columns: {list(df.columns)}")

    tr_ids = set(pd.read_csv(TRAIN_IDS)["order_id"])
    assert len(set(df["order_id"]) & tr_ids) == 0, (
        "FATAL: compare file contains training orders")
    print("[ASSERTED] real data is disjoint from the "
          "model's training orders")

    n_missing = int(df["aisle_id"].isna().sum())
    assert n_missing == 0, (
        f"FATAL: {n_missing} rows have missing aisle_id "
        f"in {EVAL_PATH} - file is dirty, do not proceed")
    df["aisle_id"] = df["aisle_id"].astype(int)

    n_items = df["order_id"].map(
        df.groupby("order_id").size())
    small = df.loc[n_items <= SMALL_MAX,
                   "aisle_id"].to_numpy()
    large = df.loc[n_items >= LARGE_MIN,
                   "aisle_id"].to_numpy()

    assert len(small) > 0 and len(large) > 0, (
        "one of the basket-size groups is empty")
    # demand geometry saw small 16,835 / large 25,344
    print(f"groups from order_id counts "
          f"(small<={SMALL_MAX}, large>={LARGE_MIN}): "
          f"small {len(small):,}, large {len(large):,} "
          f"rows (geometry test saw 16,835 / 25,344)")

    a = np.concatenate([small, large])
    assert a.min() >= 1 and a.max() <= N_AISLES, (
        "aisle_id outside 1..134 - wrong column or file")
    return small, large


def main():
    print("REAL-DATA FLEET-EFFECT TEST "
          "(no synthetic data is read)")
    if not selftest():
        print("aborting: fix the self-test before "
              "trusting any real result.")
        sys.exit(1)

    rng = np.random.RandomState(SEED)
    small, large = load_real_groups()
    res = run_test(small, large, rng,
                   "REAL DATA: small vs large baskets")

    res["design"] = {
        "n_null_draws": N_NULL_DRAWS,
        "n_test_draws": N_TEST_DRAWS,
        "travel_reps": TRAVEL_REPS,
        "percentile": PCTL,
        "small_max": SMALL_MAX,
        "large_min": LARGE_MIN,
        "rule": "metric fires on majority of draws",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {OUT_PATH}")

    # Verdict must be stated PER METRIC. The three metrics
    # can (and did) split: a blanket "effect exists" line
    # would attribute the synthetic 2.36-aisle TRAVEL gap
    # to a real effect even when travel itself fired at
    # noise level - misquoting the result it sits next to.
    print("\n" + "=" * 62)
    names = {"tvd":    "aisle mix (TVD)",
             "gini":   "concentration (Gini)",
             "travel": "travel/routing cost"}
    real = [names[k] for k in ("tvd", "gini", "travel")
            if res["effect"][k]]
    fake = [names[k] for k in ("tvd", "gini", "travel")
            if not res["effect"][k]]

    if not real:
        print("VERDICT: NO basket-size fleet effect exists "
              "in the real data\non ANY metric. CTGAN "
              "fabricated an operational effect from\n"
              "NOTHING - it passed quality/efficacy/privacy "
              "and still invented\na 2.36-aisle travel gap "
              "where reality has none.")
    elif not fake:
        print("VERDICT: basket size has a REAL effect on "
              "all three metrics:\n  " + ", ".join(real)
              + "\nCTGAN exaggerated real effects rather "
              "than inventing them.")
    else:
        print("VERDICT (mixed):")
        print(f"  REAL effect:    {', '.join(real)}")
        print(f"  NO real effect: {', '.join(fake)}")
        if not res["effect"]["travel"]:
            print("\nCTGAN exaggerated the real aisle-mix/"
                  "concentration effect,\nbut FABRICATED the "
                  "2.36-aisle travel/routing gap from\n"
                  "nothing - reality shows no travel-cost "
                  "effect at all.\nStandard synthetic-data "
                  "metrics did not catch this. That is\n"
                  "the finding.")
    print("=" * 62)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()