"""
tabsyn_fleet_effect_draws.py
============================
Is the FABRICATED small-vs-large travel gap robust, or a
one-draw fluke?

THE GAP THIS SCRIPT CLOSES
--------------------------
Both generators are on record with a single point estimate
of the small-vs-large travel gap:

    real       0.19 aisles   (no effect - fires 10% of draws)
    CTGAN      2.36 aisles   (demand_geometry.json)
    TabSyn     0.98 aisles   (results/tabsyn/..., INVERTED)

The REAL number went through the full repeated-draw
majority-vote rule in test_real_fleet_effect.py: 40 measured
null draws, a 95th-percentile bar, then 10 fresh observed
draws with a metric claiming an effect only if it fires on a
MAJORITY of them. The SYNTHETIC numbers did not - they are
each one draw. A 0.98-aisle gap sitting just above a
0.64-aisle bar could be a draw-to-draw fluctuation.

This script puts the synthetic gap through the identical
rule so the claim "TabSyn also fabricates a travel effect"
rests on a fire rate, not on one number.

WHAT IS COPIED, AND FROM WHERE
------------------------------
travel(), gini(), tvd(), metric_diffs(), run_test() and
selftest() are copied VERBATIM from test_real_fleet_effect.py
(the fixture-verified original). Note these use a FIXED
1..134 bincount support, so an aisle a generator never emits
still counts as a zero - unlike the observed-categories
gini() in validate_demand_geometry.py. That is the stricter
and more correct version, and it is the one used here.

DECLARED DEVIATIONS (nothing else changed)
------------------------------------------
 1. n_force: run_test() computes N = min(len)//2 itself. The
    recorded bars are only valid at the recorded N, so N is
    FORCED to real_fleet_effect.json["N"] (8,417) and
    asserted. TVD/Gini/travel are all sample-size dependent
    - standing rule 3. At the real data this changes
    nothing (min(16835, 25344)//2 IS 8,417), so the
    known-answer gate still reproduces exactly.
 2. extra_bar: the SAME observed draws are additionally
    counted against an externally supplied bar. This
    consumes NO randomness and cannot alter the original
    numbers - it only adds a second tally over draws that
    were already computed.
 3. I/O head: --synthetic-csv / --out-dir; outputs are new
    files under results/tabsyn/ only.

THE TWO BARS - BOTH ARE REPORTED, THEY ANSWER DIFFERENT
QUESTIONS
-------------------------------------------------------
  REAL bar (0.6418, from real_fleet_effect.json)
      "Is the generated gap bigger than the noise of REAL
       warehouse data?"  <- this is the headline claim, and
      the only bar directly comparable to the recorded real
      fire rate of 10%.

  SYNTHETIC-INTERNAL bar (measured here, 40 half-splits of
  the generator's OWN output at the same N)
      "Is the generated gap bigger than the generator's own
       sampling noise?"  <- robustness. A generator with a
      wide internal null could produce a large gap by chance.

Reporting only the first would overstate; only the second
would not be comparable to reality. Both are printed and
both are saved.

MODES
-----
  --mode selftest
      the two fixtures from the original (Case A identical
      -> NO effect; Case B planted -> DETECTED). Standing
      rule 2: verify the detector before trusting it.

  --mode known-answer
      re-runs the REAL-data path and ASSERTS it reproduces
      data/real_fleet_effect.json (bar, null_mean,
      observed_mean, fire_rate) to <= 1e-9. Contract rule 1:
      this copy does not graduate until it reproduces the
      recorded answer.

  --mode synthetic --synthetic-csv F
      the measurement. Splits F by its order_size_grp column
      and runs the identical rule.

READS  (all read-only):
    data/v3_compare.csv
    data/v3_train_order_ids.csv
    data/real_fleet_effect.json
    <--synthetic-csv>            (synthetic mode only)
WRITES (new files only):
    <--out-dir>/fleet_effect_draws.json   (or selftest.json)

Usage:
    python tabsyn_fleet_effect_draws.py --mode selftest ^
        --out-dir results/tabsyn/fleet_draws
    python tabsyn_fleet_effect_draws.py --mode known-answer ^
        --out-dir results/tabsyn/fleet_draws
    python tabsyn_fleet_effect_draws.py --mode synthetic ^
        --synthetic-csv data/tabsyn/synthetic_tabsyn.csv ^
        --out-dir results/tabsyn/fleet_draws
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# ── identical constants to test_real_fleet_effect.py ──
SEED = 42
N_NULL_DRAWS = 40    # draws to build the null distribution
N_TEST_DRAWS = 10    # repeated observed comparisons
TRAVEL_REPS = 30     # permutations averaged inside travel()
PCTL = 95            # bar = this percentile of the null

SMALL_MAX = 10       # small basket:  <= 10 items
LARGE_MIN = 14       # large basket:  >= 14 items
N_AISLES = 134       # Instacart aisles are 1..134

EVAL_PATH = "data/v3_compare.csv"
TRAIN_IDS = "data/v3_train_order_ids.csv"
RECORDED = "data/real_fleet_effect.json"
COND_COL = "order_size_grp"


# ------------------------------------------------------
# metrics - VERBATIM from test_real_fleet_effect.py
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
# core test - VERBATIM except deviations 1 and 2
# ------------------------------------------------------
def run_test(small, large, rng, label="",
             n_force=None, extra_bar=None,
             extra_label="recorded REAL bar"):
    small = np.asarray(small)
    large = np.asarray(large)

    # DEVIATION 1: N forced so the recorded bars apply.
    N = min(len(small), len(large)) // 2      # rule 2
    if n_force is not None:
        assert n_force <= N, (
            f"FATAL: cannot force N={n_force:,} - the groups "
            f"only support N={N:,} (need >= {2 * n_force:,} "
            f"rows in the smaller group). Generate a larger "
            f"sample; the bars are only valid at the "
            f"recorded N.")
        N = n_force
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

    bar = {k: float(np.percentile(v, PCTL))
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
    # DEVIATION 2: a second tally over the SAME draws.
    # Consumes no randomness; cannot change anything above.
    fires_x = {k: 0 for k in bar}
    for _ in range(N_TEST_DRAWS):
        s = small[rng.choice(len(small), N, replace=False)]
        l = large[rng.choice(len(large), N, replace=False)]
        d = metric_diffs(s, l, rng)
        for k in bar:
            obs_all[k].append(d[k])
            if d[k] > bar[k]:
                fires[k] += 1
            if extra_bar is not None and d[k] > extra_bar[k]:
                fires_x[k] += 1

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

    res = {
        "N": int(N),
        "n_small": int(len(small)),
        "n_large": int(len(large)),
        "null_mean": mean_null,
        "bar": bar,
        "observed_mean": obs_mean,
        "observed_draws": {k: [float(x) for x in v]
                           for k, v in obs_all.items()},
        "fire_rate": {k: fires[k] / N_TEST_DRAWS
                      for k in fires},
        "effect": effect,
        "any_effect": bool(any_effect),
    }

    if extra_bar is not None:
        effect_x = {k: fires_x[k] > N_TEST_DRAWS // 2
                    for k in bar}
        print(f"  SAME draws vs the {extra_label}:")
        for k in ("travel", "gini", "tvd"):
            pct = 100 * fires_x[k] // N_TEST_DRAWS
            verdict = "EFFECT" if effect_x[k] else "no effect"
            print(f"    {k:<7} bar {extra_bar[k]:.4f}   "
                  f"fires {pct:>3}%   -> {verdict}")
        res["external_bar"] = {k: float(v)
                               for k, v in extra_bar.items()}
        res["external_fire_rate"] = {
            k: fires_x[k] / N_TEST_DRAWS for k in fires_x}
        res["external_effect"] = effect_x

    return res


# ------------------------------------------------------
# fixtures (rule 6) - VERBATIM
# ------------------------------------------------------
def selftest():
    print("=" * 62)
    print("SELF-TEST ON FIXTURES (must pass before any run)")
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
    return ok, {"case_a_any_effect": ra["any_effect"],
                "case_b_any_effect": rb["any_effect"],
                "passed": bool(ok)}


# ------------------------------------------------------
# data loading
# ------------------------------------------------------
def load_real_groups():
    """VERBATIM from test_real_fleet_effect.py: same file,
    same disjointness assert, same grouping (row counts per
    order_id; mid baskets excluded)."""
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
    print(f"groups from order_id counts "
          f"(small<={SMALL_MAX}, large>={LARGE_MIN}): "
          f"small {len(small):,}, large {len(large):,} "
          f"rows (geometry test saw 16,835 / 25,344)")

    a = np.concatenate([small, large])
    assert a.min() >= 1 and a.max() <= N_AISLES, (
        "aisle_id outside 1..134 - wrong column or file")
    return small, large


def load_synth_groups(path):
    """Split a synthetic file by its GENERATED
    order_size_grp column. Both generators carry this
    column - CTGAN because it was conditioned on it,
    TabSyn because it learned it as a 7th column - so no
    order_id reconstruction is possible or needed."""
    df = pd.read_csv(path)
    assert COND_COL in df.columns, (
        f"FATAL: {path} has no {COND_COL} column - cannot "
        f"split by basket size")
    a = pd.to_numeric(df["aisle_id"], errors="coerce")
    n_bad = int(a.isna().sum())
    assert n_bad == 0, (
        f"FATAL: {n_bad} non-numeric aisle_id values in "
        f"{path}")
    df["aisle_id"] = a.astype(int)

    grp = df[COND_COL].astype(str)
    unexpected = set(grp.unique()) - {"small", "large"}
    assert not unexpected, (
        f"FATAL: unexpected {COND_COL} values {unexpected}")

    small = df.loc[grp == "small", "aisle_id"].to_numpy()
    large = df.loc[grp == "large", "aisle_id"].to_numpy()
    assert len(small) > 0 and len(large) > 0, (
        "one of the generated basket-size groups is empty")

    both = np.concatenate([small, large])
    assert both.min() >= 1 and both.max() <= N_AISLES, (
        f"FATAL: generated aisle_id outside 1..{N_AISLES}")

    # contract rule 3: report the generated proportions -
    # a badly skewed split weakens the comparison's power
    tot = len(small) + len(large)
    print(f"generated groups: small {len(small):,} "
          f"({len(small) / tot:.1%}), large "
          f"{len(large):,} ({len(large) / tot:.1%})")
    return small, large


# ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["selftest", "known-answer",
                             "synthetic"])
    ap.add_argument("--synthetic-csv",
                    help="required with --mode synthetic")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # protection rule: check the path BEFORE creating it,
    # so a typo cannot leave a stray directory behind
    assert "results/tabsyn" in os.path.normpath(
        args.out_dir).replace("\\", "/"), \
        "FATAL: out-dir must live under results/tabsyn"
    os.makedirs(args.out_dir, exist_ok=True)

    rec = json.load(open(RECORDED))

    # ---------- selftest ----------
    if args.mode == "selftest":
        ok, payload = selftest()
        out = os.path.join(args.out_dir, "selftest.json")
        json.dump(payload, open(out, "w"), indent=2)
        print(f"Saved {out}")
        sys.exit(0 if ok else 1)

    # every scoring mode runs the fixtures first
    ok, _ = selftest()
    if not ok:
        print("aborting: fix the self-test before trusting "
              "any result.")
        sys.exit(1)

    # ---------- known-answer gate ----------
    if args.mode == "known-answer":
        print("\n" + "=" * 62)
        print("KNOWN-ANSWER GATE - reproduce "
              "data/real_fleet_effect.json")
        print("=" * 62)
        rng = np.random.RandomState(SEED)
        small, large = load_real_groups()
        res = run_test(small, large, rng,
                       "REAL DATA: small vs large baskets",
                       n_force=rec["N"])

        bad = []
        for field in ("bar", "null_mean", "observed_mean",
                      "fire_rate"):
            for k in ("travel", "gini", "tvd"):
                got, want = res[field][k], rec[field][k]
                if abs(got - want) > 1e-9:
                    bad.append(f"{field}/{k}: {got!r} != "
                               f"{want!r}")
        assert res["N"] == rec["N"], (
            f"N {res['N']} != recorded {rec['N']}")
        assert not bad, (
            "FATAL: this copy does NOT reproduce the "
            "recorded real-data result:\n  "
            + "\n  ".join(bad))
        print("\n  [ASSERTED] bar, null_mean, observed_mean "
              "and fire_rate all reproduce")
        print("             data/real_fleet_effect.json to "
              "<= 1e-9. This copy graduates.")
        out = os.path.join(args.out_dir,
                           "known_answer_real.json")
        json.dump(res, open(out, "w"), indent=2, default=float)
        print(f"Saved {out}")
        return

    # ---------- synthetic ----------
    assert args.synthetic_csv, \
        "--synthetic-csv is required with --mode synthetic"

    print("\n" + "=" * 62)
    print(f"SYNTHETIC: {args.synthetic_csv}")
    print("=" * 62)
    print(f"  forcing N = {rec['N']:,} (the recorded "
          f"n_common) so the recorded")
    print(f"  real bar {rec['bar']['travel']:.4f} actually "
          f"applies - TVD/Gini/travel")
    print("  are all sample-size dependent (standing rule 3)")

    rng = np.random.RandomState(SEED)
    small, large = load_synth_groups(args.synthetic_csv)
    res = run_test(
        small, large, rng,
        f"SYNTHETIC: small vs large ({args.synthetic_csv})",
        n_force=rec["N"],
        extra_bar=rec["bar"],
        extra_label="recorded REAL bar "
                    "(real_fleet_effect.json)")

    # ---------- the summary that answers the question ----
    print("\n" + "=" * 62)
    print("IS THE FABRICATED TRAVEL GAP ROBUST?")
    print("=" * 62)
    print(f"""
    Reality, for reference (real_fleet_effect.json):
      travel  observed {rec['observed_mean']['travel']:.4f}   """
          f"""bar {rec['bar']['travel']:.4f}   """
          f"""fires {rec['fire_rate']['travel']:.0%}
      -> NO real travel effect. Basket size does not change
         routing cost in the real warehouse data.

    This generator, same rule, same N:
      travel  observed {res['observed_mean']['travel']:.4f}
        vs REAL bar      {res['external_bar']['travel']:.4f}"""
          f"""   fires {res['external_fire_rate']['travel']:.0%}"""
          f"""  -> {'EFFECT' if res['external_effect']['travel'] else 'no effect'}
        vs its OWN bar   {res['bar']['travel']:.4f}"""
          f"""   fires {res['fire_rate']['travel']:.0%}"""
          f"""  -> {'EFFECT' if res['effect']['travel'] else 'no effect'}
""")
    ext = res["external_fire_rate"]["travel"]
    own = res["fire_rate"]["travel"]
    if ext > 0.5 and own > 0.5:
        print("    ROBUST FABRICATION. The generated travel\n"
              "    gap clears BOTH bars on a majority of\n"
              "    independent draws: it is bigger than real\n"
              "    warehouse noise AND bigger than the\n"
              "    generator's own sampling noise. Reality\n"
              "    has no such effect. Report the fire rate,\n"
              "    not the point estimate.")
    elif ext > 0.5:
        print("    ABOVE REAL NOISE, BUT NOT ROBUST TO THE\n"
              "    GENERATOR'S OWN NOISE. The gap clears the\n"
              "    real bar but not the generator's internal\n"
              "    bar - so it is a real misstatement about\n"
              "    the warehouse, yet an unstable one. Say\n"
              "    'exceeds the real-data noise bar on\n"
              "    {:.0%} of draws' and show both.".format(ext))
    else:
        print("    NOT ROBUST. The gap does NOT clear the\n"
              "    real bar on a majority of draws. The\n"
              "    single point estimate OVERSTATED it -\n"
              "    correct the claim to a fire rate of\n"
              "    {:.0%} and do not call it fabricated on\n"
              "    the travel axis.".format(ext))

    print("\n    (aisle-mix TVD and Gini are reported above\n"
          "     too - reality HAS a real effect on those\n"
          "     two, so a generated effect there is\n"
          "     exaggeration, not fabrication. Only travel\n"
          "     is fabrication.)")

    res["scored_file"] = args.synthetic_csv
    res["recorded_real"] = {
        "bar": rec["bar"],
        "observed_mean": rec["observed_mean"],
        "fire_rate": rec["fire_rate"],
        "effect": rec["effect"],
    }
    res["design"] = {
        "n_null_draws": N_NULL_DRAWS,
        "n_test_draws": N_TEST_DRAWS,
        "travel_reps": TRAVEL_REPS,
        "percentile": PCTL,
        "n_forced_to": rec["N"],
        "rule": "metric fires on majority of draws",
        "bars": "own = measured from this file; external = "
                "recorded real bar from real_fleet_effect.json",
    }
    out = os.path.join(args.out_dir,
                       "fleet_effect_draws.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
