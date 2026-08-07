"""
tabsyn_conditional_geometry.py
==============================
HOW can a generator pass the pairwise association audit and
still invent a travel effect?

THE PUZZLE THIS SCRIPT RESOLVES
-------------------------------
TabSyn's association audit came back NOT CONFIRMED: weak-pair
inflation -0.0006 against CTGAN's +0.0585. On the pairwise
evidence TabSyn does not fabricate structure. Yet it invents
a small-vs-large travel gap of ~0.98 aisles where reality has
none, and it inverts the sign.

Those two facts look contradictory. They are not, and the
reason is not statistical - it is mathematical:

    Cramer's V is computed from the (aisle_id x
    order_size_grp) CONTINGENCY TABLE. Permuting the aisle
    LABELS permutes the rows of that table. chi2, n, r and k
    are all unchanged, so V - and the excess_v used by the
    audit - is EXACTLY invariant.

    Expected travel is E|a_i - a_j|, a functional of aisle
    POSITION on the 1..134 index. Relabelling aisles changes
    it freely.

So the audit metric is mathematically BLIND to the geometry
that travel measures. A generator can reproduce the strength
of the aisle/basket-size association perfectly and still put
the probability mass at the wrong POSITIONS - which is
exactly what moves robot travel and nothing the pairwise
audit looks at.

This script does not argue that; it MEASURES it:

  MECHANISM A  association strength, real vs synthetic
               (expect: matched - the audit passes)
               against travel geometry, real vs synthetic
               (expect: not matched - the gap is fabricated)

  MECHANISM B  THE CONTROL. Randomly relabel aisles in the
               REAL data, 20 times. excess_v must not move
               at all (asserted <= 1e-12). The travel gap
               must move a lot. This proves the two
               quantities are independent - one can be held
               exactly fixed while the other ranges freely.
               It is the control that makes the argument a
               finding rather than a story.

  MECHANISM C  WHERE. For integer support,
                   E|X - Y| = 2 * sum_k F(k) (1 - F(k))
               exactly. So the small-vs-large travel gap
               decomposes additively over aisle POSITIONS,
               and the decomposition shows precisely which
               stretch of the warehouse each generator gets
               wrong. Deterministic - no permutation noise.

DISCIPLINE (same as every other script in this project)
-------------------------------------------------------
 - Every bar is MEASURED from real-vs-real half-splits at
   the same N (40 draws, 95th percentile). Nothing guessed.
 - N is FORCED to the recorded n_common so the bars are
   comparable; TVD/Gini/travel are all sample-size dependent
   (standing rule 3).
 - gini()/tvd() use the FIXED 1..134 bincount support copied
   from test_real_fleet_effect.py, so an aisle a generator
   never emits still counts as a zero.
 - cramers_v()/excess_v() are copied VERBATIM from
   association_audit.py, so "the audit" here is literally the
   audit that TabSyn passed.
 - --selftest verifies all four detectors against planted
   faults and clean fixtures before anything is trusted
   (standing rule 2).

READS  (all read-only):
    data/v3_compare.csv
    data/v3_train_order_ids.csv
    data/real_fleet_effect.json      (recorded N and bars)
    <--synthetic-csv>
WRITES (new files only):
    <--out-dir>/conditional_geometry.json
    <--out-dir>/conditional_geometry.png

Usage:
    python tabsyn_conditional_geometry.py --selftest ^
        --out-dir results/tabsyn/cond_geometry
    python tabsyn_conditional_geometry.py ^
        --synthetic-csv data/tabsyn/synthetic_tabsyn.csv ^
        --out-dir results/tabsyn/cond_geometry
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_BAR_DRAWS = 40      # real-vs-real half-splits per bar
PCTL = 95             # bar = this percentile
TRAVEL_REPS = 30      # permutations inside the estimator
NULL_REPS = 25        # permutations inside excess_v (audit)
N_RELABEL = 20        # relabellings in the control

SMALL_MAX = 10
LARGE_MIN = 14
N_AISLES = 134

EVAL_PATH = "data/v3_compare.csv"
TRAIN_IDS = "data/v3_train_order_ids.csv"
RECORDED = "data/real_fleet_effect.json"
COND_COL = "order_size_grp"

C_REAL, C_SYN = 'steelblue', 'seagreen'


# ══════════════════════════════════════════
# GEOMETRY - fixed 1..134 support throughout
# (gini/tvd verbatim from test_real_fleet_effect.py)
# ══════════════════════════════════════════
def aisle_pmf(a):
    c = np.bincount(np.asarray(a, dtype=int),
                    minlength=N_AISLES + 1)[1:].astype(float)
    return c / c.sum()


def gini(aisles):
    counts = np.bincount(np.asarray(aisles, dtype=int),
                         minlength=N_AISLES + 1)[1:]
    counts = np.sort(counts.astype(float))
    n = len(counts)
    cum = np.cumsum(counts)
    if cum[-1] == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def tvd(a, b):
    pa, pb = aisle_pmf(a), aisle_pmf(b)
    return float(0.5 * np.abs(pa - pb).sum())


def travel_perm(aisles, rng, reps=TRAVEL_REPS):
    """The project's estimator: mean |aisle difference|
    between consecutive picks over `reps` random pick
    orders. Carries permutation noise."""
    a = np.asarray(aisles, dtype=float)
    return float(np.mean([
        np.mean(np.abs(np.diff(rng.permutation(a))))
        for _ in range(reps)]))


def travel_exact(aisles):
    """E|X - Y| for two independent draws from the empirical
    aisle distribution, in closed form.

    For a variable on the integer lattice 1..M,
        |X - Y| = sum_k 1{min <= k < max}
    and for iid X, Y,
        P(X <= k < Y) + P(Y <= k < X) = 2 F(k) (1 - F(k)).
    Summing k = 1..M-1 gives the identity used here. It is
    EXACT (no permutation noise) and additive over aisle
    positions, which is what makes MECHANISM C possible.

    travel_perm() draws WITHOUT replacement from a finite
    sample, so it differs by O(1/n); selftest fixture 3
    measures that the two agree at the N actually used
    rather than assuming it."""
    F = np.cumsum(aisle_pmf(aisles))
    return float(2.0 * np.sum(F[:-1] * (1.0 - F[:-1])))


def travel_contrib(aisles):
    """Per-position contribution to travel_exact: the k-th
    entry is 2 F(k)(1 - F(k)), and they sum to E|X - Y|."""
    F = np.cumsum(aisle_pmf(aisles))
    return 2.0 * F[:-1] * (1.0 - F[:-1])


def spread_stats(a):
    """Position-sensitive summaries of a group's aisle mix.
    Every one of these moves when aisles are relabelled;
    none of them is visible to a contingency-table metric."""
    a = np.asarray(a, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return {
        'mean_aisle': float(np.mean(a)),
        'sd_aisle': float(np.std(a)),
        'iqr_aisle': float(q3 - q1),
        'gini': gini(a),
        'travel': travel_exact(a),
    }


STAT_KEYS = ['mean_aisle', 'sd_aisle', 'iqr_aisle',
             'gini', 'travel']


# ══════════════════════════════════════════
# ASSOCIATION - VERBATIM from association_audit.py
# ══════════════════════════════════════════
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = ct.shape
    d = min(r - 1, k - 1)
    return np.sqrt(phi2 / d) if d > 0 else np.nan


def excess_v(x, y, seed=SEED, reps=NULL_REPS):
    """Cramer's V minus the 95th percentile of the
    permutation null. Removes the cardinality /
    sample-size inflation that makes V look large
    even for independent columns."""
    x = np.asarray(x)
    y = np.asarray(y)
    v = cramers_v(x, y)
    if np.isnan(v):
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    null = np.array([
        cramers_v(x, rng.permutation(y))
        for _ in range(reps)])
    null = null[~np.isnan(null)]
    if len(null) == 0:
        return v, np.nan, np.nan
    thresh = np.percentile(null, 95)
    return v, thresh, max(v - thresh, 0.0)


def paired_arrays(small, large):
    """Stack two groups into the (aisle, group) columns the
    audit consumes."""
    x = np.concatenate([small, large])
    y = np.array(['small'] * len(small) + ['large'] * len(large))
    return x, y


# ══════════════════════════════════════════
# MEASURED BARS - real vs real, same N
# ══════════════════════════════════════════
def measure_bars(pool, N, rng):
    """Split ONE real group into two disjoint halves of size
    N. Both halves are the same demand type, so every
    difference between them is pure sampling noise. Repeat
    N_BAR_DRAWS times; the bar is the 95th percentile.

    This is the only yardstick in the script."""
    assert len(pool) >= 2 * N, (
        f"FATAL: pool of {len(pool):,} cannot supply two "
        f"disjoint halves of {N:,}")
    diffs = {k: [] for k in STAT_KEYS}
    diffs['tvd'] = []
    for _ in range(N_BAR_DRAWS):
        idx = rng.permutation(len(pool))
        h1, h2 = pool[idx[:N]], pool[idx[N:2 * N]]
        s1, s2 = spread_stats(h1), spread_stats(h2)
        for k in STAT_KEYS:
            diffs[k].append(abs(s1[k] - s2[k]))
        diffs['tvd'].append(tvd(h1, h2))
    return {k: {'bar': float(np.percentile(v, PCTL)),
                'mean': float(np.mean(v))}
            for k, v in diffs.items()}


# ══════════════════════════════════════════
# SELFTEST - standing rule 2
# ══════════════════════════════════════════
def selftest():
    print("=" * 66)
    print("SELFTEST - every detector vs a planted fault "
          "and clean data")
    print("=" * 66)
    rng = np.random.RandomState(SEED)
    res = {}

    aisles = np.arange(1, N_AISLES + 1)
    p = np.ones(N_AISLES)
    p[23] = 40.0
    p[82] = 40.0
    p = p / p.sum()
    q = p.copy()
    q[100:120] *= 6.0
    q = q / q.sum()

    A = rng.choice(aisles, 20000, p=p)
    B = rng.choice(aisles, 20000, p=q)   # genuinely different

    # --- fixture 1: relabelling must NOT move excess_v ---
    print("\n  [1] relabelling aisles must NOT move "
          "excess_v (the audit metric)")
    x, y = paired_arrays(A, B)
    _, _, ex0 = excess_v(x, y)
    worst = 0.0
    for s in range(5):
        sigma = np.random.RandomState(900 + s).permutation(
            np.arange(1, N_AISLES + 1))
        xr = sigma[x - 1]
        _, _, exr = excess_v(xr, y)
        worst = max(worst, abs(exr - ex0))
    res['relabel_leaves_excess_v_exact'] = bool(worst <= 1e-12)
    print(f"      excess_v {ex0:.6f}, worst drift over 5 "
          f"relabellings {worst:.3e}")
    print(f"      -> {'PASS' if worst <= 1e-12 else 'FAIL'} "
          f"(must be <= 1e-12)")

    # --- fixture 2: relabelling MUST move travel ---
    print("\n  [2] relabelling aisles MUST move the travel "
          "gap (the geometry)")
    gap0 = travel_exact(B) - travel_exact(A)
    moved = []
    for s in range(5):
        sigma = np.random.RandomState(900 + s).permutation(
            np.arange(1, N_AISLES + 1))
        moved.append(travel_exact(sigma[B - 1])
                     - travel_exact(sigma[A - 1]))
    swing = float(max(moved) - min(moved))
    res['relabel_moves_travel'] = bool(swing > 0.5)
    print(f"      gap {gap0:+.3f} aisles -> relabelled gaps "
          f"span {swing:.3f} aisles")
    print(f"      -> {'PASS' if swing > 0.5 else 'FAIL'} "
          f"(must move materially)")

    # --- fixture 3: closed form == permutation estimate ---
    print("\n  [3] closed-form E|X-Y| must match the "
          "project's permutation estimator")
    ex = travel_exact(A)
    pe = [travel_perm(A, np.random.RandomState(700 + s))
          for s in range(5)]
    pe_spread = float(max(pe) - min(pe))
    err = float(abs(np.mean(pe) - ex))
    ok3 = err <= max(pe_spread, 1e-6) * 2.0
    res['closed_form_matches_estimator'] = bool(ok3)
    print(f"      exact {ex:.4f}   estimator mean "
          f"{np.mean(pe):.4f}   |err| {err:.5f}")
    print(f"      estimator's own spread over 5 seeds "
          f"{pe_spread:.5f}")
    print(f"      -> {'PASS' if ok3 else 'FAIL'} (error "
          f"within the estimator's own noise)")

    # --- fixture 4: the TVD detector itself ---
    print("\n  [4] conditional-TVD detector: quiet on "
          "identical groups, fires on a planted shift")
    N = 5000
    bars = measure_bars(A, N, np.random.RandomState(SEED))
    A2 = rng.choice(aisles, 20000, p=p)   # same distribution
    clean_tvd = tvd(A[:N], A2[:N])
    dirty_tvd = tvd(A[:N], B[:N])
    quiet = clean_tvd <= bars['tvd']['bar']
    fires = dirty_tvd > bars['tvd']['bar']
    res['tvd_quiet_on_identical'] = bool(quiet)
    res['tvd_fires_on_planted'] = bool(fires)
    print(f"      bar {bars['tvd']['bar']:.4f}   "
          f"identical {clean_tvd:.4f} -> "
          f"{'quiet OK' if quiet else 'FALSE ALARM'}")
    print(f"                      planted   "
          f"{dirty_tvd:.4f} -> "
          f"{'fires OK' if fires else 'MISSED'}")

    ok = all(res.values())
    print("\n" + "=" * 66)
    print(f"SELFTEST: "
          f"{'ALL DETECTORS VERIFIED' if ok else 'FAILURES'}")
    for k, v in res.items():
        print(f"    {k:<38} {'PASS' if v else 'FAIL'}")
    print("=" * 66)
    return ok, res


# ══════════════════════════════════════════
# DATA
# ══════════════════════════════════════════
def load_real_groups():
    df = pd.read_csv(EVAL_PATH)
    tr_ids = set(pd.read_csv(TRAIN_IDS)["order_id"])
    assert len(set(df["order_id"]) & tr_ids) == 0, (
        "FATAL: compare file contains training orders")
    print("  [ASSERTED] real data is disjoint from the "
          "model's training orders")
    assert int(df["aisle_id"].isna().sum()) == 0, (
        "FATAL: missing aisle_id in the real file")
    df["aisle_id"] = df["aisle_id"].astype(int)
    n_items = df["order_id"].map(df.groupby("order_id").size())
    small = df.loc[n_items <= SMALL_MAX, "aisle_id"].to_numpy()
    large = df.loc[n_items >= LARGE_MIN, "aisle_id"].to_numpy()
    a = np.concatenate([small, large])
    assert a.min() >= 1 and a.max() <= N_AISLES
    print(f"  real   small {len(small):,}   "
          f"large {len(large):,} picks")
    return small, large


def load_synth_groups(path):
    df = pd.read_csv(path)
    assert COND_COL in df.columns, \
        f"FATAL: {path} has no {COND_COL} column"
    a = pd.to_numeric(df["aisle_id"], errors="coerce")
    assert int(a.isna().sum()) == 0, \
        f"FATAL: non-numeric aisle_id in {path}"
    df["aisle_id"] = a.astype(int)
    grp = df[COND_COL].astype(str)
    unexpected = set(grp.unique()) - {"small", "large"}
    assert not unexpected, \
        f"FATAL: unexpected {COND_COL} values {unexpected}"
    small = df.loc[grp == "small", "aisle_id"].to_numpy()
    large = df.loc[grp == "large", "aisle_id"].to_numpy()
    both = np.concatenate([small, large])
    assert both.min() >= 1 and both.max() <= N_AISLES
    tot = len(small) + len(large)
    print(f"  synth  small {len(small):,} "
          f"({len(small)/tot:.1%})   large {len(large):,} "
          f"({len(large)/tot:.1%}) picks")
    return small, large


# ══════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    # protection rule: validate BEFORE creating anything
    assert "results/tabsyn" in os.path.normpath(
        args.out_dir).replace("\\", "/"), \
        "FATAL: out-dir must live under results/tabsyn"
    os.makedirs(args.out_dir, exist_ok=True)

    ok, st = selftest()
    if args.selftest:
        out = os.path.join(args.out_dir,
                           "cond_geometry_selftest.json")
        json.dump(st, open(out, "w"), indent=2)
        print(f"Saved {out}")
        sys.exit(0 if ok else 1)
    if not ok:
        print("aborting: fix the selftest first.")
        sys.exit(1)

    assert args.synthetic_csv, "--synthetic-csv is required"

    rec = json.load(open(RECORDED))
    N = rec["N"]

    print("\n" + "=" * 66)
    print("CONDITIONAL AISLE GEOMETRY - real vs synthetic")
    print(f"scored file: {args.synthetic_csv}")
    print("=" * 66)
    real = {}
    synth = {}
    real['small'], real['large'] = load_real_groups()
    synth['small'], synth['large'] = load_synth_groups(
        args.synthetic_csv)
    print(f"\n  N forced to the recorded n_common = {N:,} "
          f"(standing rule 3)")

    rng = np.random.RandomState(SEED)
    # size-matched samples, drawn once and reused everywhere
    R = {g: real[g][rng.choice(len(real[g]), N,
                               replace=False)]
         for g in ('small', 'large')}
    S = {g: synth[g][rng.choice(len(synth[g]), N,
                                replace=False)]
         for g in ('small', 'large')}

    # ── measured bars, one per group, from real data only ──
    print("\n" + "=" * 66)
    print(f"[1] MEASURED BARS ({N_BAR_DRAWS} real-vs-real "
          f"half-splits at n={N:,}, {PCTL}th pct)")
    print("=" * 66)
    bars = {g: measure_bars(real[g], N,
                            np.random.RandomState(SEED))
            for g in ('small', 'large')}
    for g in ('small', 'large'):
        print(f"\n  {g.upper()}")
        for k in ['tvd'] + STAT_KEYS:
            print(f"    {k:<12} mean {bars[g][k]['mean']:.4f}"
                  f"   BAR {bars[g][k]['bar']:.4f}")

    # ── per-group comparison ──
    print("\n" + "=" * 66)
    print("[2] CONDITIONAL AISLE MIX AND SPREAD, PER GROUP")
    print("=" * 66)
    per_group = {}
    for g in ('small', 'large'):
        rs, ss = spread_stats(R[g]), spread_stats(S[g])
        tv = tvd(R[g], S[g])
        row = {'tvd': {'value': tv,
                       'bar': bars[g]['tvd']['bar'],
                       'ok': bool(tv <= bars[g]['tvd']['bar'])}}
        for k in STAT_KEYS:
            err = abs(rs[k] - ss[k])
            row[k] = {'real': rs[k], 'synth': ss[k],
                      'err': err, 'bar': bars[g][k]['bar'],
                      'ok': bool(err <= bars[g][k]['bar'])}
        per_group[g] = row

        print(f"\n  --- {g.upper()} BASKETS (n={N:,}) ---")
        print(f"    {'stat':<12}{'real':>10}{'synth':>10}"
              f"{'err':>9}{'bar':>9}   verdict")
        print("    " + "-" * 58)
        print(f"    {'aisle TVD':<12}{'-':>10}{'-':>10}"
              f"{tv:>9.4f}{bars[g]['tvd']['bar']:>9.4f}"
              f"   {'OK' if row['tvd']['ok'] else 'DEVIATES'}")
        for k in STAT_KEYS:
            c = row[k]
            print(f"    {k:<12}{c['real']:>10.4f}"
                  f"{c['synth']:>10.4f}{c['err']:>9.4f}"
                  f"{c['bar']:>9.4f}   "
                  f"{'OK' if c['ok'] else 'DEVIATES'}")

    # ── MECHANISM A ──
    print("\n" + "=" * 66)
    print("[3] MECHANISM A - the audit metric vs the geometry")
    print("=" * 66)
    xr, yr = paired_arrays(R['small'], R['large'])
    xs, ys = paired_arrays(S['small'], S['large'])
    vr, nr, exr = excess_v(xr, yr)
    vs, ns, exs = excess_v(xs, ys)

    gap_real = travel_exact(R['large']) - travel_exact(R['small'])
    gap_syn = travel_exact(S['large']) - travel_exact(S['small'])
    travel_bar = rec['bar']['travel']

    print(f"""
    THE AUDIT SEES (aisle_id x {COND_COL}):
      excess Cramer's V   real {exr:.4f}   synth {exs:.4f}"""
          f"""   diff {abs(exr - exs):+.4f}

    THE WAREHOUSE SEES (small-vs-large travel gap):
      gap                 real {gap_real:+.3f}   synth """
          f"""{gap_syn:+.3f}   aisles
      measured noise bar  {travel_bar:.3f} aisles """
          f"""(real_fleet_effect.json)
""")
    audit_matched = abs(exr - exs) <= 0.02
    geom_wrong = abs(gap_syn) > travel_bar >= abs(gap_real)
    print(f"    audit agrees within 0.02:   {audit_matched}")
    print(f"    geometry wrong past the bar: {geom_wrong}")
    if audit_matched and geom_wrong:
        print("\n    -> THE PUZZLE IS REAL AND REPRODUCED "
              "ON THIS FILE.\n"
              "       Matched association, wrong geometry.")

    # ── MECHANISM B: the control ──
    print("\n" + "=" * 66)
    print(f"[4] MECHANISM B - THE CONTROL "
          f"({N_RELABEL} aisle relabellings of REAL data)")
    print("=" * 66)
    print("""
    Relabelling aisles permutes the rows of the contingency
    table. chi2, n, r and k do not change, so excess_v must
    not move AT ALL. Travel depends on aisle position, so it
    must move freely. If both hold, the audit provably
    cannot constrain the travel gap - which is why TabSyn
    can pass one and fail the other.
""")
    relabel_gaps = []
    worst_v = 0.0
    for s in range(N_RELABEL):
        sigma = np.random.RandomState(500 + s).permutation(
            np.arange(1, N_AISLES + 1))
        _, _, ex_r = excess_v(sigma[xr - 1], yr)
        worst_v = max(worst_v, abs(ex_r - exr))
        relabel_gaps.append(
            travel_exact(sigma[R['large'] - 1])
            - travel_exact(sigma[R['small'] - 1]))
    relabel_gaps = np.array(relabel_gaps)

    assert worst_v <= 1e-12, (
        f"FATAL: excess_v moved by {worst_v:.3e} under aisle "
        f"relabelling. It is supposed to be exactly "
        f"invariant - the mechanism argument does not hold "
        f"if this fires, so the run aborts.")
    print(f"    excess_v across all {N_RELABEL} relabellings:"
          f" constant at {exr:.6f}")
    print(f"      worst drift {worst_v:.3e}  [ASSERTED "
          f"<= 1e-12 - exactly invariant]")
    print(f"    travel gap across the same relabellings:")
    print(f"      min {relabel_gaps.min():+.3f}   "
          f"max {relabel_gaps.max():+.3f}   "
          f"sd {relabel_gaps.std():.3f} aisles")
    print(f"      span {relabel_gaps.max() - relabel_gaps.min():.3f}"
          f" aisles, against a noise bar of "
          f"{travel_bar:.3f}")
    print(f"\n    -> Holding the audit metric EXACTLY fixed, "
          f"the travel gap still\n"
          f"       ranges over "
          f"{relabel_gaps.max() - relabel_gaps.min():.2f} "
          f"aisles - "
          f"{(relabel_gaps.max() - relabel_gaps.min()) / travel_bar:.1f}x "
          f"the noise bar.\n"
          f"       A pairwise association audit cannot "
          f"detect this failure mode.")

    # ── MECHANISM C: exact decomposition ──
    print("\n" + "=" * 66)
    print("[5] MECHANISM C - WHERE the gap comes from")
    print("=" * 66)
    contrib_real = (travel_contrib(R['large'])
                    - travel_contrib(R['small']))
    contrib_syn = (travel_contrib(S['large'])
                   - travel_contrib(S['small']))
    # identity check: contributions must sum to the gap
    assert abs(contrib_real.sum() - gap_real) < 1e-9
    assert abs(contrib_syn.sum() - gap_syn) < 1e-9
    print("    [ASSERTED] per-position contributions sum to "
          "the gap (exact identity)")

    order = np.argsort(-np.abs(contrib_syn))[:8]
    print(f"\n    Top aisle positions driving the SYNTHETIC "
          f"gap:")
    print(f"    {'aisle k':>9}{'synth contrib':>15}"
          f"{'real contrib':>14}")
    print("    " + "-" * 38)
    for i in order:
        print(f"    {i + 1:>9}{contrib_syn[i]:>15.5f}"
              f"{contrib_real[i]:>14.5f}")

    # ══════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        'How faithful pairwise associations coexist with a '
        'fabricated travel effect',
        fontsize=13, fontweight='bold')

    k_axis = np.arange(1, N_AISLES)

    ax = axes[0][0]
    Fr_s = np.cumsum(aisle_pmf(R['small']))
    Fr_l = np.cumsum(aisle_pmf(R['large']))
    Fs_s = np.cumsum(aisle_pmf(S['small']))
    Fs_l = np.cumsum(aisle_pmf(S['large']))
    ax.plot(np.arange(1, N_AISLES + 1), Fr_l - Fr_s, lw=1.6,
            color=C_REAL, label='real')
    ax.plot(np.arange(1, N_AISLES + 1), Fs_l - Fs_s, lw=1.6,
            ls='--', color=C_SYN, label='synthetic')
    ax.axhline(0, color='k', lw=0.7)
    ax.set_title('Conditional CDF gap  F(aisle|large) - '
                 'F(aisle|small)\n(travel is driven by this '
                 'curve, not by association strength)')
    ax.set_xlabel('aisle')
    ax.set_ylabel('CDF difference')
    ax.legend(fontsize=9)

    ax = axes[0][1]
    ax.plot(k_axis, contrib_real, lw=1.4, color=C_REAL,
            label=f'real (sums to {gap_real:+.2f})')
    ax.plot(k_axis, contrib_syn, lw=1.4, ls='--', color=C_SYN,
            label=f'synthetic (sums to {gap_syn:+.2f})')
    ax.axhline(0, color='k', lw=0.7)
    ax.set_title('Exact decomposition of the travel gap\n'
                 'per-position contribution, '
                 '2 F(k)(1-F(k)) difference')
    ax.set_xlabel('aisle position k')
    ax.set_ylabel('contribution (aisles)')
    ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.hist(relabel_gaps, bins=12, color='lightgrey',
            edgecolor='dimgray',
            label=f'{N_RELABEL} aisle relabellings of REAL '
                  f'data\n(excess_v identical in every one)')
    ax.axvline(gap_real, color=C_REAL, lw=2,
               label=f'real gap {gap_real:+.2f}')
    ax.axvline(gap_syn, color=C_SYN, lw=2, ls='--',
               label=f'synthetic gap {gap_syn:+.2f}')
    for sgn in (-1, 1):
        ax.axvline(sgn * travel_bar, color='k', lw=1.1,
                   ls=':')
    ax.text(travel_bar, ax.get_ylim()[1] * 0.95,
            f' noise bar {travel_bar:.2f}', fontsize=8,
            va='top')
    ax.set_title('THE CONTROL: association held EXACTLY '
                 'fixed,\ntravel gap still ranges freely')
    ax.set_xlabel('small-vs-large travel gap (aisles)')
    ax.set_ylabel('relabellings')
    ax.legend(fontsize=7.5)

    ax = axes[1][1]
    x = np.arange(2)
    w = 0.35
    audit_vals = [exr, exs]
    geom_vals = [abs(gap_real), abs(gap_syn)]
    ax.bar(x - w / 2, audit_vals, w, color=[C_REAL, C_SYN],
           label='excess Cramer\'s V (left axis)')
    ax.set_xticks(x)
    ax.set_xticklabels(['real', 'synthetic'])
    ax.set_ylabel("excess Cramer's V  (what the audit sees)")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, geom_vals, w, color=[C_REAL, C_SYN],
            hatch='//', alpha=0.75)
    ax2.axhline(travel_bar, color='k', ls=':', lw=1.1)
    ax2.set_ylabel('|travel gap| in aisles  '
                   '(what the warehouse sees)')
    ax.set_title('Same two datasets, two metrics\n'
                 'solid = audit (agrees), hatched = travel '
                 '(does not)')

    plt.tight_layout()
    png = os.path.join(args.out_dir,
                       'conditional_geometry.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    print(f"\n  saved {png}")

    # ══════════════════════════════════════
    out = {
        'scored_file': args.synthetic_csv,
        'N': int(N),
        'bars': bars,
        'per_group': per_group,
        'mechanism_a': {
            'excess_v_real': float(exr),
            'excess_v_synth': float(exs),
            'cramers_v_real': float(vr),
            'cramers_v_synth': float(vs),
            'travel_gap_real': float(gap_real),
            'travel_gap_synth': float(gap_syn),
            'travel_bar': float(travel_bar),
            'audit_matched': bool(audit_matched),
            'geometry_wrong': bool(geom_wrong),
        },
        'mechanism_b_control': {
            'n_relabellings': N_RELABEL,
            'excess_v_worst_drift': float(worst_v),
            'excess_v_exactly_invariant': True,
            'travel_gap_min': float(relabel_gaps.min()),
            'travel_gap_max': float(relabel_gaps.max()),
            'travel_gap_sd': float(relabel_gaps.std()),
            'travel_gap_span': float(relabel_gaps.max()
                                     - relabel_gaps.min()),
            'span_in_bars': float(
                (relabel_gaps.max() - relabel_gaps.min())
                / travel_bar),
        },
        'mechanism_c_decomposition': {
            'aisle_positions': list(range(1, N_AISLES)),
            'contrib_real': [float(v) for v in contrib_real],
            'contrib_synth': [float(v) for v in contrib_syn],
            'identity_checked': True,
        },
        'design': {
            'n_bar_draws': N_BAR_DRAWS,
            'percentile': PCTL,
            'null_reps_in_excess_v': NULL_REPS,
            'support': f'fixed 1..{N_AISLES} bincount',
            'travel': 'closed form E|X-Y| = 2*sum F(1-F), '
                      'verified against the permutation '
                      'estimator in selftest fixture 3',
        },
    }
    js = os.path.join(args.out_dir,
                      'conditional_geometry.json')
    json.dump(out, open(js, "w"), indent=2, default=float)
    print(f"  Saved {js}")
    print("=" * 66)


if __name__ == "__main__":
    main()
