"""
tabsyn_audit_model.py
=====================
ADAPTED copy of audit_model_artifact.py for the TabSyn
bench (tabsyn_bench_contract.md notes this one is
"adapted - TabSyn artifact differs").

WHY ADAPTED, NOT VERBATIM: the original audits a LIVE SDV
synthesizer (loads ctgan_v3.pkl, calls .sample() /
.sample_from_conditions()). TabSyn's artifact is a set of
torch checkpoints sampled through tabsyn_sample.py in a
SEPARATE venv - there is no common live-sampling API. So
this script audits SAVED DRAW FILES instead. Every CHECK
(range, set membership, coverage-vs-bootstrap-baseline,
stability vs 3*sigma/sqrt(n), scale invariance) keeps the
original's logic and measured thresholds VERBATIM; only
the data acquisition changed from model.sample(...) to
reading CSVs.

WHAT THIS COSTS, STATED HONESTLY:
 - Reproducibility check: without a live sampler there is
   nothing to re-seed. Both generators share the known
   limitation "sampling not seed-reproducible - cite the
   saved CSV" (model_audit.json); the saved CSVs are the
   artifacts of record for BOTH.
 - Conditioning-integrity check: N/A for TabSyn by design
   (unconditional; order_size_grp is learned as a column).
   Replaced in csv mode by: both classes present + the
   generated proportions vs training proportions report
   (contract rule 3).

STANDING RULE 2 (every checker verified on a known-answer
fixture before being trusted): --selftest plants each
fault into clean real data and verifies each detector
fires on the fault and stays quiet on clean data:
   range-passes-but-set-catches (aisle_id 67.5),
   mode collapse (coverage), drifting draws (stability),
   shifted small-draw (scale), plus clean passes.

MODES
  --mode selftest
      fixture verification only (writes selftest JSON)
  --mode csv --synthetic-csv F
      audit one saved draw file (known-answer mode: run on
      data/synthetic_v3.csv -> expect ZERO issues: 0 illegal
      values, full coverage, both classes present, matching
      the Jul 17 verification record)
      optional: --draws f1 f2 f3 f4 f5   (stability across
                independent saved draws, original section 3)
                --scale-pair small.csv big.csv (original
                section 6)

READS:  data/v3_train.csv, the given CSVs
WRITES: <--out-dir>/model_audit.json (or selftest JSON)
"""

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument('--mode', required=True,
                 choices=['csv', 'selftest'])
_ap.add_argument('--synthetic-csv')
_ap.add_argument('--draws', nargs='*', default=[])
_ap.add_argument('--scale-pair', nargs=2)
_ap.add_argument('--out-dir', required=True)
_args = _ap.parse_args()
OUT_DIR = _args.out_dir

import pandas as pd
import numpy as np
import random
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs(OUT_DIR, exist_ok=True)
assert os.path.normpath(OUT_DIR).replace('\\', '/').find(
    'results/tabsyn') != -1, \
    'FATAL: out-dir must live under results/tabsyn (protection rules)'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

COND_COL = 'order_size_grp'
SMALL_MAX = 10
LARGE_MIN = 14
BASE_COLS = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_reorder', 'is_early_in_cart',
    'days_since_prior_order',
]

# Discrete columns: only a fixed SET of values is legal.
# days_since_prior_order was declared NUMERICAL during
# training, so continuous output is legitimate for it -
# a range check is the correct test there.
DISCRETE = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_reorder', 'is_early_in_cart',
]

print("=" * 70)
print("AUDIT THE GENERATOR ARTEFACT (tabsyn bench, "
      "adapted copy)")
print(f"mode: {_args.mode}")
print("=" * 70)

# ══════════════════════════════════════════
# [0] GROUND TRUTH - derived, never assumed
#     (VERBATIM from audit_model_artifact.py)
# ══════════════════════════════════════════
print("\n[0] Deriving what is LEGAL from v3_train.csv")

train = pd.read_csv('data/v3_train.csv')

VALID = {}
for c in BASE_COLS:
    v = pd.to_numeric(train[c], errors='coerce')
    VALID[c] = (float(v.min()), float(v.max()))

VALID_SET = {}
for c in DISCRETE:
    VALID_SET[c] = set(pd.to_numeric(
        train[c], errors='coerce').dropna().unique())

print(f"    {'column':<26}{'min':>8}{'max':>8}"
      f"{'#legal':>9}")
print("    " + "-" * 52)
for c, (lo, hi) in VALID.items():
    n_set = (len(VALID_SET[c])
             if c in VALID_SET else '-')
    print(f"    {c:<26}{lo:>8.1f}{hi:>8.1f}"
          f"{str(n_set):>9}")


# ══════════════════════════════════════════
# THE CHECKS - logic verbatim, df-parameterized
# ══════════════════════════════════════════
def check_range(big, issues):
    """Original section 2a."""
    print("    (a) RANGE")
    print(f"    {'column':<26}{'min':>8}{'max':>8}"
          f"{'bad':>8}")
    print("    " + "-" * 52)
    for c, (lo, hi) in VALID.items():
        v = pd.to_numeric(big[c], errors='coerce')
        bad = int(((v < lo) | (v > hi)).sum()
                  + v.isna().sum())
        if bad:
            issues.append(f"{c}: {bad} out-of-range values")
        print(f"    {c:<26}{v.min():>8.1f}{v.max():>8.1f}"
              f"{bad:>8,}")


def check_set(big, issues):
    """Original section 2b - catches what a range check
    misses (e.g. aisle_id = 67.5 is inside (1,134) but is
    not a real aisle)."""
    print("\n    (b) SET MEMBERSHIP")
    print(f"    {'column':<26}{'illegal values':>18}"
          f"{'rows':>8}")
    print("    " + "-" * 54)
    for c in DISCRETE:
        v = pd.to_numeric(big[c], errors='coerce')
        illegal = sorted(set(v.dropna().unique())
                         - VALID_SET[c])
        n_rows = int(v.isin(illegal).sum()) if illegal else 0
        if illegal:
            issues.append(
                f"{c}: {len(illegal)} illegal value(s) "
                f"in {n_rows} rows")
            print(f"    {c:<26}{str(illegal[:4]):>18}"
                  f"{n_rows:>8,}")
        else:
            print(f"    {c:<26}{'none':>18}{0:>8}")


def check_coverage(big, issues):
    """Original section 2c - catches MODE COLLAPSE.
    Baseline = coverage a PERFECT model reaches at this
    sample size, measured by resampling the REAL data -
    never an assumed 100%."""
    print("\n    (c) COVERAGE - catches mode collapse")
    print(f"    {'column':<24}{'model':>7}{'real':>7}"
          f"{'cov':>8}{'baseline':>10}")
    print("    " + "-" * 58)
    for c in DISCRETE:
        n_model = int(pd.to_numeric(
            big[c], errors='coerce').nunique())
        n_real = len(VALID_SET[c])
        cov = n_model / n_real if n_real else 0

        boot = train[c].sample(
            n=len(big), replace=True,
            random_state=SEED)
        base_cov = (int(pd.to_numeric(
            boot, errors='coerce').nunique()) / n_real
            if n_real else 0)

        ok = cov >= base_cov * 0.5
        if not ok:
            issues.append(
                f"{c}: MODE COLLAPSE - emits {n_model}/"
                f"{n_real} values ({cov:.1%}); a faithful "
                f"model reaches {base_cov:.1%}")
        print(f"    {c:<24}{n_model:>7}{n_real:>7}"
              f"{cov:>7.1%}{base_cov:>10.1%}"
              f"{'' if ok else '  <-- COLLAPSE'}")


def check_cond_values(big, issues):
    """Original section 2d + contract rule 3 proportions
    (adapted: TabSyn learns order_size_grp as a column)."""
    cvals = set(big[COND_COL].astype(str).unique())
    print(f"\n    (d) {COND_COL}: {sorted(cvals)}")
    if not cvals <= {'small', 'large'}:
        issues.append("unexpected order_size_grp values")
        print("        [!] unexpected class values")
    elif len(cvals) < 2:
        issues.append(
            f"{COND_COL} collapsed to one class")
        print("        [!] COLLAPSED - only one class")
    else:
        print("        both trained classes present")

    # contract rule 3: generated vs training proportions
    tsize = train.groupby('order_id').size()
    t = train.copy()
    t['n_items'] = t['order_id'].map(tsize)
    t[COND_COL + '_re'] = np.where(
        t['n_items'] <= SMALL_MAX, 'small',
        np.where(t['n_items'] >= LARGE_MIN,
                 'large', 'mid'))
    tshare = t[t[COND_COL + '_re'] != 'mid'][
        COND_COL + '_re'].value_counts(normalize=True)
    gshare = big[COND_COL].astype(str).value_counts(
        normalize=True)
    print(f"        {'group':<8}{'training':>10}"
          f"{'generated':>11}")
    for k in sorted(set(list(tshare.index)
                        + list(gshare.index))):
        print(f"        {k:<8}{tshare.get(k, 0.0):>10.4f}"
              f"{gshare.get(k, 0.0):>11.4f}")
    return ({k: float(tshare.get(k, 0.0))
             for k in ['small', 'large']},
            {k: float(gshare.get(k, 0.0))
             for k in gshare.index})


def check_stability(draw_dfs, issues, label="draws"):
    """Original section 3 - marginal stability across
    independent draws, judged against 3*sigma/sqrt(n)
    measured from the data itself (never a fixed %)."""
    N_DRAW = min(len(d) for d in draw_dfs)
    rows = []
    for d in draw_dfs:
        d = d.iloc[:N_DRAW]
        row = {c: float(pd.to_numeric(
            d[c], errors='coerce').mean())
            for c in BASE_COLS}
        row['small_frac'] = float(
            (d[COND_COL].astype(str) == 'small').mean())
        rows.append(row)
    dd = pd.DataFrame(rows)
    ref = draw_dfs[0].iloc[:N_DRAW]

    print(f"    {len(draw_dfs)} {label}, "
          f"n={N_DRAW:,} each")
    print(f"    {'metric':<24}{'mean':>9}{'spread':>9}"
          f"{'expected':>10}{'':>10}")
    print("    " + "-" * 62)
    for c in dd.columns:
        m = dd[c].mean()
        spread = dd[c].max() - dd[c].min()

        if c == 'small_frac':
            pv = float((ref[COND_COL].astype(str)
                        == 'small').mean())
            sigma = float(np.sqrt(pv * (1 - pv)))
        else:
            sigma = float(pd.to_numeric(
                ref[c], errors='coerce').std())

        expected = 3.0 * sigma / np.sqrt(N_DRAW)
        # sigma == 0 means the column COLLAPSED - spread 0
        # would read "perfectly stable"; flag it instead.
        if sigma == 0:
            ok = False
            issues.append(
                f"{c} COLLAPSED (zero variance)")
        else:
            ok = spread <= expected * 2.0
            if not ok:
                issues.append(
                    f"{c} unstable across draws "
                    f"({spread:.4f} vs {expected:.4f} "
                    f"expected)")
        print(f"    {c:<24}{m:>9.3f}{spread:>9.4f}"
              f"{expected:>10.4f}"
              f"{'OK' if ok else 'UNSTABLE':>10}")


def check_scale(d5, d50, issues):
    """Original section 6 - does a small draw behave like
    a large draw, judged against sampling noise at the
    small n."""
    print(f"    {'metric':<24}{'small':>9}{'large':>9}"
          f"{'diff':>9}{'expect':>9}{'':>6}")
    print("    " + "-" * 66)
    for c in BASE_COLS:
        v5 = pd.to_numeric(d5[c], errors='coerce')
        v50 = pd.to_numeric(d50[c], errors='coerce')
        m5, m50 = float(v5.mean()), float(v50.mean())
        diff = abs(m5 - m50)
        sigma = float(v50.std())
        expected = 3.0 * sigma / np.sqrt(len(d5))
        ok = diff <= expected * 2.0
        if not ok:
            issues.append(f"{c} scale-dependent")
        print(f"    {c:<24}{m5:>9.3f}{m50:>9.3f}"
              f"{diff:>9.4f}{expected:>9.4f}"
              f"{'OK' if ok else 'BAD':>6}")


# ══════════════════════════════════════════
# SELFTEST - standing rule 2 fixtures
# ══════════════════════════════════════════
if _args.mode == 'selftest':
    print("\n" + "=" * 70)
    print("SELFTEST - every detector vs planted fault "
          "+ clean data")
    print("=" * 70)
    results = {}
    rng = np.random.RandomState(SEED)
    clean = train.sample(n=50000, random_state=SEED
                         ).reset_index(drop=True).copy()
    # give the clean fixture a conditioning column the
    # checks expect (recomputed, screened buckets)
    tsize = train.groupby('order_id').size()
    ni = clean['order_id'].map(tsize)
    clean[COND_COL] = np.where(
        ni <= SMALL_MAX, 'small',
        np.where(ni >= LARGE_MIN, 'large', 'mid'))
    clean = clean[clean[COND_COL] != 'mid'
                  ].reset_index(drop=True)

    # 1. clean passes range+set+coverage+cond
    iss = []
    print("\n  [fixture 1] CLEAN real sample "
          "-> expect ZERO issues")
    check_range(clean, iss)
    check_set(clean, iss)
    check_coverage(clean, iss)
    check_cond_values(clean, iss)
    results['clean_passes'] = (len(iss) == 0)
    print(f"  -> issues: {iss if iss else 'none'}  "
          f"[{'PASS' if results['clean_passes'] else 'FAIL'}]")

    # 2. aisle_id 67.5: RANGE must pass, SET must fire
    print("\n  [fixture 2] planted aisle_id=67.5 "
          "-> range PASSES, set FIRES")
    bad = clean.copy()
    bad.loc[0, 'aisle_id'] = 67.5
    iss_r, iss_s = [], []
    check_range(bad, iss_r)
    check_set(bad, iss_s)
    results['range_blind_to_67_5'] = not any(
        'aisle_id' in i for i in iss_r)
    results['set_catches_67_5'] = any(
        'aisle_id' in i and 'illegal' in i
        for i in iss_s)
    print(f"  -> range fired: {not results['range_blind_to_67_5']} "
          f"(must be False), set fired: "
          f"{results['set_catches_67_5']} (must be True)")

    # 3. mode collapse: coverage must fire
    print("\n  [fixture 3] aisle_id collapsed to 24 "
          "-> coverage FIRES")
    col = clean.copy()
    col['aisle_id'] = 24
    iss_c = []
    check_coverage(col, iss_c)
    results['coverage_catches_collapse'] = any(
        'MODE COLLAPSE' in i for i in iss_c)
    print(f"  -> fired: {results['coverage_catches_collapse']}"
          f" (must be True)")

    # 4. stability: clean same-distribution draws pass,
    #    drifting draws fire
    print("\n  [fixture 4] stability - 5 same-dist draws "
          "pass; drifting draws fire")

    # Each fixture draw gets a REAL conditioning column
    # (recomputed from order_id, screened 10/14 buckets,
    # mid dropped - same derivation as fixture 1). A
    # constant placeholder here makes small_frac degenerate
    # (sigma 0) and trips the collapse guard on clean data.
    def _fixture_draw(s):
        d = train.sample(n=20000, random_state=s).copy()
        ni = d['order_id'].map(tsize)
        d[COND_COL] = np.where(
            ni <= SMALL_MAX, 'small',
            np.where(ni >= LARGE_MIN, 'large', 'mid'))
        return d[d[COND_COL] != 'mid'
                 ].reset_index(drop=True)

    draws_clean = [_fixture_draw(s)
                   for s in [1, 2, 3, 4, 5]]
    iss_ok = []
    check_stability(draws_clean, iss_ok,
                    label="clean draws")
    stable_clean = (len(iss_ok) == 0)
    draws_bad = []
    for i, s in enumerate([1, 2, 3, 4, 5]):
        d = _fixture_draw(s)
        d['aisle_id'] = pd.to_numeric(
            d['aisle_id']).clip(1, 134) * 1.0 + i * 2
        draws_bad.append(d)
    iss_bad = []
    check_stability(draws_bad, iss_bad,
                    label="drifting draws")
    results['stability_passes_clean'] = stable_clean
    results['stability_catches_drift'] = any(
        'unstable' in i for i in iss_bad)
    print(f"  -> clean quiet: {stable_clean} (must be True), "
          f"drift fired: {results['stability_catches_drift']}"
          f" (must be True)")

    # 5. scale: matched subsamples pass, shifted fires
    print("\n  [fixture 5] scale - clean 5k-vs-50k passes; "
          "shifted 5k fires")
    d50 = train.sample(n=50000, random_state=7)
    d5 = train.sample(n=5000, random_state=8)
    iss_sc = []
    check_scale(d5, d50, iss_sc)
    scale_clean = (len(iss_sc) == 0)
    d5b = d5.copy()
    d5b['aisle_id'] = pd.to_numeric(d5b['aisle_id']) + 5
    iss_sb = []
    check_scale(d5b, d50, iss_sb)
    results['scale_passes_clean'] = scale_clean
    results['scale_catches_shift'] = any(
        'scale-dependent' in i for i in iss_sb)
    print(f"  -> clean quiet: {scale_clean} (must be True), "
          f"shift fired: {results['scale_catches_shift']}"
          f" (must be True)")

    all_pass = all(results.values())
    print("\n" + "=" * 70)
    print(f"SELFTEST VERDICT: "
          f"{'ALL DETECTORS VERIFIED' if all_pass else 'FAILURES'}")
    for k, v in results.items():
        print(f"    {k:<32} {'PASS' if v else 'FAIL'}")
    print("=" * 70)
    json.dump(results, open(os.path.join(
        OUT_DIR, 'model_audit_selftest.json'), 'w'),
        indent=2)
    print(f"Saved "
          f"{os.path.join(OUT_DIR, 'model_audit_selftest.json')}")
    raise SystemExit(0 if all_pass else 1)


# ══════════════════════════════════════════
# CSV MODE - audit a saved draw file
# ══════════════════════════════════════════
assert _args.synthetic_csv, \
    '--synthetic-csv required in csv mode'
issues = []

print("\n" + "=" * 70)
print(f"[1] LOAD SAVED DRAW: {_args.synthetic_csv}")
print("=" * 70)
big = pd.read_csv(_args.synthetic_csv)
missing = [c for c in BASE_COLS + [COND_COL]
           if c not in big.columns]
if missing:
    print(f"    [!] MISSING COLUMNS: {missing}")
    issues.append(f"missing columns {missing}")
    raise SystemExit(1)
print(f"    {len(big):,} rows, all "
      f"{len(BASE_COLS) + 1} expected columns present")

print("\n" + "=" * 70)
print("[2] VALUE VALIDITY")
print("=" * 70)
check_range(big, issues)
check_set(big, issues)
check_coverage(big, issues)
tprops, gprops = check_cond_values(big, issues)

if _args.draws:
    print("\n" + "=" * 70)
    print("[3] SAMPLE STABILITY - independent saved draws")
    print("=" * 70)
    draw_dfs = [pd.read_csv(f) for f in _args.draws]
    check_stability(draw_dfs, issues)
else:
    print("\n[3] STABILITY: no --draws given - run with 5")
    print("    independent saved draws to audit this axis.")

print("\n[4] REPRODUCIBILITY: N/A for saved-file audit.")
print("    Same known limitation as CTGAN (model_audit.json):")
print("    sampling is not seed-reproducible - the SAVED CSV")
print("    is the artifact of record for both generators.")

print("\n[5] CONDITIONING INTEGRITY: N/A for TabSyn by")
print("    design (unconditional; order_size_grp learned as")
print("    a column). Class presence + proportions reported")
print("    in section [2d] instead (contract rule 3).")

if _args.scale_pair:
    print("\n" + "=" * 70)
    print("[6] SCALE INVARIANCE - small vs large saved draw")
    print("=" * 70)
    d5 = pd.read_csv(_args.scale_pair[0])
    d50 = pd.read_csv(_args.scale_pair[1])
    check_scale(d5, d50, issues)
else:
    print("\n[6] SCALE: no --scale-pair given - provide a")
    print("    small and a large saved draw to audit this.")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if not issues:
    print("""
    THE AUDITED DRAW IS SOUND on every checked axis.
    Every value it emits is one that actually exists in
    the real data - no impossible aisles, no fractional
    hours, no mode collapse.
""")
else:
    print(f"\n    {len(issues)} ISSUE(S) FOUND:\n")
    for i in issues:
        print(f"      - {i}")

json.dump({
    'audited_file': _args.synthetic_csv,
    'draws': _args.draws,
    'scale_pair': _args.scale_pair,
    'training_proportions': tprops,
    'generated_proportions': gprops,
    'issues': issues,
}, open(os.path.join(OUT_DIR, 'model_audit.json'), 'w'),
    indent=2)
print(f"Saved {os.path.join(OUT_DIR, 'model_audit.json')}")
print("=" * 70)
