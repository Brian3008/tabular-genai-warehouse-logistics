"""
tabsyn_association_audit.py
===========================
Verbatim copy of association_audit.py for the TabSyn bench
(tabsyn_bench_contract.md). Measurement code is
character-identical to the original; deviations declared.

DECLARED DEVIATIONS (nothing else changed):
 1. I/O head: --synthetic-csv and --out-dir replace the
    hard-coded 'data/synthetic_v3.csv' and the hard-coded
    'data/association_audit.csv' output. The original CSV
    is never touched.
 2. Docstring narrative trimmed to this header; the
    original's full reasoning (Cramer's V inflation trap,
    permutation-null safeguard, fixture verification) is
    in association_audit.py and applies unchanged - the
    CODE below is its code verbatim.

The synthetic file MUST carry an order_size_grp column
(CTGAN: conditioned on it; TabSyn: learned as a 7th
column). The real side recomputes it from order_id with
the screened 10/14 buckets - never the stored column
(project-notes caveat 4).

READS:  data/v3_train.csv, data/v3_train_order_ids.csv,
        data/v3_eval.csv, <--synthetic-csv>
WRITES: <--out-dir>/association_audit.csv

Known-answer gate (contract rule 1): with
--synthetic-csv data/synthetic_v3.csv this must reproduce
data/association_audit.csv exactly (deterministic: seeded
subsample + seeded permutation nulls) - 21 pairs, weak-pair
mean inflation +0.0585, strong-pair +0.0304.
"""

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument('--synthetic-csv', required=True)
_ap.add_argument('--out-dir', required=True)
_args = _ap.parse_args()
SYNTH_CSV = _args.synthetic_csv
OUT_DIR = _args.out_dir

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

os.makedirs(OUT_DIR, exist_ok=True)
assert os.path.normpath(OUT_DIR).replace('\\', '/').find(
    'results/tabsyn') != -1, \
    'FATAL: out-dir must live under results/tabsyn (protection rules)'

SEED = 42
np.random.seed(SEED)

NULL_REPS = 25       # permutations per pair
SMALL_MAX = 10
LARGE_MIN = 14
COND_COL = 'order_size_grp'

# The six trained columns. days_since_prior_order is
# continuous - we bin it so Cramer's V applies.
COLS = [
    'aisle_id',
    'order_dow',
    'order_hour_of_day',
    'is_reorder',
    'is_early_in_cart',
    'days_since_prior_order',
]

print("=" * 70)
print("ASSOCIATION AUDIT (tabsyn bench copy)")
print("Does the generator fabricate structure where none exists?")
print(f"scored file: {SYNTH_CSV}")
print("=" * 70)


# ══════════════════════════════════════════
# METRIC
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


# ══════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════
print("\n[1] Loading ...")
train = pd.read_csv('data/v3_train.csv')
synth = pd.read_csv(SYNTH_CSV)

tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
ev_ids = set(pd.read_csv(
    'data/v3_eval.csv')['order_id'])
assert len(tr_ids & ev_ids) == 0
assert set(train['order_id']) <= tr_ids
print("    [OK] disjointness verified")

# rebuild the conditioning column on the real side
osize = train.groupby('order_id').size()
train['n_items'] = train['order_id'].map(osize)
train[COND_COL] = np.where(
    train['n_items'] <= SMALL_MAX, 'small',
    np.where(train['n_items'] >= LARGE_MIN,
             'large', 'mid'))
real = train[train[COND_COL] != 'mid'].copy()

print(f"    real:      {len(real):,} rows")
print(f"    synthetic: {len(synth):,} rows")

# bin the continuous column so Cramer's V applies,
# using the SAME bin edges on both sides
edges = [-1, 0, 7, 14, 30, 999]
for d in (real, synth):
    d['dspo_bin'] = pd.cut(
        pd.to_numeric(d['days_since_prior_order'],
                      errors='coerce').fillna(0),
        bins=edges).astype(str)

AUDIT_COLS = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_reorder', 'is_early_in_cart', 'dspo_bin',
    COND_COL,
]

# ══════════════════════════════════════════
# EQUAL SAMPLE SIZE - the critical safeguard
# ══════════════════════════════════════════
n = min(len(real), len(synth))
print(f"\n[2] Subsampling BOTH to n = {n:,}")
print("    (Cramer's V inflates at small n on")
print("     high-cardinality columns. Comparing")
print("     272k real against 50k synthetic would")
print("     fake an inflation in the synthetic side.)")

R = real.sample(n=n, random_state=SEED)
S = synth.sample(n=n, random_state=SEED)

# sanity: both have the conditioning column
assert COND_COL in R.columns
assert COND_COL in S.columns, \
    f"{SYNTH_CSV} is missing order_size_grp"


# ══════════════════════════════════════════
# AUDIT EVERY PAIR
# ══════════════════════════════════════════
print(f"\n[3] Auditing all column pairs")
print(f"    ({NULL_REPS} permutations per pair -")
print("     this takes a few minutes)\n")

rows = []
pairs = [(AUDIT_COLS[i], AUDIT_COLS[j])
         for i in range(len(AUDIT_COLS))
         for j in range(i + 1, len(AUDIT_COLS))]

for idx, (a, b) in enumerate(pairs, 1):
    print(f"    [{idx:>2}/{len(pairs)}] "
          f"{a} x {b}", flush=True)
    rv, rn, rex = excess_v(R[a], R[b])
    sv, sn, sex = excess_v(S[a], S[b])
    if np.isnan(rex) or np.isnan(sex):
        continue
    ratio = (sex / rex) if rex > 0.005 else np.nan
    rows.append({
        'pair': f"{a} x {b}",
        'real_v': rv, 'real_excess': rex,
        'synth_v': sv, 'synth_excess': sex,
        'ratio': ratio,
    })

df = pd.DataFrame(rows).sort_values(
    'real_excess')

# ══════════════════════════════════════════
# RESULTS - ordered by REAL association
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS  (sorted: weakest real association first)")
print("=" * 70)
print(f"\n{'Pair':<40}{'realEx':>9}"
      f"{'synEx':>9}{'ratio':>9}")
print("-" * 70)

for _, r in df.iterrows():
    if np.isnan(r['ratio']):
        note = "  <- no real assoc"
        rs = "  n/a"
        if r['synth_excess'] > 0.02:
            note = "  <-- FABRICATED"
    else:
        rs = f"{r['ratio']:>8.0%}"
        if r['ratio'] > 2.0:
            note = "  <-- FABRICATED"
        elif r['ratio'] > 1.3:
            note = "  <-- over"
        elif r['ratio'] < 0.7:
            note = "  <-- under"
        else:
            note = ""
    print(f"{r['pair']:<40}"
          f"{r['real_excess']:>9.4f}"
          f"{r['synth_excess']:>9.4f}"
          f"{rs}{note}")

# ══════════════════════════════════════════
# THE PATTERN
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("IS FABRICATION WORST WHERE THE REAL")
print("ASSOCIATION IS WEAKEST?")
print("=" * 70)

weak = df[df['real_excess'] < 0.02]
strong = df[df['real_excess'] >= 0.05]

def summarise(sub, label):
    if len(sub) == 0:
        print(f"\n  {label}: none")
        return None
    inflate = (sub['synth_excess']
               - sub['real_excess']).mean()
    print(f"\n  {label}  ({len(sub)} pairs)")
    print(f"    mean real  excess: "
          f"{sub['real_excess'].mean():.4f}")
    print(f"    mean synth excess: "
          f"{sub['synth_excess'].mean():.4f}")
    print(f"    mean inflation:    {inflate:+.4f}")
    return inflate

inf_weak = summarise(
    weak, "WEAK real association (<0.02)")
inf_strong = summarise(
    strong, "STRONG real association (>=0.05)")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if inf_weak is None or inf_strong is None:
    print("\n  Not enough pairs in one bucket to")
    print("  compare. Read the table above directly.")
elif inf_weak > 0.01 and inf_weak > inf_strong:
    print(f"""
  CONFIRMED - fabrication is worst where the real
  association is weakest.

  Where the real association is WEAK, the model
  inflates it by {inf_weak:+.4f} on average.
  Where the real association is STRONG, it inflates
  by only {inf_strong:+.4f}.

  PRACTICAL CONSEQUENCE:
    Features whose real associations are weak cannot
    be trusted in the synthetic data.

    This must be reported, not calibrated away.
""")
elif inf_weak <= 0.01:
    print(f"""
  NOT CONFIRMED.

  Weak-association pairs are inflated by only
  {inf_weak:+.4f} - within noise. The fabrication is
  NOT a general property of this generator on this
  data.
""")
else:
    print(f"""
  MIXED.

  weak-pair inflation:   {inf_weak:+.4f}
  strong-pair inflation: {inf_strong:+.4f}

  Fabrication is present but not clearly worse where
  the real association is weak. Read the table and
  judge pair by pair - do not over-claim a general
  law from this.
""")

df.to_csv(os.path.join(OUT_DIR, 'association_audit.csv'),
          index=False)
print(f"Saved {os.path.join(OUT_DIR, 'association_audit.csv')}")
print("=" * 70)
