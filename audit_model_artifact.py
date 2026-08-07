"""
audit_model_artifact.py
=======================
Tests ctgan_v3.pkl - the MODEL, not one sample of it.

WHY THIS EXISTS
---------------
Every metric so far was computed from synthetic_v3.csv:
ONE 50,000-row draw. But the .pkl is the artefact you
would hand to Locus, and the thing the fleet analysis
gets built on.

If the model emits an impossible aisle_id, or the
conditioning silently leaks, or the marginals wander
between draws - no single-sample metric would catch it.

EVERY DETECTOR WAS VERIFIED ON A KNOWN-ANSWER FIXTURE
BEFORE THIS FILE WAS WRITTEN
-----------------------------------------------------
I have repeatedly written checks that were themselves
wrong - hard-coding a threshold instead of measuring it
(false alarms), or setting a boundary that lets a real
failure through. So each detector below was run against
data with a DELIBERATELY PLANTED fault, and against
clean data, to confirm it catches the fault and does not
cry wolf.

TWO BUGS WERE CAUGHT THAT WAY WHILE BUILDING THIS FILE:

  1. A RANGE CHECK IS NOT ENOUGH.
     aisle_id = 67.5 sits INSIDE the range (1, 134) and
     PASSES a range test - but it is not a real aisle,
     and a warehouse simulator looking up shelf 67.5
     would break.
     FIXED: also check membership in the SET of values
     that actually exist in the training data.

  2. THE CONDITIONING CHECK LET A LEAK THROUGH.
     My first version tested `hit >= 0.999`. With 10,000
     rows, a 10-row leak gives EXACTLY 0.999 - and
     PASSED. Conditioning is not "mostly" - it is exact.
     FIXED: require ZERO wrong rows.

Both were found by the fixtures, not by reading the code.

THRESHOLDS ARE MEASURED, NOT ASSUMED
------------------------------------
A PERFECT model still varies between draws, purely from
sampling noise. I measured it: 5 draws of 20,000 rows
from an IDENTICAL distribution vary by up to ~2%. So a
fixed "2% = unstable" rule would flag a flawless model
as broken.

Stability is therefore judged against the spread a
perfect model WOULD show - 3*sigma/sqrt(n), computed
from the data itself. Verified: passes a perfect model,
catches a drifting one.

NO RETRAINING. Under a minute.
"""

import pandas as pd
import numpy as np
import random
import torch
import json
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

COND_COL = 'order_size_grp'
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
print("AUDIT THE MODEL ARTEFACT (ctgan_v3.pkl)")
print("=" * 70)

issues = []

# ══════════════════════════════════════════
# [0] GROUND TRUTH - derived, never assumed
# ══════════════════════════════════════════
print("\n[0] Deriving what is LEGAL from v3_train.csv")
print("""
    I originally hard-coded aisle_id as (1, 134) from
    memory. If the real data differed at all, that would
    flag VALID model output as invalid - a false alarm.
    The training data is the ground truth.
""")

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
# [1] RELOAD INTEGRITY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[1] RELOAD INTEGRITY")
print("=" * 70)

try:
    model = CTGANSynthesizer.load('data/ctgan_v3.pkl')
    probe = model.sample(num_rows=100)
    print("\n    model loads and generates")
    missing = [c for c in BASE_COLS + [COND_COL]
               if c not in probe.columns]
    if missing:
        print(f"    [!] MISSING COLUMNS: {missing}")
        issues.append(f"missing columns {missing}")
    else:
        print(f"    all {len(BASE_COLS) + 1} expected "
              f"columns present")
except Exception as e:
    print(f"    [FATAL] could not load: {e}")
    raise SystemExit(1)


# ══════════════════════════════════════════
# [2] VALUE VALIDITY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[2] VALUE VALIDITY")
print("=" * 70)
print("""
    A single impossible value would break a warehouse
    simulator. TWO checks are needed - a range test
    alone is not sufficient.
""")

big = model.sample(num_rows=50000)

# --- 2a: RANGE ---
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

# --- 2b: SET MEMBERSHIP ---
# VERIFIED ON A FIXTURE: aisle_id = 67.5 PASSES the range
# check above (it is inside 1-134) yet is NOT a real
# aisle. Only this check catches it.
print("\n    (b) SET MEMBERSHIP")
print("        catches what a range check misses:")
print("        e.g. aisle_id = 67.5 is inside (1,134)")
print("        but is not a real aisle\n")
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
        print(f"      [!] not real {c} values - these")
        print(f"          would break a simulator")
    else:
        print(f"    {c:<26}{'none':>18}{0:>8}")

# --- 2c: COVERAGE - catches MODE COLLAPSE ---
# THE GAP THIS CLOSES:
# If a column collapsed to a single value (every
# aisle_id = 24), then:
#   range check  -> PASSES (24 is in range)
#   set check    -> PASSES (24 is a legal aisle)
#   stability    -> PASSES (sigma = 0, so spread = 0,
#                   and 0 <= 0 reads as "perfectly
#                   stable")
#
# A catastrophic mode collapse would be reported as
# FLAWLESS by all three. Verified on a fixture.
#
# Coverage catches it instantly: a collapsed column
# emits 1 of 134 values (0.7%); a healthy one emits
# ~100%.
print("\n    (c) COVERAGE - catches mode collapse")
print("        a collapsed column (all one value) would")
print("        PASS range, set AND stability checks.")
print("        Only coverage catches it.\n")
# BASELINE: what coverage would a PERFECT model reach?
# The real aisle distribution is heavily skewed - a few
# aisles dominate. So even a faithful model might miss a
# rare value by chance. We measure that baseline by
# resampling the REAL data to the same size, rather than
# assuming 100% is achievable.
print(f"    {'column':<24}{'model':>7}{'real':>7}"
      f"{'cov':>8}{'baseline':>10}")
print("    " + "-" * 58)
for c in DISCRETE:
    n_model = int(pd.to_numeric(
        big[c], errors='coerce').nunique())
    n_real = len(VALID_SET[c])
    cov = n_model / n_real if n_real else 0

    # what a PERFECT model achieves at this sample size
    boot = train[c].sample(
        n=len(big), replace=True,
        random_state=SEED)
    base_cov = (int(pd.to_numeric(
        boot, errors='coerce').nunique()) / n_real
        if n_real else 0)

    # collapse if we cover far less than a perfect
    # model would at the same n
    ok = cov >= base_cov * 0.5
    if not ok:
        issues.append(
            f"{c}: MODE COLLAPSE - emits {n_model}/"
            f"{n_real} values ({cov:.1%}); a faithful "
            f"model reaches {base_cov:.1%}")
    print(f"    {c:<24}{n_model:>7}{n_real:>7}"
          f"{cov:>7.1%}{base_cov:>10.1%}"
          f"{'' if ok else '  <-- COLLAPSE'}")

print("\n    'baseline' = coverage a PERFECT model")
print("    reaches at this sample size (measured by")
print("    resampling the real data). Judged against")
print("    that, not against an assumed 100%.")

# --- 2d: conditioning column ---
cvals = set(big[COND_COL].unique())
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


# ══════════════════════════════════════════
# [3] SAMPLE STABILITY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[3] SAMPLE STABILITY - 5 independent draws")
print("=" * 70)
print("""
    If the marginals wander between draws, no single
    measured number means anything.

    The threshold is MEASURED, not guessed. A perfect
    model still varies by ~3*sigma/sqrt(n) from sampling
    noise alone - a fixed 2% rule would flag it as
    broken. Verified on fixtures: passes a perfect
    model, catches a drifting one.
""")

N_DRAW = 20000
draws = []
for _ in range(5):
    d = model.sample(num_rows=N_DRAW)
    row = {c: float(pd.to_numeric(
        d[c], errors='coerce').mean())
        for c in BASE_COLS}
    row['small_frac'] = float(
        (d[COND_COL] == 'small').mean())
    draws.append(row)

dd = pd.DataFrame(draws)
ref = model.sample(num_rows=N_DRAW)

print(f"    {'metric':<24}{'mean':>9}{'spread':>9}"
      f"{'expected':>10}{'':>10}")
print("    " + "-" * 62)
for c in dd.columns:
    m = dd[c].mean()
    spread = dd[c].max() - dd[c].min()

    if c == 'small_frac':
        pv = float((ref[COND_COL] == 'small').mean())
        sigma = float(np.sqrt(pv * (1 - pv)))
    else:
        sigma = float(pd.to_numeric(
            ref[c], errors='coerce').std())

    expected = 3.0 * sigma / np.sqrt(N_DRAW)
    # sigma == 0 means the column COLLAPSED to a single
    # value. spread would also be 0, so the test would
    # read "perfectly stable" - a catastrophic failure
    # reported as flawless. Coverage (section 2d) is the
    # real detector, but flag it here too.
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


# ══════════════════════════════════════════
# [4] REPRODUCIBILITY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[4] REPRODUCIBILITY")
print("=" * 70)
print("""
    Same seed, two draws - identical? A dissertation
    result that cannot be regenerated is not a result.
""")


def seeded_sample(n):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return model.sample(num_rows=n)


a = seeded_sample(3000)
b = seeded_sample(3000)
same = a[BASE_COLS].reset_index(drop=True).equals(
    b[BASE_COLS].reset_index(drop=True))

print(f"    two seeded draws identical: {same}")
if same:
    print("    REPRODUCIBLE - same seed, same output.")
else:
    print("""
    NOT reproducible from the global seed alone. SDV
    likely holds internal RNG state that resetting
    torch / numpy / random does not touch.

    This is NOT fatal, but it changes how you report:
    cite the SAVED synthetic_v3.csv as the dataset,
    rather than claiming it can be regenerated from the
    seed. Keep that file safe.
""")
    issues.append(
        "sampling not reproducible from seed "
        "(cite the saved CSV instead)")


# ══════════════════════════════════════════
# [5] CONDITIONING INTEGRITY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[5] CONDITIONING INTEGRITY")
print("=" * 70)
print("""
    Condition on 'small' - does EVERY row come back
    small? A silent leak here would corrupt the entire
    fleet analysis.

    BUG I CAUGHT: my first version tested hit >= 0.999.
    With 10,000 rows a 10-row leak gives EXACTLY 0.999
    and PASSED. Conditioning is not "mostly" - it is
    exact. This now demands ZERO wrong rows.
""")

for grp in ['small', 'large']:
    g = model.sample_from_conditions(
        conditions=[Condition(
            num_rows=10000,
            column_values={COND_COL: grp})])
    wrong = int((g[COND_COL] != grp).sum())
    print(f"    conditioned on '{grp}': "
          f"{wrong} wrong rows out of {len(g):,}")
    if wrong > 0:
        got = list(g.loc[g[COND_COL] != grp,
                         COND_COL].unique())
        print(f"      [!] LEAK - got {got} instead")
        issues.append(
            f"conditioning leak on '{grp}': "
            f"{wrong} rows")
    else:
        print("      exact - zero leakage")


# ══════════════════════════════════════════
# [6] SCALE INVARIANCE
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[6] SCALE INVARIANCE")
print("=" * 70)
print("""
    Does a 5k draw behave like a 50k draw? The threshold
    is again the sampling noise a PERFECT model would
    show at n=5,000 - not a guessed percentage.
""")

d5 = model.sample(num_rows=5000)
d50 = model.sample(num_rows=50000)

print(f"    {'metric':<24}{'5k':>9}{'50k':>9}"
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
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if not issues:
    print("""
    THE MODEL ARTEFACT IS SOUND.

    It loads and generates. Every value it emits is one
    that actually exists in the real data - no impossible
    aisles, no fractional hours. It is stable across
    draws, conditions exactly, and behaves the same at
    any sample size.

    ctgan_v3.pkl is safe to build the fleet analysis on,
    and safe to hand to Locus.
""")
else:
    print(f"\n    {len(issues)} ISSUE(S) FOUND:\n")
    for i in issues:
        print(f"      - {i}")
    print("""
    Fix or explicitly accept each of these BEFORE the
    fleet rebuild.

    A model emitting impossible values would silently
    corrupt a warehouse simulator. A conditioning leak
    would corrupt the fleet comparison itself.
""")

json.dump({'issues': issues},
          open('data/model_audit.json', 'w'),
          indent=2)
print("Saved data/model_audit.json")
print("=" * 70)