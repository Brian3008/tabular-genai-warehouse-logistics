"""
check_testA_leak.py
===================
Test A claimed aisle MIX predicts basket size
with +45.5% lift. That is suspiciously strong.

THE CONCERN
-----------
The classifier sees 134 aisle PROPORTIONS per
order. But proportions have different GRANULARITY
depending on basket size:

    3-item basket  -> proportions can only be
                      0.333, 0.667, 1.000
    40-item basket -> proportions like 0.025,
                      0.075, 0.125...

So the classifier might be reading basket size off
the *arithmetic granularity* of the numbers, not
off WHICH aisles are in the mix. That would be
leakage wearing a convincing disguise - exactly
the kind of thing that produced the fake 98.7%.

THE FIX
-------
Rerun Test A using ONLY the first 3 items of every
order. Now every order contributes exactly 3 rows,
so granularity is IDENTICAL across all orders
(proportions can only be 0.333/0.667/1.0 for
everyone). Any remaining lift must come from
WHICH aisles are present - real signal.

CONTROL
-------
We also shuffle the labels and rerun. A shuffled
run should give ~0% lift. If it does not, the test
itself is broken.

READING THE RESULT
------------------
  lift stays high  -> Test A was clean.
                      Aisle mix genuinely predicts
                      basket size.
  lift collapses   -> Test A was granularity
                      leakage. DISCARD IT.
                      Rest the case on Tests B & C,
                      which are already clean.
                      Conclusion unchanged, claim
                      stays honest.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 64)
print("IS TEST A REAL, OR GRANULARITY LEAKAGE?")
print("=" * 64)

print("\nLoading...")
df = pd.read_csv('data/clean_orders_v3.csv',
                 nrows=6000000)
print(f"Rows:   {len(df):,}")

# ── SAME SIZE GROUPS AS TEST A ──
osize = df.groupby('order_id').size()
q33, q67 = osize.quantile([0.33, 0.67])

def grp(n):
    if n <= q33: return 'small'
    if n >= q67: return 'large'
    return 'mid'

df['size_grp'] = df['order_id'].map(
    osize.apply(grp))
print(f"Small <= {q33:.0f} items, "
      f"Large >= {q67:.0f} items")


def run_clf(mix, y, tag):
    base = y.value_counts(normalize=True).max()
    Xtr, Xte, ytr, yte = train_test_split(
        mix, y, test_size=0.3,
        random_state=SEED, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=120, max_depth=16,
        random_state=SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(
        yte, clf.predict(Xte))
    lift = acc - base
    print(f"\n  {tag}")
    print(f"    Baseline: {base:.1%}")
    print(f"    Accuracy: {acc:.1%}")
    print(f"    Lift:     {lift:+.1%}")
    return lift


# ══════════════════════════════════════════
# ORIGINAL TEST A (for reference)
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("[1] ORIGINAL TEST A - all items")
print("    (granularity DIFFERS by basket size)")
print("=" * 64)

sub = df[df['size_grp'].isin(['small', 'large'])]
oids = sub['order_id'].drop_duplicates()
if len(oids) > 100000:
    oids = oids.sample(n=100000,
                       random_state=SEED)
sub_all = sub[sub['order_id'].isin(oids)]

mix_all = pd.crosstab(
    sub_all['order_id'], sub_all['aisle_id'],
    normalize='index')
y_all = sub_all.groupby('order_id')[
    'size_grp'].first().loc[mix_all.index]

print(f"\n  orders: {mix_all.shape[0]:,}")
lift_orig = run_clf(
    mix_all, y_all,
    "All items (original Test A)")


# ══════════════════════════════════════════
# THE REAL TEST - identical granularity
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("[2] GRANULARITY-CONTROLLED TEST A")
print("    first 3 items only - EVERY order")
print("    contributes exactly 3 rows")
print("=" * 64)

f3 = sub_all[sub_all['is_early_in_cart'] == 1]

# keep only orders that actually have 3 items
# in the first-3 slice, so granularity is
# truly identical for everyone
cnt = f3.groupby('order_id').size()
keep = cnt[cnt == 3].index
f3 = f3[f3['order_id'].isin(keep)]

mix_f3 = pd.crosstab(
    f3['order_id'], f3['aisle_id'],
    normalize='index')
y_f3 = f3.groupby('order_id')[
    'size_grp'].first().loc[mix_f3.index]

print(f"\n  orders with exactly 3 early items: "
      f"{mix_f3.shape[0]:,}")
print(f"  every row sums to 1.0 from exactly")
print(f"  3 items -> granularity IDENTICAL")
print(f"\n  class balance:")
for k, v in y_f3.value_counts(
        normalize=True).items():
    print(f"    {k:<8} {v:.1%}")

lift_ctrl = run_clf(
    mix_f3, y_f3,
    "First 3 items (granularity controlled)")


# ══════════════════════════════════════════
# SANITY CONTROL - shuffled labels
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("[3] SANITY CHECK - shuffled labels")
print("    (should give ~0% lift; if not,")
print("     the test itself is broken)")
print("=" * 64)

y_shuf = pd.Series(
    np.random.permutation(y_f3.values),
    index=y_f3.index)
lift_shuf = run_clf(
    mix_f3, y_shuf,
    "Shuffled labels (control)")


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("VERDICT")
print("=" * 64)
print(f"\n  Original Test A (all items):   "
      f"{lift_orig:+.1%}")
print(f"  Granularity-controlled:        "
      f"{lift_ctrl:+.1%}")
print(f"  Shuffled control (should ~0):  "
      f"{lift_shuf:+.1%}")

if abs(lift_shuf) > 0.03:
    print("""
  WARNING: shuffled control is NOT ~0.
  The test harness itself is unreliable.
  Do not trust any of these numbers.
""")
elif lift_ctrl > 0.10:
    print(f"""
  TEST A IS CLEAN.

  With granularity held identical across every
  order, aisle mix STILL predicts basket size
  at {lift_ctrl:+.1%} lift (shuffled control:
  {lift_shuf:+.1%}).

  The signal comes from WHICH aisles appear,
  not from arithmetic. Order size is confirmed
  as a genuine conditioning axis on all three
  tests.

  => Safe to report Test A. Build the
     conditioning on order size.
""")
elif lift_ctrl > 0.03:
    print(f"""
  PARTIALLY CLEAN.

  Controlled lift {lift_ctrl:+.1%} is real but
  much weaker than the original {lift_orig:+.1%}.
  Most of Test A's headline number WAS
  granularity leakage.

  => DO NOT report the {lift_orig:+.1%} figure.
     Report {lift_ctrl:+.1%}, and rest the main
     case on Tests B and C (tail TVD 0.18/0.15),
     which are clean by construction.

  => Order size is still a valid axis. The
     claim is just smaller than it looked.
""")
else:
    print(f"""
  TEST A WAS LEAKAGE.

  Once granularity is controlled, lift collapses
  to {lift_ctrl:+.1%}. The +45.5% was the
  classifier reading basket size off the
  arithmetic of the proportions.

  => DISCARD Test A entirely.

  => Tests B and C still PASS and are clean by
     construction (tail TVD 0.1842 and 0.1536,
     vs 0.036 weekend control). Order size
     remains a genuine axis - the aisle mix
     really does shift. We simply cannot claim
     a classifier separates them at 99%.

  => Conclusion unchanged. Claim now honest.
""")
print("=" * 64)