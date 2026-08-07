"""
check_order_size.py
===================
Is ORDER SIZE a genuine conditioning axis?

Why this test is different from signal_search:
  - signal_search asked: can ONE ROW's aisle predict
    basket size?  (+0.2% - basically no)
  - But a warehouse never sees one row. It sees a
    whole ORDER. The real question is whether the
    AISLE COMPOSITION of an order predicts its size.

TRAPS THIS TEST AVOIDS
----------------------
1. CIRCULARITY. department_id and aisle_popularity
   are lookups from aisle_id. They are EXCLUDED.
   Only the aisle mix itself is used.

2. ARITHMETIC TRIVIALITY. Bigger baskets obviously
   touch more distinct aisles - that is counting,
   not demand structure. So we use PROPORTIONS
   (aisle mix), never raw counts, and we also
   report a size-matched comparison.

3. THE 24/83 MASK. Aisles 24 and 83 dominate
   everything. We report the tail separately.

PASS CRITERIA (all three must hold):
  a) Order-level aisle mix predicts size with
     real lift (>10% over baseline)
  b) Tail TVD > 0.10 (shift is not just 24/83)
  c) The shift survives when we control for the
     number of items (i.e. it is not arithmetic)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 64)
print("IS ORDER SIZE A GENUINE CONDITIONING AXIS?")
print("Order-level test. No aisle-derived features.")
print("=" * 64)

print("\nLoading...")
df = pd.read_csv('data/clean_orders_v3.csv',
                 nrows=6000000)
print(f"Rows:   {len(df):,}")
print(f"Orders: {df['order_id'].nunique():,}")

DOMINANT = [24, 83]

# ══════════════════════════════════════════
# DEFINE ORDER SIZE GROUPS
# ══════════════════════════════════════════
osize = df.groupby('order_id').size()
q33, q67 = osize.quantile([0.33, 0.67])

def size_grp(n):
    if n <= q33:
        return 'small'
    if n >= q67:
        return 'large'
    return 'mid'

size_map = osize.apply(size_grp)
df['size_grp'] = df['order_id'].map(size_map)

print(f"\nSmall: <= {q33:.0f} items")
print(f"Large: >= {q67:.0f} items")
print("\nOrders per group:")
print(size_map.value_counts().to_string())


def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


# ══════════════════════════════════════════
# TEST A - ORDER-LEVEL AISLE MIX -> SIZE
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("TEST A: does an order's AISLE MIX predict")
print("its size?  (proportions, not counts)")
print("=" * 64)

sub = df[df['size_grp'].isin(['small', 'large'])]

# Build order x aisle PROPORTION matrix.
# Proportions, so basket size itself carries
# no information - only the MIX does.
print("\nBuilding order-level aisle mix matrix...")
oids = sub['order_id'].drop_duplicates()
if len(oids) > 120000:
    oids = oids.sample(n=120000,
                       random_state=SEED)
sub = sub[sub['order_id'].isin(oids)]

mix = pd.crosstab(sub['order_id'],
                  sub['aisle_id'],
                  normalize='index')
y = sub.groupby('order_id')[
    'size_grp'].first().loc[mix.index]

print(f"  matrix: {mix.shape[0]:,} orders "
      f"x {mix.shape[1]} aisles")
print("  (rows sum to 1.0 - pure mix, "
       "no size info)")

base = y.value_counts(normalize=True).max()
Xtr, Xte, ytr, yte = train_test_split(
    mix, y, test_size=0.3,
    random_state=SEED, stratify=y)
clf = RandomForestClassifier(
    n_estimators=120, max_depth=16,
    random_state=SEED, n_jobs=-1)
clf.fit(Xtr, ytr)
acc_mix = accuracy_score(
    yte, clf.predict(Xte))
lift_mix = acc_mix - base

print(f"\n  Baseline: {base:.1%}")
print(f"  Accuracy: {acc_mix:.1%}")
print(f"  Lift:     {lift_mix:+.1%}")

if lift_mix > 0.10:
    print("\n  -> Aisle MIX carries real")
    print("     information about order size.")
else:
    print("\n  -> Aisle mix does NOT predict size.")


# ══════════════════════════════════════════
# TEST B - AISLE DISTRIBUTION SHIFT
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("TEST B: does the aisle mix actually SHIFT?")
print("=" * 64)

sm = df[df['size_grp'] == 'small'][
    'aisle_id'].value_counts(normalize=True)
lg = df[df['size_grp'] == 'large'][
    'aisle_id'].value_counts(normalize=True)

full = tvd(sm, lg)

sm_t = sm[~sm.index.isin(DOMINANT)]
lg_t = lg[~lg.index.isin(DOMINANT)]
sm_t = sm_t / sm_t.sum()
lg_t = lg_t / lg_t.sum()
tail = tvd(sm_t, lg_t)

common = sorted(set(sm.index) & set(lg.index))
rho, _ = spearmanr(
    sm.reindex(common).values,
    lg.reindex(common).values)

print(f"\n  Full TVD:  {full:.4f}")
print(f"  Tail TVD:  {tail:.4f}  "
      f"(excl. aisles 24/83)")
print(f"  Rank rho:  {rho:.3f}")

print(f"\n  Small top 8: "
      f"{list(sm.head(8).index.astype(int))}")
print(f"  Large top 8: "
      f"{list(lg.head(8).index.astype(int))}")

# Biggest movers
print("\n  Aisles that shift most "
      "(small -> large):")
diff = (lg.reindex(common, fill_value=0)
        - sm.reindex(common, fill_value=0))
for a in diff.abs().sort_values(
        ascending=False).head(8).index:
    d = diff[a]
    arrow = "UP  " if d > 0 else "DOWN"
    print(f"    aisle {int(a):>3}  {arrow} "
          f"{abs(d)*100:5.2f} pts   "
          f"(small {sm.get(a,0)*100:5.2f}% -> "
          f"large {lg.get(a,0)*100:5.2f}%)")


# ══════════════════════════════════════════
# TEST C - IS IT JUST ARITHMETIC?
# ══════════════════════════════════════════
# Controls for the trivial fact that bigger
# baskets touch more aisles. We compare only
# the FIRST 3 items of every order, so every
# order contributes exactly 3 rows regardless
# of its true size. If the mix STILL differs,
# the shift is real, not counting.
print("\n" + "=" * 64)
print("TEST C: control for basket size")
print("(first 3 items of every order only -")
print(" every order contributes exactly 3 rows)")
print("=" * 64)

first3 = df[df['is_early_in_cart'] == 1]

sm3 = first3[first3['size_grp'] == 'small'][
    'aisle_id'].value_counts(normalize=True)
lg3 = first3[first3['size_grp'] == 'large'][
    'aisle_id'].value_counts(normalize=True)

full3 = tvd(sm3, lg3)
sm3t = sm3[~sm3.index.isin(DOMINANT)]
lg3t = lg3[~lg3.index.isin(DOMINANT)]
sm3t = sm3t / sm3t.sum()
lg3t = lg3t / lg3t.sum()
tail3 = tvd(sm3t, lg3t)

print(f"\n  Size-controlled full TVD: {full3:.4f}")
print(f"  Size-controlled tail TVD: {tail3:.4f}")
print(f"\n  Small top 5: "
      f"{list(sm3.head(5).index.astype(int))}")
print(f"  Large top 5: "
      f"{list(lg3.head(5).index.astype(int))}")

if tail3 > 0.10:
    print("\n  -> Shift SURVIVES size control.")
    print("     It is real demand structure,")
    print("     not an artefact of counting.")
else:
    print("\n  -> Shift COLLAPSES under control.")
    print("     It was mostly arithmetic.")


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("VERDICT")
print("=" * 64)

a_pass = lift_mix > 0.10
b_pass = tail > 0.10
c_pass = tail3 > 0.10

print(f"\n  A) mix predicts size      "
      f"{'PASS' if a_pass else 'FAIL'}  "
      f"(lift {lift_mix:+.1%})")
print(f"  B) aisle mix shifts       "
      f"{'PASS' if b_pass else 'FAIL'}  "
      f"(tail TVD {tail:.4f})")
print(f"  C) survives size control  "
      f"{'PASS' if c_pass else 'FAIL'}  "
      f"(tail TVD {tail3:.4f})")

if a_pass and b_pass and c_pass:
    print("""
  ALL THREE PASS.

  Order size is a GENUINE, NON-CIRCULAR
  conditioning axis. Small and large baskets
  hit measurably different shelves, and the
  difference is not an artefact of counting.

  It is also warehouse-meaningful: a 3-item
  top-up and a 40-item weekly shop are
  genuinely different picking jobs.

  => BUILD THE CONDITIONING ON ORDER SIZE.
""")
else:
    print("""
  DOES NOT PASS ALL THREE.

  Combined with the five axes already tested
  (calendar, reorder, user frequency, cart
  depth, demand type - all flat, and dept mix
  which is circular), this settles it:

  Instacart aisle demand is structurally
  stable across order context.

  => Frame conditional generation honestly as
     controllable specification, and report
     the negative finding. It is real and
     worth reporting.
""")
print("=" * 64)