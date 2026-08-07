"""
signal_search.py
================
Before concluding "there is no learnable
context->aisle signal in Instacart", test every
plausible axis properly.

Axes tested:
  1. order_size        (basket size)
  2. user_frequency    (heavy vs light shopper)
  3. department_mix    (produce-heavy vs pantry-heavy)
  4. cart_depth        (early vs late cart positions)
  5. demand_type       (reorder share - already known)
  6. is_weekend        (known negative - control)

For each axis we measure:
  - TVD on the full aisle distribution
  - TVD EXCLUDING the two dominant aisles (24, 83)
    -> because if 24/83 swamp everything, raw TVD
       hides real movement in the rest
  - Rank correlation of aisle ordering
  - Lift: can aisle alone predict the axis?

A signal is USABLE if aisle-alone lift is
meaningful (>5%) AND the tail TVD is >0.10.
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
print("SIGNAL SEARCH - test every plausible axis")
print("=" * 64)

print("\nLoading...")
df = pd.read_csv('data/clean_orders_v3.csv',
                 nrows=4000000)
print(f"Rows: {len(df):,}")
print(f"Orders: {df['order_id'].nunique():,}")

DOMINANT = [24, 83]


# ══════════════════════════════════════════
# BUILD CANDIDATE AXES
# ══════════════════════════════════════════
print("\nBuilding candidate axes...")

# --- 1. order size ---
osize = df.groupby('order_id').size()
q33, q67 = osize.quantile([0.33, 0.67])
df['order_size_grp'] = df['order_id'].map(
    osize.apply(
        lambda n: 'small' if n <= q33
        else 'large' if n >= q67
        else 'mid'))
print(f"  order_size:   small<={q33:.0f}, "
      f"large>={q67:.0f}")

# --- 2. user frequency (proxy: days_since_prior) ---
udspo = df.groupby('order_id')[
    'days_since_prior_order'].first()
u33, u67 = udspo.quantile([0.33, 0.67])
df['user_freq_grp'] = df['order_id'].map(
    udspo.apply(
        lambda d: 'frequent' if d <= u33
        else 'infrequent' if d >= u67
        else 'mid'))
print(f"  user_freq:    frequent<={u33:.0f}d, "
      f"infrequent>={u67:.0f}d")

# --- 3. department mix (produce share per order) ---
PRODUCE_DEPTS = [4]   # produce
prod_share = df.groupby('order_id')[
    'department_id'].apply(
    lambda s: s.isin(PRODUCE_DEPTS).mean())
p33, p67 = prod_share.quantile([0.33, 0.67])
df['dept_mix_grp'] = df['order_id'].map(
    prod_share.apply(
        lambda p: 'low_produce' if p <= p33
        else 'high_produce' if p >= p67
        else 'mid'))
print(f"  dept_mix:     low<={p33:.2f}, "
      f"high>={p67:.2f} produce share")

# --- 4. cart depth ---
df['cart_depth_grp'] = np.where(
    df['is_early_in_cart'] == 1,
    'early', 'late')
print(f"  cart_depth:   early (pos<=3) vs late")

# --- 5/6 already present ---
print(f"  demand_type:  (already computed)")
print(f"  is_weekend:   (control - known noise)")


# ══════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════
def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


def analyse(col, g1, g2, label):
    """Compare aisle distributions between two
    groups of an axis."""
    s1 = df[df[col] == g1]
    s2 = df[df[col] == g2]
    if len(s1) < 10000 or len(s2) < 10000:
        return None

    a1 = s1['aisle_id'].value_counts(
        normalize=True)
    a2 = s2['aisle_id'].value_counts(
        normalize=True)

    full_tvd = tvd(a1, a2)

    # exclude dominant aisles, renormalise
    t1 = a1[~a1.index.isin(DOMINANT)]
    t2 = a2[~a2.index.isin(DOMINANT)]
    t1 = t1 / t1.sum()
    t2 = t2 / t2.sum()
    tail_tvd = tvd(t1, t2)

    # rank correlation of aisle ordering
    common = sorted(
        set(a1.index) & set(a2.index))
    rho, _ = spearmanr(
        a1.reindex(common).values,
        a2.reindex(common).values)

    # top-5 lists
    top1 = list(a1.head(5).index.astype(int))
    top2 = list(a2.head(5).index.astype(int))

    return {
        'label':    label,
        'full_tvd': full_tvd,
        'tail_tvd': tail_tvd,
        'rho':      rho,
        'top1':     top1,
        'top2':     top2,
        'n1':       len(s1),
        'n2':       len(s2),
    }


def aisle_lift(col, groups):
    """Can aisle_id ALONE predict this axis?
    is_reorder deliberately withheld."""
    s = df[df[col].isin(groups)]
    if len(s) > 400000:
        s = s.sample(n=400000,
                     random_state=SEED)
    y = s[col]
    base = y.value_counts(
        normalize=True).max()
    X = s[['aisle_id']]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3,
        random_state=SEED, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=60, max_depth=12,
        random_state=SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(
        yte, clf.predict(Xte))
    return base, acc, acc - base


# ══════════════════════════════════════════
# RUN ALL AXES
# ══════════════════════════════════════════
AXES = [
    ('order_size_grp',  'small',       'large',
     'Order size (small vs large)'),
    ('user_freq_grp',   'frequent',    'infrequent',
     'User freq (frequent vs infrequent)'),
    ('dept_mix_grp',    'low_produce', 'high_produce',
     'Dept mix (low vs high produce)'),
    ('cart_depth_grp',  'early',       'late',
     'Cart depth (early vs late)'),
    ('demand_type',     'novelty',     'replenishment',
     'Demand type (novelty vs replen)'),
    ('is_weekend',      0,             1,
     'Weekend (CONTROL - known noise)'),
]

print("\n" + "=" * 64)
print("AISLE DISTRIBUTION SHIFT BY AXIS")
print("=" * 64)
print(f"\n{'Axis':<36}{'FullTVD':>9}"
      f"{'TailTVD':>9}{'Rank rho':>10}")
print("-" * 64)

results = []
for col, g1, g2, label in AXES:
    r = analyse(col, g1, g2, label)
    if r is None:
        print(f"{label:<36}  (too few rows)")
        continue
    results.append((col, g1, g2, r))
    print(f"{r['label']:<36}"
          f"{r['full_tvd']:>9.4f}"
          f"{r['tail_tvd']:>9.4f}"
          f"{r['rho']:>10.3f}")

print("\n  TailTVD = shift EXCLUDING aisles 24/83")
print("  (24 and 83 dominate everything, so the")
print("   tail is where real movement would show)")
print("  Rank rho near 1.0 = aisle ordering")
print("   basically unchanged")

# ── TOP AISLE LISTS ──
print("\n" + "=" * 64)
print("TOP 5 AISLES PER GROUP")
print("=" * 64)
for col, g1, g2, r in results:
    same = "SAME" if r['top1'] == r['top2'] \
        else "DIFFERENT"
    print(f"\n  {r['label']}   [{same}]")
    print(f"    {str(g1):<14} {r['top1']}")
    print(f"    {str(g2):<14} {r['top2']}")

# ── PREDICTIVE LIFT ──
print("\n" + "=" * 64)
print("CAN AISLE ALONE PREDICT THE AXIS?")
print("(is_reorder withheld - no shortcuts)")
print("=" * 64)
print(f"\n{'Axis':<36}{'Base':>8}"
      f"{'Acc':>8}{'Lift':>9}")
print("-" * 64)

lifts = {}
for col, g1, g2, r in results:
    base, acc, lift = aisle_lift(
        col, [g1, g2])
    lifts[col] = lift
    print(f"{r['label']:<36}"
          f"{base:>7.1%}{acc:>8.1%}"
          f"{lift:>+9.1%}")

# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 64)
print("VERDICT")
print("=" * 64)

usable = []
for col, g1, g2, r in results:
    if col == 'is_weekend':
        continue
    if lifts[col] > 0.05 and r['tail_tvd'] > 0.10:
        usable.append((r['label'], lifts[col],
                       r['tail_tvd']))

if usable:
    print("\n  USABLE AXES FOUND:\n")
    for lab, lift, tt in sorted(
            usable, key=lambda x: -x[1]):
        print(f"    {lab}")
        print(f"      aisle-alone lift: {lift:+.1%}")
        print(f"      tail TVD:         {tt:.4f}")
    print("\n  => Build conditioning on the")
    print("     strongest of these. The signal")
    print("     is genuinely there to learn.")
else:
    print("""
  NO AXIS PASSES BOTH TESTS.

  Every plausible conditioning axis fails to
  move aisle demand meaningfully. Aisles 24
  and 83 dominate under all conditions, and
  aisle alone cannot predict any context.

  This is now a MEASURED finding across six
  axes, not an assumption:

    Grocery e-commerce aisle demand is
    structurally stable across order context.

  => Conditional generation must be framed as
     controllable specification, not emergent
     discovery. That is honest and defensible.

  => AND it is a real negative result worth
     reporting: warehouse planners assuming
     seasonal aisle shifts in grocery would
     be wrong.

  => For Locus (non-grocery), the signal may
     well exist. The method is sound; this
     dataset simply lacks the structure.
""")
print("=" * 64)