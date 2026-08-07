import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*60)
print("IS CONTEXT->AISLE STRUCTURE IN THE")
print("REAL DATA AT ALL?")
print("="*60)

df = pd.read_csv('data/clean_orders_v2.csv',
                 nrows=2000000)
print(f"\nRows loaded: {len(df):,}")

# ── 1. MUTUAL INFORMATION ──
# How much does aisle tell us about context?
print("\n" + "="*60)
print("MUTUAL INFORMATION: aisle_id vs context")
print("(0 = no association at all)")
print("="*60)

for col in ['is_weekend', 'is_reorder',
            'is_peak_hour', 'order_dow',
            'order_hour_of_day']:
    mi = mutual_info_score(
        df['aisle_id'], df[col])
    print(f"  aisle_id ~ {col:<20} {mi:.4f}")

# Baseline: aisle vs itself-derived
mi_self = mutual_info_score(
    df['aisle_id'], df['aisle_popularity'])
print(f"\n  aisle_id ~ aisle_popularity  "
      f"{mi_self:.4f}  <-- derived, for scale")

# ── 2. PREDICT CONTEXT FROM AISLE ALONE ──
print("\n" + "="*60)
print("CAN AISLE ALONE PREDICT CONTEXT?")
print("(if yes, structure exists to learn)")
print("="*60)

sample = df.sample(n=300000,
                   random_state=SEED)
X = sample[['aisle_id']]

for target in ['is_weekend', 'is_reorder',
               'is_peak_hour']:
    y = sample[target]
    base = max(y.mean(), 1 - y.mean())
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3,
        random_state=SEED)
    clf = RandomForestClassifier(
        n_estimators=50, random_state=SEED,
        n_jobs=-1, max_depth=10)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(
        yte, clf.predict(Xte))
    lift = acc - base
    print(f"\n  Predict {target}:")
    print(f"    Majority baseline: {base:.1%}")
    print(f"    From aisle alone:  {acc:.1%}")
    print(f"    Lift:              {lift:+.1%}")

# ── 3. DOES AISLE MIX SHIFT BY CONTEXT? ──
print("\n" + "="*60)
print("DOES THE AISLE MIX ACTUALLY SHIFT?")
print("(total variation distance, 0-1)")
print("="*60)

def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()

wk = df[df['is_weekend'] == 1][
    'aisle_id'].value_counts(normalize=True)
wd = df[df['is_weekend'] == 0][
    'aisle_id'].value_counts(normalize=True)
print(f"\n  Weekend vs Weekday:   "
      f"TVD = {tvd(wk, wd):.4f}")

ro = df[df['is_reorder'] == 1][
    'aisle_id'].value_counts(normalize=True)
nr = df[df['is_reorder'] == 0][
    'aisle_id'].value_counts(normalize=True)
print(f"  Reorder vs New:       "
      f"TVD = {tvd(ro, nr):.4f}")

am = df[df['order_hour_of_day'] < 12][
    'aisle_id'].value_counts(normalize=True)
pm = df[df['order_hour_of_day'] >= 12][
    'aisle_id'].value_counts(normalize=True)
print(f"  Morning vs Afternoon: "
      f"TVD = {tvd(am, pm):.4f}")

print("\n  (TVD < 0.05 = distributions")
print("   are basically identical)")

# ── 4. TOP AISLES BY CONTEXT ──
print("\n" + "="*60)
print("TOP 5 AISLES BY CONTEXT")
print("="*60)
print("\n  Weekend:", list(
    wk.head(5).index.astype(int)))
print("  Weekday:", list(
    wd.head(5).index.astype(int)))
print("\n  Reorder:", list(
    ro.head(5).index.astype(int)))
print("  New:    ", list(
    nr.head(5).index.astype(int)))

print("\n" + "="*60)
print("READ THIS:")
print("="*60)
print("""
If TVD is tiny (<0.05) and aisle gives
near-zero lift predicting context, then
context->aisle structure DOES NOT EXIST
in Instacart. No model can learn it.
=> Honest reframing is the only path.

If TVD is meaningful (>0.10) and aisle
gives real lift, the structure IS there
and we can rebuild the conditioning to
learn it properly.
""")