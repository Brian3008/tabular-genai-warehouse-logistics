import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*62)
print("IS demand_type REAL, OR JUST is_reorder")
print("WEARING A HAT?")
print("="*62)

df = pd.read_csv('data/clean_orders_v3.csv',
                 nrows=3000000)
print(f"\nRows: {len(df):,}")

def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()

# ── THE KEY TEST ──
# Hold is_reorder CONSTANT. Within rows that are
# all reorders, does demand_type still shift aisle?
# If yes -> demand_type carries real order-level
#           information beyond the row's own flag.
# If no  -> it is just is_reorder aggregated.
print("\n" + "="*62)
print("HOLDING is_reorder CONSTANT")
print("="*62)

for flag in [1, 0]:
    sub = df[df['is_reorder'] == flag]
    nov = sub[sub['demand_type'] == 'novelty'][
        'aisle_id'].value_counts(normalize=True)
    rep = sub[sub['demand_type']
              == 'replenishment'][
        'aisle_id'].value_counts(normalize=True)
    if len(nov) == 0 or len(rep) == 0:
        print(f"\n  is_reorder={flag}: "
              f"not enough rows")
        continue
    d = tvd(nov, rep)
    name = "REORDER rows" if flag else "NEW rows"
    print(f"\n  Within {name} only "
          f"({len(sub):,} rows):")
    print(f"    TVD(novelty vs replenishment) "
          f"= {d:.4f}")
    print(f"    novelty top5:      "
          f"{list(nov.head(5).index.astype(int))}")
    print(f"    replenishment top5:"
          f" {list(rep.head(5).index.astype(int))}")

# ── PREDICT demand_type FROM AISLE ALONE ──
print("\n" + "="*62)
print("CAN AISLE ALONE PREDICT demand_type?")
print("(is_reorder NOT given to the classifier)")
print("="*62)

s = df[df['demand_type'].isin(
    ['novelty', 'replenishment'])].sample(
    n=400000, random_state=SEED)

y = s['demand_type']
base = y.value_counts(normalize=True).max()

for feats in [['aisle_id'],
              ['aisle_id', 'is_reorder']]:
    X = s[feats]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3,
        random_state=SEED, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=60, max_depth=12,
        random_state=SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(
        yte, clf.predict(Xte))
    print(f"\n  Features {feats}")
    print(f"    Baseline: {base:.1%}")
    print(f"    Accuracy: {acc:.1%}")
    print(f"    Lift:     {acc-base:+.1%}")

print("\n" + "="*62)
print("VERDICT")
print("="*62)
print("""
If TVD stays >0.05 with is_reorder held
constant, demand_type carries genuine
order-level structure. Conditioning on it
is legitimate.

If TVD collapses to ~0, demand_type is just
is_reorder aggregated, and conditioning on it
would repeat the aisle_popularity mistake
under a new name. We would rethink.
""")