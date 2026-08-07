import pandas as pd
import numpy as np
import random
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*60)
print("HONEST CONDITIONING TEST")
print("Condition ONLY on flags unrelated to")
print("aisle and hour. Do aisle/hour still")
print("separate on their own?")
print("="*60)

model = CTGANSynthesizer.load(
    'data/ctgan_final.pkl')
print("\nModel loaded (ctgan_final.pkl)")

N = 10000

# ONLY four flags. No aisle_popularity,
# no time_of_day, no order_frequency.
scenarios = {
    'Normal': {
        'is_weekend':   0,
        'is_reorder':   1,
        'is_peak_hour': 1,
        'is_night':     0,
    },
    'Christmas': {
        'is_weekend':   1,
        'is_reorder':   1,
        'is_peak_hour': 1,
        'is_night':     0,
    },
    'Black Friday': {
        'is_weekend':   0,
        'is_reorder':   0,
        'is_peak_hour': 1,
        'is_night':     0,
    },
}

print("\nConditioned on: is_weekend,")
print("  is_reorder, is_peak_hour, is_night")
print("FREE: aisle_id, aisle_popularity,")
print("  time_of_day, order_hour_of_day,")
print("  order_frequency, department_id")

data = {}
for name, cond in scenarios.items():
    print(f"\nGenerating {name}...")
    c = Condition(num_rows=N,
                  column_values=cond)
    df = model.sample_from_conditions(
        conditions=[c])
    df['scenario'] = name
    data[name] = df

# ── DID AISLE / HOUR SEPARATE? ──
print("\n" + "="*60)
print("FREE FEATURES — DID THEY SEPARATE?")
print("="*60)
print(f"\n{'Metric':<24}{'Normal':>9}"
      f"{'Xmas':>9}{'BlackFri':>10}")
print("-"*52)

for col, label in [
    ('aisle_id',          'Avg Aisle'),
    ('order_hour_of_day', 'Avg Hour'),
    ('order_dow',         'Avg Day'),
]:
    v = [pd.to_numeric(data[s][col],
         errors='coerce').mean()
         for s in scenarios]
    print(f"{label:<24}{v[0]:>9.2f}"
          f"{v[1]:>9.2f}{v[2]:>10.2f}")

print(f"\n{'Top aisle':<24}", end="")
for s in scenarios:
    t = int(data[s]['aisle_id'].mode()[0])
    print(f"{t:>9}", end="")
print()
print(f"{'Top aisle share':<24}", end="")
for s in scenarios:
    t = data[s]['aisle_id'].mode()[0]
    p = (data[s]['aisle_id'] == t).mean()*100
    print(f"{p:>8.1f}%", end="")
print()

print(f"\n{'aisle_popularity (high%)':<24}", end="")
for s in scenarios:
    if 'aisle_popularity' in data[s].columns:
        p = (data[s]['aisle_popularity']
             == 'high').mean()*100
        print(f"{p:>8.1f}%", end="")
print()

# ── CLASSIFIER ON FREE FEATURES ONLY ──
print("\n" + "="*60)
print("CLASSIFIER — free features only")
print("="*60)

allf = pd.concat(data.values(),
                 ignore_index=True)
free_cols = ['aisle_id', 'department_id',
             'order_hour_of_day']
X = allf[free_cols].apply(
    pd.to_numeric, errors='coerce').fillna(0)
y = allf['scenario']

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.3,
    random_state=SEED, stratify=y)
clf = RandomForestClassifier(
    n_estimators=100, random_state=SEED,
    n_jobs=-1)
clf.fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))

print(f"\nFeatures used: {free_cols}")
print(f"Accuracy:  {acc:.1%}")
print(f"Random:    33.3%")
print(f"\nPrevious claim (v2 test): 98.7%")

print("\n" + "="*60)
print("VERDICT")
print("="*60)
if acc > 0.75:
    print(f"""
{acc:.1%} on aisle/hour alone, with NO
aisle or time conditioning applied.
The model genuinely LEARNED that these
demand contexts map to different aisles.
The structural-learning claim HOLDS.
""")
elif acc > 0.50:
    print(f"""
{acc:.1%} — partial separation. Some real
structure was learned, but weaker than
the 98.7% figure suggested. State this
honestly: scenarios differ, but much of
the earlier separation came from the
aisle_popularity / time_of_day conditioning.
""")
else:
    print(f"""
{acc:.1%} — close to random. Aisle and hour
do NOT separate on their own. The earlier
98.7% came from conditioning on
aisle_popularity and time_of_day, which
determine aisle and hour directly.

REFRAME HONESTLY: the scenarios are
distinct because they were explicitly
conditioned across seven attributes.
Separability confirms the conditioning
was applied faithfully — not that
structure emerged unprompted.
""")
print("="*60)