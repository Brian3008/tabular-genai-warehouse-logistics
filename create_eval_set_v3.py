"""
create_eval_set_v3.py
=====================
Builds a PROVABLY DISJOINT train/eval split.

WHAT WAS WRONG IN v2
--------------------
create_eval_set.py sampled 10k eval rows from all
32.4M rows of clean_orders_v2.csv.

train_best_model.py sampled 250k+ training rows
from the SAME 32.4M rows.

Nothing held them apart. So the model was scored
on rows it may have trained on. Worse,
train_best_model.py evaluated each model against a
slice of its OWN TRAINING SAMPLE:

    real_eval = sample[eval_cols].sample(n=5000)

Every score that selected "CTGAN Run 3" as the
winner was a training-set score.

HOW v3 FIXES IT
---------------
1. Split by ORDER_ID, not by row.
   An order is the natural unit - a warehouse
   fulfils whole orders. Splitting by row would
   put some items of an order in train and others
   in eval, which leaks.

2. Save the training order IDs to disk.
   Every downstream script loads them and ASSERTS
   it is not touching them. Disjointness is
   verified, not assumed.

3. NO OVERSAMPLING.
   v2 deliberately added 15k extra night rows,
   30k peak, 40k reorder. The model then learned
   the inflated night rate (0.111 vs real 0.050) -
   correctly, because that is what it was trained
   on. Then post-processing "fixed" it by injecting
   real hours. v3 trains on the true distribution
   so there is nothing to fix.

OUTPUTS
-------
  data/v3_train_order_ids.csv   the training orders
  data/v3_train.csv             training rows
  data/v3_eval.csv              10k held-out rows (quality)
  data/v3_compare.csv           50k held-out rows (ML/corr)
  data/v3_real_proportions.csv  ground truth

Nothing from v2 is overwritten.
"""

import sys
# Some prints use non-ASCII (e.g. the intersection symbol).
# A default Windows console is cp1252 and CRASHES on them;
# force UTF-8 so the script runs outside PyCharm too.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# Training sample size. 320k rows was verified to
# work (ML efficacy 100.4%, correlation 97.8%).
TRAIN_ROWS = 320000

print("=" * 64)
print("CREATE EVAL SET v3 - PROVABLY DISJOINT SPLIT")
print("=" * 64)

# ══════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════
print("\n[1] Loading clean_orders_v3.csv ...")
df = pd.read_csv('data/clean_orders_v3.csv')
print(f"    rows:   {len(df):,}")
print(f"    orders: {df['order_id'].nunique():,}")

# ══════════════════════════════════════════
# ADD ORDER SIZE GROUP (the conditioning axis)
# ══════════════════════════════════════════
print("\n[2] Computing order_size_grp ...")
print("    (the verified conditioning axis -")
print("     tail TVD 0.1536 vs 0.0357 noise floor)")

# WARNING - STALE COLUMN TRAP: this computes order_size_grp
# with 33/67 TERTILES, and that version is what gets saved
# into v3_train.csv / v3_eval.csv / v3_compare.csv. The
# buckets the project actually uses are small<=10 / large>=14
# (chosen later by the threshold screen). Every live script
# RECOMPUTES the column with 10/14 - never trust the stored
# column from these CSVs.
osize = df.groupby('order_id').size()
q33, q67 = osize.quantile([0.33, 0.67])

def size_grp(n):
    if n <= q33:
        return 'small'
    if n >= q67:
        return 'large'
    return 'mid'

df['order_size_grp'] = df['order_id'].map(
    osize.apply(size_grp))

print(f"    small: <= {q33:.0f} items")
print(f"    large: >= {q67:.0f} items")
print("\n    row distribution:")
for k, v in df['order_size_grp'].value_counts(
        normalize=True).items():
    print(f"      {k:<8} {v:6.1%}")

# ══════════════════════════════════════════
# SPLIT BY ORDER_ID
# ══════════════════════════════════════════
print("\n[3] Splitting by ORDER_ID (not by row) ...")

all_orders = df['order_id'].unique()
rng = np.random.RandomState(SEED)
rng.shuffle(all_orders)

# Reserve enough orders for train + both eval sets.
# Average items per order tells us how many orders
# we need to hit TRAIN_ROWS.
avg_items = len(df) / len(all_orders)
n_train_orders = int(TRAIN_ROWS / avg_items)

# Eval sets need to be comfortably large
n_eval_orders    = int(20000 / avg_items) + 500
n_compare_orders = int(80000 / avg_items) + 500

need = (n_train_orders + n_eval_orders
        + n_compare_orders)
if need > len(all_orders):
    raise SystemExit(
        f"Not enough orders: need {need:,}, "
        f"have {len(all_orders):,}")

train_orders   = all_orders[:n_train_orders]
eval_orders    = all_orders[
    n_train_orders:n_train_orders + n_eval_orders]
compare_orders = all_orders[
    n_train_orders + n_eval_orders:
    n_train_orders + n_eval_orders
    + n_compare_orders]

print(f"    avg items/order:  {avg_items:.1f}")
print(f"    train orders:     "
      f"{len(train_orders):,}")
print(f"    eval orders:      "
      f"{len(eval_orders):,}")
print(f"    compare orders:   "
      f"{len(compare_orders):,}")

# ══════════════════════════════════════════
# PROVE DISJOINTNESS
# ══════════════════════════════════════════
print("\n[4] PROVING disjointness ...")

s_tr = set(train_orders)
s_ev = set(eval_orders)
s_cp = set(compare_orders)

assert len(s_tr & s_ev) == 0, \
    "OVERLAP: train and eval share orders!"
assert len(s_tr & s_cp) == 0, \
    "OVERLAP: train and compare share orders!"
assert len(s_ev & s_cp) == 0, \
    "OVERLAP: eval and compare share orders!"

print(f"    train ∩ eval:    "
      f"{len(s_tr & s_ev)} orders  [OK]")
print(f"    train ∩ compare: "
      f"{len(s_tr & s_cp)} orders  [OK]")
print(f"    eval  ∩ compare: "
      f"{len(s_ev & s_cp)} orders  [OK]")
print("\n    ASSERTED: no order appears in more")
print("    than one split. Verified, not assumed.")

# ══════════════════════════════════════════
# BUILD THE SETS
# ══════════════════════════════════════════
print("\n[5] Building row sets ...")

train   = df[df['order_id'].isin(s_tr)]
eval_df = df[df['order_id'].isin(s_ev)]
comp_df = df[df['order_id'].isin(s_cp)]

# Trim to target sizes (whole orders only -
# never split an order across the boundary)
def trim_to(d, target):
    if len(d) <= target:
        return d
    # cut at an order boundary
    ids = d['order_id'].unique()
    cum = d.groupby('order_id').size().loc[
        ids].cumsum()
    keep = cum[cum <= target].index
    if len(keep) == 0:
        keep = ids[:1]
    return d[d['order_id'].isin(keep)]

train   = trim_to(train,   TRAIN_ROWS)
eval_df = trim_to(eval_df, 10000)
comp_df = trim_to(comp_df, 50000)

print(f"    train:   {len(train):,} rows, "
      f"{train['order_id'].nunique():,} orders")
print(f"    eval:    {len(eval_df):,} rows, "
      f"{eval_df['order_id'].nunique():,} orders")
print(f"    compare: {len(comp_df):,} rows, "
      f"{comp_df['order_id'].nunique():,} orders")

# ══════════════════════════════════════════
# NO OVERSAMPLING - VERIFY TRUE DISTRIBUTION
# ══════════════════════════════════════════
print("\n[6] Verifying NO oversampling ...")
print("    (v2 inflated night to ~11% by adding")
print("     15k extra night rows. v3 does not.)")

print(f"\n    {'Metric':<22}{'Full':>9}"
      f"{'Train':>9}{'Eval':>9}{'Comp':>9}")
print("    " + "-" * 58)

METRICS = [
    ('is_weekend',        'Weekend ratio'),
    ('is_peak_hour',      'Peak hour ratio'),
    ('is_night',          'Night ratio'),
    ('is_reorder',        'Reorder ratio'),
    ('is_early_in_cart',  'Early cart ratio'),
    ('order_hour_of_day', 'Avg hour'),
    ('order_dow',         'Avg day'),
]

real_props = {}
ok = True
for col, label in METRICS:
    f = df[col].astype(float).mean()
    t = train[col].astype(float).mean()
    e = eval_df[col].astype(float).mean()
    c = comp_df[col].astype(float).mean()
    real_props[col] = f
    # train must NOT deviate from full data
    if abs(f - t) > 0.02 * max(abs(f), 1):
        ok = False
    print(f"    {label:<22}{f:>9.3f}"
          f"{t:>9.3f}{e:>9.3f}{c:>9.3f}")

print()
if ok:
    print("    TRAIN MATCHES FULL DATA.")
    print("    No oversampling. Night ratio will")
    print("    come out near the true 0.050, so")
    print("    there is NOTHING to calibrate.")
else:
    print("    WARNING: train deviates from full")
    print("    data. Investigate before training.")

# order size balance
print(f"\n    {'order_size_grp':<22}"
      f"{'Full':>9}{'Train':>9}")
print("    " + "-" * 40)
for g in ['small', 'mid', 'large']:
    f = (df['order_size_grp'] == g).mean()
    t = (train['order_size_grp'] == g).mean()
    print(f"    {g:<22}{f:>9.3f}{t:>9.3f}")

# ══════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════
print("\n[7] Saving ...")

pd.DataFrame({
    'order_id': sorted(
        train['order_id'].unique())
}).to_csv('data/v3_train_order_ids.csv',
          index=False)

train.to_csv('data/v3_train.csv', index=False)
eval_df.to_csv('data/v3_eval.csv', index=False)
comp_df.to_csv('data/v3_compare.csv', index=False)
pd.DataFrame([real_props]).to_csv(
    'data/v3_real_proportions.csv', index=False)

print("    data/v3_train_order_ids.csv")
print("    data/v3_train.csv")
print("    data/v3_eval.csv")
print("    data/v3_compare.csv")
print("    data/v3_real_proportions.csv")

# ══════════════════════════════════════════
# FINAL VERIFICATION FROM DISK
# ══════════════════════════════════════════
print("\n[8] Re-verifying from saved files ...")

t_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
e_ids = set(pd.read_csv(
    'data/v3_eval.csv')['order_id'])
c_ids = set(pd.read_csv(
    'data/v3_compare.csv')['order_id'])

assert len(t_ids & e_ids) == 0
assert len(t_ids & c_ids) == 0
assert len(e_ids & c_ids) == 0

print(f"    train orders on disk: {len(t_ids):,}")
print(f"    overlap with eval:    "
      f"{len(t_ids & e_ids)}")
print(f"    overlap with compare: "
      f"{len(t_ids & c_ids)}")
print("\n    VERIFIED FROM DISK. Disjoint.")

print("\n" + "=" * 64)
print("DONE")
print("=" * 64)
print("""
Every downstream script must now:
  - train ONLY on v3_train.csv
  - score ONLY on v3_eval.csv / v3_compare.csv
  - load v3_train_order_ids.csv and assert it
    is not touching training orders

That makes the split verifiable at every step,
not just here.
""")