import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Fix all seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*60)
print("CREATING FIXED STRATIFIED EVALUATION SET")
print("Results will be identical every run")
print("="*60)

# Load full real data
print("\nLoading full dataset...")
df = pd.read_csv('data/clean_orders_v2.csv')
print(f"Full dataset: {len(df):,} rows")

# ── STRATIFIED SAMPLING ──
# Stratify on the most important columns
# This guarantees exact proportions
# are preserved in the evaluation set
print("\nCreating stratification key...")

# Create a combined stratification key
# using the most important features
df['strat_key'] = (
    df['is_weekend'].astype(str) + '_' +
    df['is_peak_hour'].astype(str) + '_' +
    df['is_night'].astype(str) + '_' +
    df['is_reorder'].astype(str) + '_' +
    df['time_of_day'].astype(str)
)

print("Unique strat combinations:",
      df['strat_key'].nunique())
print("\nStratification key distribution:")
print(df['strat_key'].value_counts(
    normalize=True).round(4).head(10))

# Remove extremely rare combinations
# that cannot be stratified
strat_counts = df['strat_key'].value_counts()
valid_strats = strat_counts[
    strat_counts >= 10].index
df_valid = df[
    df['strat_key'].isin(valid_strats)].copy()

print(f"\nRows after removing rare strats: "
      f"{len(df_valid):,}")

# Create stratified evaluation set
# 10,000 rows, perfectly representative
print("\nCreating stratified eval set...")
_, eval_set = train_test_split(
    df_valid,
    test_size=10000,
    stratify=df_valid['strat_key'],
    random_state=SEED
)

print(f"Eval set shape: {eval_set.shape}")

# Create stratified comparison set
# 50,000 rows for distribution comparison
print("Creating stratified comparison set...")
_, compare_set = train_test_split(
    df_valid,
    test_size=50000,
    stratify=df_valid['strat_key'],
    random_state=SEED
)

print(f"Compare set shape: {compare_set.shape}")

# Drop the strat key column
eval_set    = eval_set.drop(
    columns=['strat_key'])
compare_set = compare_set.drop(
    columns=['strat_key'])
df_valid    = df_valid.drop(
    columns=['strat_key'])

# ── VERIFY PROPORTIONS ──
print("\n" + "="*60)
print("VERIFYING STRATIFICATION QUALITY")
print("="*60)
print(f"\n{'Metric':<22} {'Full Data':>10} "
      f"{'Eval Set':>10} {'Compare':>10} "
      f"{'Match':>6}")
print("-"*62)

metrics = [
    ('Weekend ratio',    'is_weekend'),
    ('Peak hour ratio',  'is_peak_hour'),
    ('Night ratio',      'is_night'),
    ('Reorder ratio',    'is_reorder'),
    ('Early cart ratio', 'is_early_in_cart'),
    ('Avg hour',         'order_hour_of_day'),
    ('Avg day of week',  'order_dow'),
]

real_props = {}
for name, col in metrics:
    full_val    = df[col].astype(float).mean()
    eval_val    = eval_set[col].astype(
        float).mean()
    compare_val = compare_set[col].astype(
        float).mean()
    real_props[col] = full_val

    eval_diff = abs(full_val - eval_val)
    match = "✅" if eval_diff < 0.005 else \
            "🟡" if eval_diff < 0.01 else "❌"

    print(f"{name:<22} {full_val:>10.4f} "
          f"{eval_val:>10.4f} "
          f"{compare_val:>10.4f} {match:>6}")

# Time of day
print("\nTime of day verification:")
print(f"{'Category':<12} {'Full':>8} "
      f"{'Eval':>8} {'Compare':>8}")
for cat in ['morning', 'afternoon',
            'evening', 'late', 'night']:
    fv = (df['time_of_day'] == cat).mean()
    ev = (eval_set[
        'time_of_day'] == cat).mean()
    cv = (compare_set[
        'time_of_day'] == cat).mean()
    print(f"{cat:<12} {fv:>8.4f} "
          f"{ev:>8.4f} {cv:>8.4f}")

# Order frequency
print("\nOrder frequency verification:")
print(f"{'Category':<12} {'Full':>8} "
      f"{'Eval':>8} {'Compare':>8}")
for cat in ['weekly', 'biweekly',
            'monthly', 'first']:
    fv = (df['order_frequency'] == cat).mean()
    ev = (eval_set[
        'order_frequency'] == cat).mean()
    cv = (compare_set[
        'order_frequency'] == cat).mean()
    print(f"{cat:<12} {fv:>8.4f} "
          f"{ev:>8.4f} {cv:>8.4f}")

# ── SAVE EVERYTHING ──
print("\nSaving fixed evaluation sets...")

eval_set.to_csv(
    'data/fixed_real_eval.csv',
    index=False)
print("Saved fixed_real_eval.csv "
      "(10,000 rows, stratified)")

compare_set.to_csv(
    'data/fixed_real_compare.csv',
    index=False)
print("Saved fixed_real_compare.csv "
      "(50,000 rows, stratified)")

# Save ground truth proportions
pd.DataFrame([real_props]).to_csv(
    'data/real_proportions.csv',
    index=False)
print("Saved real_proportions.csv")

# ── FINAL SUMMARY ──
print("\n" + "="*60)
print("STRATIFICATION COMPLETE")
print("="*60)
print("\nThese files are now permanent:")
print("  data/fixed_real_eval.csv")
print("    → Use for SDMetrics quality scoring")
print("  data/fixed_real_compare.csv")
print("    → Use for distribution comparison")
print("  data/real_proportions.csv")
print("    → Ground truth proportions")
print("\nResults will be IDENTICAL")
print("every time you run evaluations.")
print("\nNever delete these files!")
print("="*60)
print("\nDone!")