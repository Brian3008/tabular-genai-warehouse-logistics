import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("VIEWING BEST MODEL RESULTS")
print("="*60)

# Load real data
print("\nLoading real data...")
df = pd.read_csv('data/clean_orders_v2.csv')

# Load best synthetic data
# No need to reload model - just use saved csv
print("Loading best synthetic data...")
synthetic = pd.read_csv(
    'data/synthetic_orders_best.csv')

print(f"Real data rows:      {len(df):,}")
print(f"Synthetic data rows: {len(synthetic):,}")

# Real proportions
real_props = {
    'is_weekend':        df['is_weekend'].mean(),
    'is_peak_hour':      df['is_peak_hour'].mean(),
    'is_night':          df['is_night'].mean(),
    'is_reorder':        df['is_reorder'].mean(),
    'is_early_in_cart':  df['is_early_in_cart'].mean(),
    'order_hour_of_day': df['order_hour_of_day'].mean(),
    'order_dow':         df['order_dow'].mean(),
}

# ── FULL COMPARISON ──
print("\n" + "="*60)
print("BEST MODEL (91.38%) - FULL RESULTS")
print("="*60)
print(f"\n{'Metric':<22} {'Real':>8} "
      f"{'Synthetic':>10} {'Diff':>8} "
      f"{'Match':>6}")
print("-"*58)

metrics = [
    ('Weekend ratio',    'is_weekend'),
    ('Peak hour ratio',  'is_peak_hour'),
    ('Night ratio',      'is_night'),
    ('Reorder ratio',    'is_reorder'),
    ('Early cart ratio', 'is_early_in_cart'),
    ('Avg hour',         'order_hour_of_day'),
    ('Avg day of week',  'order_dow'),
]

perfect = 0
good    = 0
bad     = 0

for name, col in metrics:
    rv   = real_props[col]
    sv   = synthetic[col].astype(float).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    if match == "✅":
        perfect += 1
    elif match == "🟡":
        good += 1
    else:
        bad += 1
    print(f"{name:<22} {rv:>8.3f} "
          f"{sv:>10.3f} {diff:>8.3f} {match:>6}")

print(f"\nResult: {perfect}/7 perfect ✅  "
      f"{good}/7 good 🟡  "
      f"{bad}/7 needs work ❌")

# Time of day
print("\nTime of day distribution:")
print(f"{'Category':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Match':>6}")
for cat in ['morning', 'afternoon',
            'evening', 'late', 'night']:
    rv   = (df['time_of_day'] == cat).mean()
    sv   = (synthetic['time_of_day'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# Order frequency
print("\nOrder frequency distribution:")
print(f"{'Category':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Match':>6}")
for cat in ['weekly', 'biweekly',
            'monthly', 'first']:
    rv   = (df['order_frequency'] == cat).mean()
    sv   = (synthetic['order_frequency'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# Overall quality score
print("\nCalculating quality score...")
eval_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_reorder',
    'is_early_in_cart'
]

eval_metadata = SingleTableMetadata()
eval_metadata.detect_from_dataframe(
    df[eval_cols].sample(
        n=5000, random_state=42))
for col in eval_cols:
    eval_metadata.update_column(
        column_name=col,
        sdtype='categorical')

quality = evaluate_quality(
    real_data=df[eval_cols].sample(
        n=5000, random_state=42),
    synthetic_data=synthetic[
        eval_cols].sample(
        n=5000, random_state=42),
    metadata=eval_metadata,
    verbose=True
)

score = quality.get_score()
print(f"\nOverall quality score: {score:.4f}")

# ── CHARTS ──
print("\nGenerating results charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Best CTGAN Model Results\n'
    f'Quality Score: {score:.4f} | '
    f'Perfect metrics: {perfect}/7',
    fontsize=14, fontweight='bold')

# Binary features
binary_cols  = ['is_weekend', 'is_peak_hour',
                'is_night', 'is_reorder']
binary_names = ['Weekend', 'Peak Hour',
                'Night', 'Reorder']
real_vals = [real_props[c]
             for c in binary_cols]
synt_vals = [synthetic[c].mean()
             for c in binary_cols]
x     = np.arange(len(binary_names))
width = 0.35
axes[0][0].bar(x - width/2, real_vals,
               width, label='Real',
               color='steelblue', alpha=0.8)
axes[0][0].bar(x + width/2, synt_vals,
               width, label='Synthetic',
               color='green', alpha=0.8)
axes[0][0].set_title('Binary Feature Ratios')
axes[0][0].set_xticks(x)
axes[0][0].set_xticklabels(binary_names)
axes[0][0].set_ylabel('Ratio')
axes[0][0].legend()
axes[0][0].set_ylim(0, 1)

# Hour distribution
real_h = df['order_hour_of_day']\
    .value_counts(normalize=True).sort_index()
synt_h = synthetic['order_hour_of_day']\
    .value_counts(normalize=True).sort_index()
axes[0][1].plot(real_h.index, real_h.values,
                label='Real',
                color='steelblue', linewidth=2)
axes[0][1].plot(synt_h.index, synt_h.values,
                label='Synthetic',
                color='green', linewidth=2,
                linestyle='--')
axes[0][1].set_title('Hour of Day Distribution')
axes[0][1].set_xlabel('Hour')
axes[0][1].set_ylabel('Proportion')
axes[0][1].legend()

# Day distribution
real_d = df['order_dow']\
    .value_counts(normalize=True).sort_index()
synt_d = synthetic['order_dow']\
    .value_counts(normalize=True).sort_index()
axes[1][0].bar(real_d.index - 0.2,
               real_d.values, 0.4,
               label='Real',
               color='steelblue', alpha=0.8)
axes[1][0].bar(synt_d.index + 0.2,
               synt_d.values, 0.4,
               label='Synthetic',
               color='green', alpha=0.8)
axes[1][0].set_title('Day of Week Distribution')
axes[1][0].set_xlabel('Day (0=Sunday)')
axes[1][0].set_ylabel('Proportion')
axes[1][0].legend()

# Aisle distribution
real_a = df['aisle_id']\
    .value_counts(normalize=True).sort_index()
synt_a = synthetic['aisle_id']\
    .value_counts(normalize=True).sort_index()
axes[1][1].plot(real_a.index, real_a.values,
                label='Real',
                color='steelblue',
                linewidth=1, alpha=0.8)
axes[1][1].plot(synt_a.index, synt_a.values,
                label='Synthetic',
                color='green',
                linewidth=1,
                linestyle='--', alpha=0.8)
axes[1][1].set_title('Aisle Distribution')
axes[1][1].set_xlabel('Aisle ID')
axes[1][1].set_ylabel('Proportion')
axes[1][1].legend()

plt.tight_layout()
plt.savefig('data/view_results.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nChart saved to data/view_results.png")
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Overall quality score: {score:.4f}")
print(f"Perfect metrics:       {perfect}/7 ✅")
print(f"Good metrics:          {good}/7 🟡")
print(f"Needs work:            {bad}/7 ❌")
print("="*60)
print("\nDone!")