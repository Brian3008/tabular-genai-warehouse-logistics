import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n" + "="*60)
print("CTGAN FINAL - FIXING REMAINING GAPS")
print("Target: perfect night, peak hour,")
print("and reorder ratios")
print("="*60)

# ── LOAD DATA ──
print("\nLoading dataset...")
df = pd.read_csv('data/clean_orders_v2.csv')

print("\nReal data distribution:")
print(f"  Night orders:   "
      f"{df['is_night'].mean()*100:.2f}%")
print(f"  Peak hours:     "
      f"{df['is_peak_hour'].mean()*100:.2f}%")
print(f"  Reorders:       "
      f"{df['is_reorder'].mean()*100:.2f}%")
print(f"  Weekend:        "
      f"{df['is_weekend'].mean()*100:.2f}%")

# ── SMART SAMPLING ──
# Key insight: preserve exact real proportions
# but use enough data for CTGAN to learn well
print("\nBuilding precisely balanced sample...")

# Base sample - large enough for good learning
base = df.sample(n=280000, random_state=42)

# Night orders - preserve real proportion (5%)
# 280000 * 0.05 = 14000 night orders needed
# Check how many we have in base
base_night = base[base['is_night'] == 1]
print(f"Night orders in base: "
      f"{len(base_night)} "
      f"({len(base_night)/len(base)*100:.2f}%)")

# If night orders are underrepresented
# add just enough to reach real proportion
target_night_pct = 0.05  # exact real proportion
target_night_count = int(
    len(base) * target_night_pct)
current_night = len(base_night)

if current_night < target_night_count:
    extra_night = df[
        df['is_night'] == 1].sample(
        n=target_night_count - current_night,
        random_state=42,
        replace=True)
    sample = pd.concat(
        [base, extra_night],
        ignore_index=True)
else:
    sample = base.copy()

# Boost reorders slightly
# Real proportion is 59%
# Just add a small boost to help CTGAN learn
reorder_boost = df[
    df['is_reorder'] == 1].sample(
    n=20000, random_state=42)

sample = pd.concat(
    [sample, reorder_boost],
    ignore_index=True
).drop_duplicates(subset=[
    'order_id', 'aisle_id',
    'order_dow', 'order_hour_of_day'
])

print(f"\nFinal sample size: {len(sample):,}")
print(f"\nSample distribution:")
print(f"  Night orders:   "
      f"{sample['is_night'].mean()*100:.2f}%"
      f" (target: 5.00%)")
print(f"  Peak hours:     "
      f"{sample['is_peak_hour'].mean()*100:.2f}%"
      f" (target: 57.30%)")
print(f"  Reorders:       "
      f"{sample['is_reorder'].mean()*100:.2f}%"
      f" (target: 59.00%)")
print(f"  Weekend:        "
      f"{sample['is_weekend'].mean()*100:.2f}%"
      f" (target: 36.60%)")

# ── METADATA ──
print("\nDefining metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(sample)

cat_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'time_of_day',
    'is_reorder', 'order_frequency',
    'is_early_in_cart', 'aisle_popularity'
]

for col in cat_cols:
    metadata.update_column(
        column_name=col,
        sdtype='categorical')

metadata.update_column(
    column_name='days_since_prior_order',
    sdtype='numerical')
metadata.update_column(
    column_name='order_id',
    sdtype='id')

print("Metadata defined!")

# ── EVALUATION SETUP ──
eval_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_reorder',
    'is_early_in_cart'
]

eval_metadata = SingleTableMetadata()
eval_metadata.detect_from_dataframe(
    sample[eval_cols])
for col in eval_cols:
    eval_metadata.update_column(
        column_name=col,
        sdtype='categorical')

real_eval = sample[eval_cols].sample(
    n=5000, random_state=42)

# ── TRAIN CTGAN FINAL ──
# Using Run 3 config which was the winner
# Small batch, very slow learning rate,
# more discriminator steps
print("\n" + "="*60)
print("TRAINING CTGAN FINAL")
print("Run 3 winning config:")
print("Small batch=300, lr=0.00005,")
print("disc_steps=3, 400 epochs")
print("Estimated time: 4-5 hours")
print("="*60)

model = CTGANSynthesizer(
    metadata,
    epochs=400,
    batch_size=300,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=0.00005,
    discriminator_lr=0.00005,
    discriminator_steps=3,
    verbose=True
)

model.fit(sample)
print("\nTraining complete!")

# ── EVALUATE ──
print("\nEvaluating...")
synthetic_eval = model.sample(num_rows=5000)

quality = evaluate_quality(
    real_data=real_eval,
    synthetic_data=synthetic_eval[eval_cols],
    metadata=eval_metadata,
    verbose=True
)

score = quality.get_score()
print(f"\nQuality score: {score:.4f}")

# Compare with previous best
prev_best = 0.9138
if score > prev_best:
    print(f"NEW BEST! Improved from "
          f"{prev_best:.4f} to {score:.4f}")
    model.save('data/best_model.pkl')
    print("Saved as new best model!")
else:
    print(f"Score {score:.4f} vs previous "
          f"best {prev_best:.4f}")
    print("Keeping previous best model.")
    response = input(
        "Save this model anyway? (y/n): ")
    if response.lower() == 'y':
        model.save('data/ctgan_final.pkl')
        print("Saved as ctgan_final.pkl")

# ── GENERATE AND COMPARE ──
print("\nGenerating 50,000 synthetic orders...")
synthetic = model.sample(num_rows=50000)

print("\n" + "="*60)
print("DETAILED COMPARISON")
print("="*60)
print(f"\n{'Metric':<22} {'Real':>8} "
      f"{'Synthetic':>10} {'Diff':>8} "
      f"{'Match':>6}")
print("-"*58)

# Real proportions from original dataset
real_props = {
    'is_weekend':        df['is_weekend'].mean(),
    'is_peak_hour':      df['is_peak_hour'].mean(),
    'is_night':          df['is_night'].mean(),
    'is_reorder':        df['is_reorder'].mean(),
    'is_early_in_cart':  df['is_early_in_cart'].mean(),
    'order_hour_of_day': df['order_hour_of_day'].mean(),
    'order_dow':         df['order_dow'].mean(),
}

metrics = [
    ('Weekend ratio',
     'is_weekend'),
    ('Peak hour ratio',
     'is_peak_hour'),
    ('Night ratio',
     'is_night'),
    ('Reorder ratio',
     'is_reorder'),
    ('Early cart ratio',
     'is_early_in_cart'),
    ('Avg hour',
     'order_hour_of_day'),
    ('Avg day of week',
     'order_dow'),
]

perfect  = 0
good     = 0
bad      = 0

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

print(f"\nScore: {perfect}/7 perfect ✅  "
      f"{good}/7 good 🟡  "
      f"{bad}/7 needs work ❌")

# Time of day
print("\nTime of day:")
print(f"{'Category':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Match':>6}")
tod_perfect = 0
for cat in ['morning', 'afternoon',
            'evening', 'late', 'night']:
    rv   = (df['time_of_day'] == cat).mean()
    sv   = (synthetic['time_of_day'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    if match == "✅":
        tod_perfect += 1
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# Order frequency
print("\nOrder frequency:")
print(f"{'Category':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Match':>6}")
freq_perfect = 0
for cat in ['weekly', 'biweekly',
            'monthly', 'first']:
    rv   = (df['order_frequency'] == cat).mean()
    sv   = (synthetic['order_frequency'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    if match == "✅":
        freq_perfect += 1
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# ── VISUALISATION ──
print("\nGenerating charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'CTGAN Final Model\n'
    f'Quality Score: {score:.4f} | '
    f'Previous Best: {prev_best:.4f}',
    fontsize=13, fontweight='bold')

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
axes[0][0].set_title('Binary Feature Ratios\n'
                     '(closer = better)')
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
axes[0][1].plot(
    real_h.index, real_h.values,
    label='Real', color='steelblue',
    linewidth=2)
axes[0][1].plot(
    synt_h.index, synt_h.values,
    label='Synthetic', color='green',
    linewidth=2, linestyle='--')
axes[0][1].set_title('Hour Distribution\n'
                     '(lines should overlap)')
axes[0][1].set_xlabel('Hour of Day')
axes[0][1].set_ylabel('Proportion')
axes[0][1].legend()

# Day distribution
real_d = df['order_dow']\
    .value_counts(normalize=True).sort_index()
synt_d = synthetic['order_dow']\
    .value_counts(normalize=True).sort_index()
axes[1][0].bar(
    real_d.index - 0.2, real_d.values,
    0.4, label='Real',
    color='steelblue', alpha=0.8)
axes[1][0].bar(
    synt_d.index + 0.2, synt_d.values,
    0.4, label='Synthetic',
    color='green', alpha=0.8)
axes[1][0].set_title('Day of Week Distribution')
axes[1][0].set_xlabel('Day (0=Sunday)')
axes[1][0].set_ylabel('Proportion')
axes[1][0].legend()

# Score comparison
scores_compare = {
    'Previous\nBest': prev_best,
    'This\nModel':    score
}
colours = ['steelblue',
           'green' if score >= prev_best
           else 'orange']
bars = axes[1][1].bar(
    scores_compare.keys(),
    scores_compare.values(),
    color=colours)
axes[1][1].set_title('Quality Score Comparison')
axes[1][1].set_ylabel('Quality Score')
axes[1][1].set_ylim(0.85, 1.0)
for bar, v in zip(
        bars, scores_compare.values()):
    axes[1][1].text(
        bar.get_x() + bar.get_width()/2,
        v + 0.001,
        f'{v:.4f}',
        ha='center', fontsize=13,
        fontweight='bold')

plt.tight_layout()
plt.savefig(
    'data/ctgan_final_comparison.png',
    dpi=150, bbox_inches='tight')
plt.show()

print("\nChart saved to "
      "data/ctgan_final_comparison.png")
print(f"\nFinal quality score: {score:.4f}")
print(f"Previous best:       {prev_best:.4f}")

if score >= prev_best:
    print("\nSUCCESS: New best model achieved!")
else:
    print("\nPrevious model remains the best.")
    print("Your 91.38% score stands as best.")
print("\nDone!")