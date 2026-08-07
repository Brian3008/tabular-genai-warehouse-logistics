import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ── LOAD AND PREPARE DATA ──
print("\n" + "="*60)
print("BEST MODEL TRAINING PIPELINE")
print("CTGAN x3 + TVAE x1 → Auto picks winner")
print("="*60)

print("\nLoading enriched dataset...")
df = pd.read_csv('data/clean_orders_v2.csv')

print("\nReal data class distribution:")
print(f"  Night orders:   "
      f"{df['is_night'].mean()*100:.1f}%")
print(f"  Peak hours:     "
      f"{df['is_peak_hour'].mean()*100:.1f}%")
print(f"  Reorders:       "
      f"{df['is_reorder'].mean()*100:.1f}%")
print(f"  Weekend:        "
      f"{df['is_weekend'].mean()*100:.1f}%")

# ── SMART BALANCED SAMPLING ──
print("\nBuilding balanced training sample...")

base = df.sample(n=250000, random_state=42)

night = df[df['is_night'] == 1].sample(
    n=15000, random_state=42,
    replace=True)

peak = df[df['is_peak_hour'] == 1].sample(
    n=30000, random_state=42)

reorders = df[df['is_reorder'] == 1].sample(
    n=40000, random_state=42)

sample = pd.concat(
    [base, night, peak, reorders],
    ignore_index=True
).drop_duplicates(subset=[
    'order_id', 'aisle_id',
    'order_dow', 'order_hour_of_day'
])

print(f"Training sample size: {len(sample):,}")
print(f"\nBalanced class distribution:")
print(f"  Night orders:   "
      f"{sample['is_night'].mean()*100:.1f}%")
print(f"  Peak hours:     "
      f"{sample['is_peak_hour'].mean()*100:.1f}%")
print(f"  Reorders:       "
      f"{sample['is_reorder'].mean()*100:.1f}%")
print(f"  Weekend:        "
      f"{sample['is_weekend'].mean()*100:.1f}%")

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

# ── TRACKING ──
best_score  = 0
best_model  = None
best_name   = ""
all_results = []

def evaluate_model(model, name):
    global best_score, best_model, best_name
    print(f"\nEvaluating {name}...")
    synthetic_eval = model.sample(num_rows=5000)
    quality = evaluate_quality(
        real_data=real_eval,
        synthetic_data=synthetic_eval[eval_cols],
        metadata=eval_metadata,
        verbose=False
    )
    score = quality.get_score()
    all_results.append({
        'name':  name,
        'score': score
    })
    print(f"{name} quality score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model
        best_name  = name
        print(f"*** NEW BEST: {name} "
              f"with score {score:.4f} ***")
        model.save('data/best_model.pkl')
    return score

# ── CTGAN RUN 1 ──
# Standard settings from original paper
print("\n" + "="*60)
print("CTGAN RUN 1 of 3")
print("Standard settings, fast convergence")
print("="*60)

ctgan1 = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256),
    generator_lr=0.0002,
    discriminator_lr=0.0002,
    discriminator_steps=1,
    verbose=True
)
ctgan1.fit(sample)
evaluate_model(ctgan1, "CTGAN Run 1")

# ── CTGAN RUN 2 ──
# Larger tapering network, slower learning
print("\n" + "="*60)
print("CTGAN RUN 2 of 3")
print("Large tapering network, careful learning")
print("="*60)

ctgan2 = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    generator_dim=(512, 256, 128),
    discriminator_dim=(512, 256, 128),
    generator_lr=0.0001,
    discriminator_lr=0.0001,
    discriminator_steps=2,
    verbose=True
)
ctgan2.fit(sample)
evaluate_model(ctgan2, "CTGAN Run 2")

# ── CTGAN RUN 3 ──
# Small batch, very slow learning,
# more discriminator steps
print("\n" + "="*60)
print("CTGAN RUN 3 of 3")
print("Small batch, slow learning, most careful")
print("="*60)

ctgan3 = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=300,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=0.00005,
    discriminator_lr=0.00005,
    discriminator_steps=3,
    verbose=True
)
ctgan3.fit(sample)
evaluate_model(ctgan3, "CTGAN Run 3")

# ── TVAE ──
# Completely different architecture
# Often better on correlated tabular data
print("\n" + "="*60)
print("TVAE")
print("Variational autoencoder architecture")
print("Often beats CTGAN on correlated data")
print("="*60)

tvae = TVAESynthesizer(
    metadata,
    epochs=200,
    batch_size=500,
    compress_dims=(512, 512),
    decompress_dims=(512, 512),
    verbose=True
)
tvae.fit(sample)
evaluate_model(tvae, "TVAE")

# ── FINAL SCORES ──
print("\n" + "="*60)
print("ALL MODEL SCORES")
print("="*60)
for r in sorted(
        all_results,
        key=lambda x: x['score'],
        reverse=True):
    winner = " <- WINNER" \
        if r['name'] == best_name else ""
    print(f"  {r['name']:<20} "
          f"{r['score']:.4f}{winner}")
print("="*60)
print(f"\nBest model: {best_name}")
print(f"Best score: {best_score:.4f}")
print(f"Saved to:   data/best_model.pkl")

# ── GENERATE FINAL SYNTHETIC DATA ──
print("\nGenerating 50,000 synthetic orders"
      " using best model...")
synthetic = best_model.sample(num_rows=50000)
synthetic.to_csv(
    'data/synthetic_orders_best.csv',
    index=False)
print("Saved synthetic_orders_best.csv")

# ── DETAILED FINAL COMPARISON ──
print("\n" + "="*60)
print("FINAL COMPARISON: REAL vs BEST SYNTHETIC")
print("="*60)
print(f"\n{'Metric':<22} {'Real':>8} "
      f"{'Synthetic':>10} {'Diff':>8} "
      f"{'Match':>6}")
print("-"*58)

metrics = [
    ('Weekend ratio',
     'is_weekend',        'mean'),
    ('Peak hour ratio',
     'is_peak_hour',      'mean'),
    ('Night ratio',
     'is_night',          'mean'),
    ('Reorder ratio',
     'is_reorder',        'mean'),
    ('Early cart ratio',
     'is_early_in_cart',  'mean'),
    ('Avg hour',
     'order_hour_of_day', 'mean'),
    ('Avg day of week',
     'order_dow',         'mean'),
]

for name, col, agg in metrics:
    rv   = sample[col].astype(float).mean()
    sv   = synthetic[col].astype(float).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    print(f"{name:<22} {rv:>8.3f} "
          f"{sv:>10.3f} {diff:>8.3f} {match:>6}")

# Time of day
print("\nTime of day:")
print(f"{'Category':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Match':>6}")
for cat in ['morning', 'afternoon',
            'evening', 'late', 'night']:
    rv   = (sample['time_of_day'] == cat).mean()
    sv   = (synthetic['time_of_day'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# Order frequency
print("\nOrder frequency:")
print(f"{'Category':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Match':>6}")
for cat in ['weekly', 'biweekly',
            'monthly', 'first']:
    rv   = (sample['order_frequency'] == cat).mean()
    sv   = (synthetic['order_frequency'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# ── VISUALISATION ──
print("\nGenerating final comparison charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Best Model: {best_name}\n'
    f'Quality Score: {best_score:.4f}',
    fontsize=14, fontweight='bold')

# Binary features
binary_cols  = ['is_weekend', 'is_peak_hour',
                'is_night', 'is_reorder']
binary_names = ['Weekend', 'Peak Hour',
                'Night', 'Reorder']
real_vals = [sample[c].mean()
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
real_h = sample['order_hour_of_day']\
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
axes[0][1].set_title('Hour of Day Distribution')
axes[0][1].set_xlabel('Hour')
axes[0][1].set_ylabel('Proportion')
axes[0][1].legend()

# Day distribution
real_d = sample['order_dow']\
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

# All model scores
names   = [r['name']  for r in all_results]
scores  = [r['score'] for r in all_results]
colours = ['green' if n == best_name
           else 'steelblue' for n in names]
bars = axes[1][1].bar(names, scores,
                       color=colours)
axes[1][1].set_title(
    'All Model Scores\n(green = winner)')
axes[1][1].set_ylabel('Quality Score')
axes[1][1].set_ylim(0, 1)
axes[1][1].tick_params(
    axis='x', rotation=15)
for bar, v in zip(bars, scores):
    axes[1][1].text(
        bar.get_x() + bar.get_width()/2,
        v + 0.005,
        f'{v:.4f}',
        ha='center', fontsize=11,
        fontweight='bold')

plt.tight_layout()
plt.savefig(
    'data/best_model_comparison.png',
    dpi=150, bbox_inches='tight')
plt.show()

print("\nChart saved to "
      "data/best_model_comparison.png")
print(f"\nFINAL WINNER: {best_name}")
print(f"FINAL SCORE:  {best_score:.4f}")
print("\nAll done!")
print("\nFiles saved:")
print("  data/best_model.pkl")
print("  data/synthetic_orders_best.csv")
print("  data/best_model_comparison.png")