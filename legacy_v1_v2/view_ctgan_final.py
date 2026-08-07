import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CTGAN FINAL MODEL RESULTS (90.85%)")
print("="*60)

# Load real data
print("\nLoading real data...")
df = pd.read_csv('data/clean_orders_v2.csv')

# Load CTGAN Final model
print("Loading CTGAN Final model...")
model = CTGANSynthesizer.load(
    'data/ctgan_final.pkl')
print("Model loaded!")

# Generate synthetic data
print("\nGenerating 50,000 synthetic orders...")
synthetic = model.sample(num_rows=50000)
print(f"Generated shape: {synthetic.shape}")

# Save it for future reference
synthetic.to_csv(
    'data/synthetic_orders_final.csv',
    index=False)
print("Saved synthetic_orders_final.csv")

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

# ── COMPARISON ──
print("\n" + "="*60)
print("CTGAN FINAL - FULL RESULTS")
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
    sv   = (synthetic[
        'order_frequency'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# Quality score
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
print(f"\nCTGAN Final quality score: "
      f"{score:.4f}")
print(f"Best model score:          0.9138")

# ── SIDE BY SIDE CHART ──
print("\nGenerating comparison chart...")

# Load best synthetic for comparison
best_synthetic = pd.read_csv(
    'data/synthetic_orders_best.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'CTGAN Final (90.85%) vs '
    f'Best Model (91.38%)\n'
    f'Side by Side Comparison',
    fontsize=13, fontweight='bold')

binary_cols  = ['is_weekend', 'is_peak_hour',
                'is_night', 'is_reorder']
binary_names = ['Weekend', 'Peak Hour',
                'Night', 'Reorder']
real_vals = [real_props[c]
             for c in binary_cols]
best_vals = [best_synthetic[c].mean()
             for c in binary_cols]
final_vals = [synthetic[c].mean()
              for c in binary_cols]

x     = np.arange(len(binary_names))
width = 0.25

axes[0][0].bar(x - width, real_vals,
               width, label='Real',
               color='steelblue', alpha=0.9)
axes[0][0].bar(x, best_vals,
               width, label='Best (91.38%)',
               color='green', alpha=0.9)
axes[0][0].bar(x + width, final_vals,
               width, label='Final (90.85%)',
               color='orange', alpha=0.9)
axes[0][0].set_title('Binary Features\n'
                     'Real vs Best vs Final')
axes[0][0].set_xticks(x)
axes[0][0].set_xticklabels(binary_names)
axes[0][0].set_ylabel('Ratio')
axes[0][0].legend(fontsize=8)
axes[0][0].set_ylim(0, 1)

# Hour distribution
real_h  = df['order_hour_of_day']\
    .value_counts(normalize=True).sort_index()
best_h  = best_synthetic['order_hour_of_day']\
    .value_counts(normalize=True).sort_index()
final_h = synthetic['order_hour_of_day']\
    .value_counts(normalize=True).sort_index()

axes[0][1].plot(real_h.index, real_h.values,
                label='Real',
                color='steelblue', linewidth=2)
axes[0][1].plot(best_h.index, best_h.values,
                label='Best (91.38%)',
                color='green', linewidth=2,
                linestyle='--')
axes[0][1].plot(final_h.index, final_h.values,
                label='Final (90.85%)',
                color='orange', linewidth=2,
                linestyle=':')
axes[0][1].set_title('Hour Distribution')
axes[0][1].set_xlabel('Hour of Day')
axes[0][1].set_ylabel('Proportion')
axes[0][1].legend(fontsize=8)

# Day distribution
real_d  = df['order_dow']\
    .value_counts(normalize=True).sort_index()
best_d  = best_synthetic['order_dow']\
    .value_counts(normalize=True).sort_index()
final_d = synthetic['order_dow']\
    .value_counts(normalize=True).sort_index()

axes[1][0].plot(real_d.index, real_d.values,
                label='Real',
                color='steelblue', linewidth=2)
axes[1][0].plot(best_d.index, best_d.values,
                label='Best (91.38%)',
                color='green', linewidth=2,
                linestyle='--')
axes[1][0].plot(final_d.index, final_d.values,
                label='Final (90.85%)',
                color='orange', linewidth=2,
                linestyle=':')
axes[1][0].set_title('Day of Week Distribution')
axes[1][0].set_xlabel('Day (0=Sunday)')
axes[1][0].set_ylabel('Proportion')
axes[1][0].legend(fontsize=8)

# Score comparison
score_labels  = ['Real\n(ground truth)',
                 'Best Model\n(91.38%)',
                 'CTGAN Final\n(90.85%)']
score_values  = [1.0, 0.9138, score]
score_colours = ['steelblue', 'green', 'orange']

bars = axes[1][1].bar(
    score_labels, score_values,
    color=score_colours, alpha=0.85)
axes[1][1].set_title('Quality Score Comparison')
axes[1][1].set_ylabel('Quality Score')
axes[1][1].set_ylim(0.85, 1.0)
for bar, v in zip(bars, score_values):
    axes[1][1].text(
        bar.get_x() + bar.get_width()/2,
        v + 0.001,
        f'{v:.4f}',
        ha='center', fontsize=11,
        fontweight='bold')

plt.tight_layout()
plt.savefig(
    'data/ctgan_final_vs_best.png',
    dpi=150, bbox_inches='tight')
plt.show()

print("\nChart saved to "
      "data/ctgan_final_vs_best.png")
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"CTGAN Final score: {score:.4f}")
print(f"Best model score:  0.9138")
print(f"Perfect metrics:   {perfect}/7 ✅")
print(f"Good metrics:      {good}/7 🟡")
print(f"Needs work:        {bad}/7 ❌")
print("\nConclusion: Best model (91.38%) "
      "remains the winner")
print("="*60)
print("\nDone!")