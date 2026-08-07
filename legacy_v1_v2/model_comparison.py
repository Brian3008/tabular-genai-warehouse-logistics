import pandas as pd
import numpy as np
import random
import torch
from sdv.single_table import (
    CTGANSynthesizer, TVAESynthesizer,
    GaussianCopulaSynthesizer)
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("="*60)
print("MODEL COMPARISON TABLE")
print("CTGAN vs TVAE vs GaussianCopula vs Baseline")
print("="*60)

# ── LOAD DATA ──
print("\nLoading data...")
real_eval  = pd.read_csv('data/fixed_real_eval.csv')
real_props = pd.read_csv(
    'data/real_proportions.csv').iloc[0].to_dict()
df_full    = pd.read_csv('data/clean_orders_v2.csv')

eval_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_reorder',
    'is_early_in_cart'
]

eval_metadata = SingleTableMetadata()
eval_metadata.detect_from_dataframe(
    real_eval[eval_cols])
for col in eval_cols:
    eval_metadata.update_column(
        column_name=col, sdtype='categorical')

# Real test set for ML efficacy
real_compare = pd.read_csv(
    'data/fixed_real_compare.csv')
_, real_test = train_test_split(
    real_compare, test_size=0.3,
    random_state=SEED)

feature_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_early_in_cart']
target_col = 'is_reorder'

# Real baseline efficacy
clf_real = RandomForestClassifier(
    n_estimators=100, random_state=SEED, n_jobs=-1)
clf_real.fit(real_compare[feature_cols],
             real_compare[target_col])
real_acc = accuracy_score(
    real_test[target_col],
    clf_real.predict(real_test[feature_cols]))

# ── HELPER FUNCTIONS ──
def quality_score(synth):
    q = evaluate_quality(
        real_data=real_eval[eval_cols],
        synthetic_data=synth[eval_cols].sample(
            n=min(10000, len(synth)),
            random_state=SEED),
        metadata=eval_metadata, verbose=False)
    return q.get_score()

def ml_efficacy(synth):
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED, n_jobs=-1)
    clf.fit(synth[feature_cols],
            synth[target_col])
    acc = accuracy_score(
        real_test[target_col],
        clf.predict(real_test[feature_cols]))
    return acc / real_acc

def corr_similarity(synth):
    num_cols = feature_cols + [target_col]
    rc = real_compare[num_cols].astype(
        float).corr()
    sc = synth[num_cols].astype(float).corr()
    mask = ~np.eye(len(num_cols), dtype=bool)
    diff = (rc - sc).abs().values[mask].mean()
    return 1 - diff

# ── EVALUATE EACH MODEL ──
results = []

# 1. CTGAN (your final synthetic data)
print("\nEvaluating CTGAN (final)...")
ctgan_data = pd.read_csv(
    'data/FINAL_synthetic_orders.csv')
results.append({
    'Model':    'CTGAN (final)',
    'Quality':  quality_score(ctgan_data),
    'Correlation': corr_similarity(ctgan_data),
    'ML Efficacy': ml_efficacy(ctgan_data),
})

# 2. Shuffling Baseline
print("Evaluating Shuffling Baseline...")
baseline = df_full.sample(
    n=50000, random_state=SEED).copy()
for col in eval_cols + [target_col]:
    baseline[col] = df_full[col].sample(
        frac=1, random_state=SEED
    ).values[:50000]
results.append({
    'Model':    'Shuffling Baseline',
    'Quality':  quality_score(baseline),
    'Correlation': corr_similarity(baseline),
    'ML Efficacy': ml_efficacy(baseline),
})

# 3. GaussianCopula (quick train)
print("Training GaussianCopula (fast)...")
sample = df_full.sample(
    n=50000, random_state=SEED)
gc_metadata = SingleTableMetadata()
gc_metadata.detect_from_dataframe(sample)
for col in eval_cols:
    gc_metadata.update_column(
        column_name=col, sdtype='categorical')
gc_metadata.update_column(
    column_name='days_since_prior_order',
    sdtype='numerical')
gc_metadata.update_column(
    column_name='order_id', sdtype='id')
gc_metadata.update_column(
    column_name='time_of_day',
    sdtype='categorical')
gc_metadata.update_column(
    column_name='order_frequency',
    sdtype='categorical')
gc_metadata.update_column(
    column_name='aisle_popularity',
    sdtype='categorical')

copula = GaussianCopulaSynthesizer(gc_metadata)
copula.fit(sample)
copula_data = copula.sample(num_rows=50000)
results.append({
    'Model':    'GaussianCopula',
    'Quality':  quality_score(copula_data),
    'Correlation': corr_similarity(copula_data),
    'ML Efficacy': ml_efficacy(copula_data),
})

# ── RESULTS TABLE ──
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
print(f"\n{'Model':<22} {'Quality':>8} "
      f"{'Corr':>8} {'ML Eff':>8}")
print("-"*48)
for r in results:
    print(f"{r['Model']:<22} "
          f"{r['Quality']*100:>7.1f}% "
          f"{r['Correlation']*100:>7.1f}% "
          f"{r['ML Efficacy']*100:>7.1f}%")
print("-"*48)

# Save table
results_df = pd.DataFrame(results)
results_df.to_csv(
    'data/model_comparison.csv', index=False)
print("\nSaved model_comparison.csv")

# ── CHART ──
fig, ax = plt.subplots(figsize=(12, 6))
models = [r['Model'] for r in results]
quality = [r['Quality']*100 for r in results]
corr    = [r['Correlation']*100 for r in results]
mleff   = [min(r['ML Efficacy']*100, 105)
           for r in results]

x = np.arange(len(models))
w = 0.25
ax.bar(x - w, quality, w,
       label='Quality', color='steelblue')
ax.bar(x, corr, w,
       label='Correlation', color='green')
ax.bar(x + w, mleff, w,
       label='ML Efficacy', color='orange')
ax.set_title(
    'Model Comparison Across Three Metrics',
    fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Score (%)')
ax.legend()
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig('data/model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("Chart saved to data/model_comparison.png")
print("\nDone!")