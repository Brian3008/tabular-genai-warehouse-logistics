import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*60)
print("ML EFFICACY — 10% MODEL")
print("Can a model trained on 10%-model")
print("synthetic data work on real data?")
print("="*60)

# ── LOAD DATA ──
print("\nLoading data...")
real = pd.read_csv('data/fixed_real_compare.csv')
synthetic = pd.read_csv(
    'data/synthetic_10percent.csv')

real_train, real_test = train_test_split(
    real, test_size=0.3, random_state=SEED)

print(f"Real train:  {len(real_train):,}")
print(f"Real test:   {len(real_test):,}")
print(f"Synthetic:   {len(synthetic):,}")

feature_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_early_in_cart'
]
target_col = 'is_reorder'

# ── TRAIN ON REAL ──
print("\n[1] Training on REAL data...")
clf_real = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED, n_jobs=-1)
clf_real.fit(
    real_train[feature_cols],
    real_train[target_col])
pred_real = clf_real.predict(
    real_test[feature_cols])
prob_real = clf_real.predict_proba(
    real_test[feature_cols])[:, 1]
acc_real = accuracy_score(
    real_test[target_col], pred_real)
f1_real  = f1_score(
    real_test[target_col], pred_real)
auc_real = roc_auc_score(
    real_test[target_col], prob_real)
print(f"  Accuracy: {acc_real:.4f}")
print(f"  F1:       {f1_real:.4f}")
print(f"  AUC:      {auc_real:.4f}")

# ── TRAIN ON 10% SYNTHETIC ──
print("\n[2] Training on 10% SYNTHETIC data...")
clf_synth = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED, n_jobs=-1)
clf_synth.fit(
    synthetic[feature_cols],
    synthetic[target_col])
pred_synth = clf_synth.predict(
    real_test[feature_cols])
prob_synth = clf_synth.predict_proba(
    real_test[feature_cols])[:, 1]
acc_synth = accuracy_score(
    real_test[target_col], pred_synth)
f1_synth  = f1_score(
    real_test[target_col], pred_synth)
auc_synth = roc_auc_score(
    real_test[target_col], prob_synth)
print(f"  Accuracy: {acc_synth:.4f}")
print(f"  F1:       {f1_synth:.4f}")
print(f"  AUC:      {auc_synth:.4f}")

# ── EFFICACY ──
print("\n" + "="*60)
print("EFFICACY RESULTS")
print("="*60)
print(f"\n{'Metric':<12}{'Real':>8}"
      f"{'Synthetic':>10}{'Ratio':>8}")
print("-"*40)

acc_ratio = acc_synth / acc_real
f1_ratio  = f1_synth / f1_real \
    if f1_real > 0 else 0
auc_ratio = auc_synth / auc_real

print(f"{'Accuracy':<12}{acc_real:>8.4f}"
      f"{acc_synth:>10.4f}"
      f"{acc_ratio:>7.1%}")
print(f"{'F1':<12}{f1_real:>8.4f}"
      f"{f1_synth:>10.4f}"
      f"{f1_ratio:>7.1%}")
print(f"{'AUC':<12}{auc_real:>8.4f}"
      f"{auc_synth:>10.4f}"
      f"{auc_ratio:>7.1%}")

overall = np.mean(
    [acc_ratio, f1_ratio, auc_ratio])
print(f"\nOVERALL EFFICACY: {overall:.1%}")

print("\nComparison:")
print(f"  Full model efficacy: 100.3%")
print(f"  10% model efficacy:  {overall:.1%}")

if overall >= 0.95:
    print("\n  EXCELLENT — 10% model synthetic")
    print("  data works as well as real data")
elif overall >= 0.90:
    print("\n  VERY GOOD — small drop from")
    print("  full model but still strong")
elif overall >= 0.80:
    print("\n  GOOD — usable with some loss")
else:
    print("\n  MODERATE — more training needed")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print(f"""
Full model (93.47% quality): 100.3% efficacy
10% model  (91.21% quality): {overall:.1%} efficacy

This confirms whether synthetic data from
a smaller training set is still useful
for downstream machine learning tasks.
""")
print("Done!")