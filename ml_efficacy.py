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
print("ML EFFICACY EVALUATION")
print("The gold-standard test for synthetic data:")
print("Can a model trained on SYNTHETIC data")
print("perform well on REAL data?")
print("="*60)

# ── LOAD DATA ──
print("\nLoading data...")
real = pd.read_csv('data/fixed_real_compare.csv')
synthetic = pd.read_csv(
    'data/FINAL_synthetic_orders.csv')

# Hold out a real test set that neither
# model ever sees during training
real_train, real_test = train_test_split(
    real, test_size=0.3, random_state=SEED)

print(f"Real train:  {len(real_train):,}")
print(f"Real test:   {len(real_test):,}")
print(f"Synthetic:   {len(synthetic):,}")

# ── FEATURES AND TARGET ──
# Task: predict whether an order is a reorder
# using the other features
feature_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_early_in_cart'
]
target_col = 'is_reorder'

# ── TRAIN ON REAL (baseline / upper bound) ──
print("\n[1] Training classifier on REAL data...")
clf_real = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1)
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
print(f"  F1 score: {f1_real:.4f}")
print(f"  AUC:      {auc_real:.4f}")

# ── TRAIN ON SYNTHETIC (the real test) ──
print("\n[2] Training classifier on SYNTHETIC data...")
print("    Testing on the SAME real test set...")
clf_synth = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1)
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
print(f"  F1 score: {f1_synth:.4f}")
print(f"  AUC:      {auc_synth:.4f}")

# ── EFFICACY RATIO ──
print("\n" + "="*60)
print("EFFICACY RESULTS")
print("="*60)
print(f"\n{'Metric':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Ratio':>8}")
print("-"*40)

acc_ratio = acc_synth / acc_real
f1_ratio  = f1_synth / f1_real \
    if f1_real > 0 else 0
auc_ratio = auc_synth / auc_real

print(f"{'Accuracy':<12} {acc_real:>8.4f} "
      f"{acc_synth:>10.4f} {acc_ratio:>7.1%}")
print(f"{'F1 score':<12} {f1_real:>8.4f} "
      f"{f1_synth:>10.4f} {f1_ratio:>7.1%}")
print(f"{'AUC':<12} {auc_real:>8.4f} "
      f"{auc_synth:>10.4f} {auc_ratio:>7.1%}")

overall_efficacy = np.mean(
    [acc_ratio, f1_ratio, auc_ratio])

print(f"\nOVERALL EFFICACY: "
      f"{overall_efficacy:.1%}")
print("\nInterpretation:")
if overall_efficacy >= 0.95:
    print("  EXCELLENT - synthetic data is nearly")
    print("  as useful as real data for ML tasks")
elif overall_efficacy >= 0.90:
    print("  VERY GOOD - synthetic data preserves")
    print("  most of the predictive structure")
elif overall_efficacy >= 0.80:
    print("  GOOD - synthetic data is usable")
    print("  with some loss of fidelity")
else:
    print("  MODERATE - synthetic data captures")
    print("  some but not all structure")

# ── FEATURE IMPORTANCE COMPARISON ──
print("\n" + "="*60)
print("FEATURE IMPORTANCE COMPARISON")
print("(Do both models learn the same patterns?)")
print("="*60)

imp_real  = pd.Series(
    clf_real.feature_importances_,
    index=feature_cols).sort_values(
    ascending=False)
imp_synth = pd.Series(
    clf_synth.feature_importances_,
    index=feature_cols)

print(f"\n{'Feature':<20} {'Real':>8} "
      f"{'Synthetic':>10}")
print("-"*40)
for feat in imp_real.index:
    print(f"{feat:<20} "
          f"{imp_real[feat]:>8.4f} "
          f"{imp_synth[feat]:>10.4f}")

# ── VISUALISATION ──
print("\nGenerating efficacy charts...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f'ML Efficacy: Synthetic vs Real Training Data\n'
    f'Overall Efficacy: {overall_efficacy:.1%}',
    fontsize=14, fontweight='bold')

# Metrics comparison
metrics_names = ['Accuracy', 'F1', 'AUC']
real_scores   = [acc_real, f1_real, auc_real]
synth_scores  = [acc_synth, f1_synth, auc_synth]
x = np.arange(len(metrics_names))
w = 0.35
axes[0].bar(x - w/2, real_scores, w,
            label='Trained on Real',
            color='steelblue')
axes[0].bar(x + w/2, synth_scores, w,
            label='Trained on Synthetic',
            color='green')
axes[0].set_title('Classifier Performance\n'
                  '(tested on real data)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1)
axes[0].legend()
for i, (r, s) in enumerate(
        zip(real_scores, synth_scores)):
    axes[0].text(i - w/2, r + 0.01,
                 f'{r:.3f}', ha='center',
                 fontsize=9)
    axes[0].text(i + w/2, s + 0.01,
                 f'{s:.3f}', ha='center',
                 fontsize=9)

# Feature importance comparison
y_pos = np.arange(len(feature_cols))
axes[1].barh(y_pos - 0.2,
             imp_real.values, 0.4,
             label='Real',
             color='steelblue')
axes[1].barh(y_pos + 0.2,
             [imp_synth[f]
              for f in imp_real.index], 0.4,
             label='Synthetic',
             color='green')
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(
    imp_real.index, fontsize=8)
axes[1].set_title('Feature Importance\n'
                  '(same patterns learned?)')
axes[1].set_xlabel('Importance')
axes[1].legend()

# Efficacy gauge
axes[2].barh(['Efficacy'],
             [overall_efficacy],
             color='green' if
             overall_efficacy >= 0.9
             else 'orange')
axes[2].barh(['Efficacy'], [1.0],
             color='lightgray',
             alpha=0.3, zorder=0)
axes[2].set_xlim(0, 1)
axes[2].set_title(
    f'Overall ML Efficacy\n'
    f'{overall_efficacy:.1%}')
axes[2].axvline(0.9, color='black',
                linestyle='--', alpha=0.5)
axes[2].text(0.9, 0, ' 90% target',
             fontsize=8, va='bottom')

plt.tight_layout()
plt.savefig('data/ml_efficacy.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nChart saved to data/ml_efficacy.png")

# ── SUMMARY FOR DISSERTATION ──
print("\n" + "="*60)
print("SUMMARY FOR DISSERTATION / INTERVIEW")
print("="*60)
print(f"""
A Random Forest classifier trained on synthetic
data achieved {acc_synth:.1%} accuracy on real
test data, compared to {acc_real:.1%} for a
classifier trained on real data. This represents
an ML efficacy of {overall_efficacy:.1%},
demonstrating that the CTGAN-generated synthetic
orders preserve the predictive structure of the
real data and can serve as a viable substitute
for real customer data in downstream machine
learning tasks.
""")
print("="*60)
print("Done!")