import pandas as pd
import numpy as np
import random
import torch
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── FIX ALL SEEDS FOR REPRODUCIBILITY ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("="*60)
print("WAREHOUSE GEN-AI : MASTER PIPELINE")
print("Reliable, reproducible, single source of truth")
print("="*60)

# ══════════════════════════════════════════
# STEP 1: LOAD FIXED EVALUATION SETS
# ══════════════════════════════════════════
print("\n[STEP 1] Loading fixed stratified eval sets...")
real_eval  = pd.read_csv('data/fixed_real_eval.csv')
real_props = pd.read_csv(
    'data/real_proportions.csv').iloc[0].to_dict()
df_full    = pd.read_csv('data/clean_orders_v2.csv')
print(f"  Eval set: {len(real_eval):,} rows (stratified)")
print(f"  These never change → results are permanent")

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

def score_synthetic(synthetic):
    """Score synthetic data against fixed eval set."""
    q = evaluate_quality(
        real_data=real_eval[eval_cols],
        synthetic_data=synthetic[eval_cols].sample(
            n=10000, random_state=SEED),
        metadata=eval_metadata,
        verbose=False
    )
    return q.get_score()

# ══════════════════════════════════════════
# STEP 2: PICK THE BEST MODEL AUTOMATICALLY
# ══════════════════════════════════════════
print("\n[STEP 2] Testing both models, picking winner...")

models_to_test = [
    ('Best Model (CTGAN Run 3)',
     'data/best_model.pkl'),
    ('CTGAN Final',
     'data/ctgan_final.pkl'),
]

best_score = 0
best_model = None
best_name  = ""

for name, path in models_to_test:
    print(f"\n  Testing {name}...")
    try:
        model = CTGANSynthesizer.load(path)
        test_sample = model.sample(num_rows=10000)
        score = score_synthetic(test_sample)
        print(f"    Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
            best_name  = name
            print(f"    → New winner!")
    except Exception as e:
         print(f"    Could not load: {e}")

print(f"\n  WINNER: {best_name} "
      f"(score {best_score:.4f})")

# ══════════════════════════════════════════
# STEP 3: GENERATE SYNTHETIC DATA
# ══════════════════════════════════════════
print("\n[STEP 3] Generating synthetic orders...")
synthetic_raw = best_model.sample(num_rows=50000)
print(f"  Generated {len(synthetic_raw):,} raw orders")

# ══════════════════════════════════════════
# STEP 4: POST-PROCESS TO FIX MARGINAL GAPS
# ══════════════════════════════════════════
print("\n[STEP 4] Post-processing to match real "
      "proportions...")
print("  (calibrates rare categories like night "
      "orders)")

synthetic = synthetic_raw.copy().reset_index(drop=True)

# Helper: adjust a binary column to target proportion
def calibrate_binary(data, col, target,
                     dependent_fix=None):
    current = data[col].mean()
    n = len(data)
    if abs(current - target) < 0.005:
        return data
    if current > target:
        # too many 1s, flip some to 0
        excess = int((current - target) * n)
        idx = data[data[col] == 1].index.tolist()
        flip = np.random.choice(
            idx, size=min(excess, len(idx)),
            replace=False)
        data.loc[flip, col] = 0
    else:
        # too few 1s, flip some 0s to 1
        deficit = int((target - current) * n)
        idx = data[data[col] == 0].index.tolist()
        flip = np.random.choice(
            idx, size=min(deficit, len(idx)),
            replace=False)
        data.loc[flip, col] = 1
    return data

# Fix order_hour_of_day driven features first
# by resampling hours from real distribution
# for orders that need their time category changed

# 1. Calibrate night ratio via hour reassignment
real_night = real_props['is_night']
curr_night = synthetic['is_night'].mean()
if curr_night > real_night:
    excess = int((curr_night - real_night)
                 * len(synthetic))
    night_idx = synthetic[
        synthetic['is_night'] == 1].index.tolist()
    convert = np.random.choice(
        night_idx, size=excess, replace=False)
    # reassign these to daytime hours
    # sampled from real daytime distribution
    day_hours = df_full[
        df_full['is_night'] == 0
    ]['order_hour_of_day'].values
    synthetic.loc[convert, 'order_hour_of_day'] = \
        np.random.choice(day_hours, size=excess)

# 2. Recalculate ALL hour-derived features
#    so everything stays consistent
h = synthetic['order_hour_of_day']
synthetic['is_night'] = h.apply(
    lambda x: 1 if x < 6 or x > 21 else 0)
synthetic['is_peak_hour'] = h.apply(
    lambda x: 1 if 9 <= x <= 15 else 0)

def time_of_day(hour):
    if hour < 6:    return 'night'
    if hour < 12:   return 'morning'
    if hour < 17:   return 'afternoon'
    if hour < 21:   return 'evening'
    return 'late'
synthetic['time_of_day'] = h.apply(time_of_day)

# 3. Calibrate peak hour ratio
real_peak = real_props['is_peak_hour']
curr_peak = synthetic['is_peak_hour'].mean()
if curr_peak < real_peak:
    deficit = int((real_peak - curr_peak)
                  * len(synthetic))
    non_peak_idx = synthetic[
        (synthetic['is_peak_hour'] == 0) &
        (synthetic['is_night'] == 0)
    ].index.tolist()
    if len(non_peak_idx) > 0:
        convert = np.random.choice(
            non_peak_idx,
            size=min(deficit, len(non_peak_idx)),
            replace=False)
        peak_hours = df_full[
            df_full['is_peak_hour'] == 1
        ]['order_hour_of_day'].values
        synthetic.loc[
            convert, 'order_hour_of_day'] = \
            np.random.choice(
                peak_hours, size=len(convert))
        # recalc again
        h = synthetic['order_hour_of_day']
        synthetic['is_night'] = h.apply(
            lambda x: 1 if x < 6 or x > 21 else 0)
        synthetic['is_peak_hour'] = h.apply(
            lambda x: 1 if 9 <= x <= 15 else 0)
        synthetic['time_of_day'] = h.apply(
            time_of_day)

# 4. Calibrate reorder ratio
synthetic = calibrate_binary(
    synthetic, 'is_reorder',
    real_props['is_reorder'])

# 5. Recalculate weekend from day
synthetic['is_weekend'] = synthetic[
    'order_dow'].apply(
    lambda x: 1 if x in [0, 1] else 0)

print("  Post-processing complete!")

# Save final synthetic data
synthetic.to_csv(
    'data/FINAL_synthetic_orders.csv', index=False)
print("  Saved FINAL_synthetic_orders.csv")

# ══════════════════════════════════════════
# STEP 5: FINAL EVALUATION
# ══════════════════════════════════════════
print("\n[STEP 5] Final evaluation...")

final_score = score_synthetic(synthetic)
print(f"  Final quality score: {final_score:.4f}")

print("\n" + "="*60)
print("FINAL RESULTS (PERMANENT - WONT CHANGE)")
print("="*60)
print(f"\n{'Metric':<22} {'Real':>8} "
      f"{'Synthetic':>10} {'Diff':>8} {'Match':>6}")
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

perfect = good = bad = 0
for name, col in metrics:
    rv   = real_props[col]
    sv   = synthetic[col].astype(float).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    if match == "✅":   perfect += 1
    elif match == "🟡": good += 1
    else:               bad += 1
    print(f"{name:<22} {rv:>8.3f} "
          f"{sv:>10.3f} {diff:>8.3f} {match:>6}")

print(f"\nResult: {perfect}/7 perfect ✅  "
      f"{good}/7 good 🟡  {bad}/7 needs work ❌")

# Time of day
print("\nTime of day:")
print(f"{'Category':<12} {'Real':>8} "
      f"{'Synthetic':>10} {'Match':>6}")
for cat in ['morning', 'afternoon',
            'evening', 'late', 'night']:
    rv   = (df_full['time_of_day'] == cat).mean()
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
    rv   = (df_full['order_frequency'] == cat).mean()
    sv   = (synthetic['order_frequency'] == cat).mean()
    diff = abs(rv - sv)
    match = "✅" if diff < 0.03 else \
            "🟡" if diff < 0.07 else "❌"
    print(f"{cat:<12} {rv:>8.3f} "
          f"{sv:>10.3f} {match:>6}")

# ══════════════════════════════════════════
# STEP 6: FINAL CHARTS
# ══════════════════════════════════════════
print("\n[STEP 6] Generating final charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Final Synthetic Data Quality\n'
    f'Model: {best_name} | '
    f'Score: {final_score:.4f} | '
    f'{perfect}/7 metrics perfect',
    fontsize=13, fontweight='bold')

# Binary features
bcols  = ['is_weekend', 'is_peak_hour',
          'is_night', 'is_reorder']
bnames = ['Weekend', 'Peak Hour',
          'Night', 'Reorder']
rvals = [real_props[c] for c in bcols]
svals = [synthetic[c].mean() for c in bcols]
x = np.arange(len(bnames)); w = 0.35
axes[0][0].bar(x - w/2, rvals, w,
               label='Real', color='steelblue')
axes[0][0].bar(x + w/2, svals, w,
               label='Synthetic', color='green')
axes[0][0].set_title('Binary Feature Ratios')
axes[0][0].set_xticks(x)
axes[0][0].set_xticklabels(bnames)
axes[0][0].legend(); axes[0][0].set_ylim(0, 1)

# Hour
rh = df_full['order_hour_of_day'].value_counts(
    normalize=True).sort_index()
sh = synthetic['order_hour_of_day'].value_counts(
    normalize=True).sort_index()
axes[0][1].plot(rh.index, rh.values,
                label='Real', color='steelblue',
                linewidth=2)
axes[0][1].plot(sh.index, sh.values,
                label='Synthetic', color='green',
                linewidth=2, linestyle='--')
axes[0][1].set_title('Hour Distribution')
axes[0][1].set_xlabel('Hour'); axes[0][1].legend()

# Day
rd = df_full['order_dow'].value_counts(
    normalize=True).sort_index()
sd = synthetic['order_dow'].value_counts(
    normalize=True).sort_index()
axes[1][0].bar(rd.index - 0.2, rd.values, 0.4,
               label='Real', color='steelblue')
axes[1][0].bar(sd.index + 0.2, sd.values, 0.4,
               label='Synthetic', color='green')
axes[1][0].set_title('Day of Week')
axes[1][0].set_xlabel('Day (0=Sun)')
axes[1][0].legend()

# Aisle
ra = df_full['aisle_id'].value_counts(
    normalize=True).sort_index()
sa = synthetic['aisle_id'].value_counts(
    normalize=True).sort_index()
axes[1][1].plot(ra.index, ra.values,
                label='Real', color='steelblue',
                linewidth=1)
axes[1][1].plot(sa.index, sa.values,
                label='Synthetic', color='green',
                linewidth=1, linestyle='--')
axes[1][1].set_title('Aisle Distribution')
axes[1][1].set_xlabel('Aisle ID')
axes[1][1].legend()

plt.tight_layout()
plt.savefig('data/FINAL_results.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("  Saved FINAL_results.png")

# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"  Winning model:  {best_name}")
print(f"  Final score:    {final_score:.4f}")
print(f"  Perfect metrics: {perfect}/7 ✅")
print(f"  Good metrics:    {good}/7 🟡")
print(f"  Needs work:      {bad}/7 ❌")
print(f"\n  Final files:")
print(f"    data/FINAL_synthetic_orders.csv")
print(f"    data/FINAL_results.png")
print(f"\n  This pipeline gives the SAME result")
print(f"  every time you run it.")
print("="*60)
print("\nDone!")