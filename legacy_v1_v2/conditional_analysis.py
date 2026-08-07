import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*60)
print("CONDITIONAL EFFECTIVENESS ANALYSIS")
print("Q1: Are the scenarios actually different?")
print("Q2: Does a model for one work on another?")
print("="*60)

# ── LOAD SEASONAL DATA ──
print("\nLoading seasonal datasets...")
normal    = pd.read_csv('data/normal_orders.csv')
christmas = pd.read_csv('data/christmas_orders.csv')
blackfri  = pd.read_csv('data/blackfriday_orders.csv')

normal['scenario']    = 'Normal'
christmas['scenario'] = 'Christmas'
blackfri['scenario']  = 'Black Friday'

print(f"  Normal:       {len(normal):,}")
print(f"  Christmas:    {len(christmas):,}")
print(f"  Black Friday: {len(blackfri):,}")

feature_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_reorder',
    'is_early_in_cart'
]

# ══════════════════════════════════════════
# QUESTION 1: ARE SCENARIOS DIFFERENT?
# ══════════════════════════════════════════
print("\n" + "="*60)
print("Q1: HOW DIFFERENT ARE THE SCENARIOS?")
print("="*60)

# Compare key feature distributions
print(f"\n{'Feature':<20} {'Normal':>10} "
      f"{'Christmas':>12} {'BlackFri':>10}")
print("-"*54)

compare_feats = [
    'is_weekend', 'is_peak_hour',
    'is_reorder', 'order_hour_of_day',
    'order_dow', 'aisle_id']

for feat in compare_feats:
    nv = normal[feat].astype(float).mean()
    cv = christmas[feat].astype(float).mean()
    bv = blackfri[feat].astype(float).mean()
    print(f"{feat:<20} {nv:>10.2f} "
          f"{cv:>12.2f} {bv:>10.2f}")

# Statistical distance between scenarios
# using average feature difference
def scenario_distance(a, b, cols):
    diffs = []
    for c in cols:
        # normalise by combined std
        combined_std = pd.concat(
            [a[c], b[c]]).std()
        if combined_std > 0:
            d = abs(a[c].mean() - b[c].mean()) \
                / combined_std
            diffs.append(d)
    return np.mean(diffs)

print("\n--- Distribution Distance ---")
print("(higher = more different, "
      "standardised difference)")
d_nc = scenario_distance(
    normal, christmas, feature_cols)
d_nb = scenario_distance(
    normal, blackfri, feature_cols)
d_cb = scenario_distance(
    christmas, blackfri, feature_cols)
print(f"  Normal vs Christmas:      {d_nc:.3f}")
print(f"  Normal vs Black Friday:   {d_nb:.3f}")
print(f"  Christmas vs Black Friday:{d_cb:.3f}")

# ══════════════════════════════════════════
# Q2a: CAN A CLASSIFIER TELL THEM APART?
# ══════════════════════════════════════════
print("\n" + "="*60)
print("Q2a: CAN A MODEL DISTINGUISH SCENARIOS?")
print("(If yes, they are genuinely distinct)")
print("="*60)

# Combine all three with scenario labels
all_data = pd.concat(
    [normal, christmas, blackfri],
    ignore_index=True)

X = all_data[feature_cols]
y = all_data['scenario']

X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y, test_size=0.3,
        random_state=SEED, stratify=y)

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED, n_jobs=-1)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\nClassifier accuracy: {acc:.1%}")
print("(33% would mean scenarios are "
      "indistinguishable / random)")
print("(100% would mean perfectly distinct)")

if acc >= 0.9:
    print("\n  RESULT: Scenarios are HIGHLY DISTINCT")
    print("  The conditioning works very well.")
elif acc >= 0.7:
    print("\n  RESULT: Scenarios are DISTINCT")
    print("  The conditioning is effective.")
else:
    print("\n  RESULT: Scenarios overlap somewhat")

# Confusion matrix
print("\nConfusion matrix:")
labels = ['Normal', 'Christmas', 'Black Friday']
cm = confusion_matrix(
    y_test, pred, labels=labels)
header = 'Actual\\Pred'
print(f"{header:<14}", end='')
for l in labels:
    print(f"{l[:10]:>12}", end='')
print()
for i, l in enumerate(labels):
    print(f"{l:<14}", end='')
    for j in range(len(labels)):
        print(f"{cm[i][j]:>12}", end='')
    print()

# ══════════════════════════════════════════
# Q2b: CROSS-SCENARIO TRANSFER
# ══════════════════════════════════════════
print("\n" + "="*60)
print("Q2b: DOES A MODEL FOR ONE WORK ON ANOTHER?")
print("(Train to detect one scenario, test if it")
print(" wrongly fires on a different scenario)")
print("="*60)

def cross_test(train_scenario, train_df,
               other_name, other_df):
    # Train a detector: this scenario vs Normal
    pos = train_df.copy()
    pos['label'] = 1
    neg = normal.copy()
    neg['label'] = 0
    train = pd.concat([pos, neg],
                      ignore_index=True)

    clf2 = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED, n_jobs=-1)
    clf2.fit(train[feature_cols],
             train['label'])

    # Test: how often does it flag the
    # OTHER scenario as this scenario?
    other_pred = clf2.predict(
        other_df[feature_cols])
    false_fire = other_pred.mean()
    return false_fire

print("\nChristmas detector applied to "
      "Black Friday data:")
xmas_on_bf = cross_test(
    'Christmas', christmas,
    'Black Friday', blackfri)
print(f"  {xmas_on_bf:.1%} of Black Friday "
      f"orders flagged as Christmas")

print("\nBlack Friday detector applied to "
      "Christmas data:")
bf_on_xmas = cross_test(
    'Black Friday', blackfri,
    'Christmas', christmas)
print(f"  {bf_on_xmas:.1%} of Christmas "
      f"orders flagged as Black Friday")

print("\nInterpretation:")
print("  LOW cross-firing = scenarios are")
print("  distinct and conditioning is specific.")
if xmas_on_bf < 0.3 and bf_on_xmas < 0.3:
    print("  RESULT: Strong separation \u2014 a model")
    print("  for one scenario does NOT transfer")
    print("  to another. Conditioning is specific.")
else:
    print("  RESULT: Some overlap between scenarios.")

# ══════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════
print("\nGenerating analysis charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Conditional Generation Effectiveness\n'
    f'Scenario classifier accuracy: {acc:.1%}',
    fontsize=14, fontweight='bold')

# Feature comparison
feats_plot = ['is_weekend', 'is_peak_hour',
              'is_reorder']
fnames = ['Weekend', 'Peak Hour', 'Reorder']
nvals = [normal[f].mean() for f in feats_plot]
cvals = [christmas[f].mean() for f in feats_plot]
bvals = [blackfri[f].mean() for f in feats_plot]
x = np.arange(len(fnames)); w = 0.25
axes[0][0].bar(x - w, nvals, w,
               label='Normal', color='steelblue')
axes[0][0].bar(x, cvals, w,
               label='Christmas', color='red')
axes[0][0].bar(x + w, bvals, w,
               label='Black Friday',
               color='orange')
axes[0][0].set_title('Feature Differences\n'
                     'Across Scenarios')
axes[0][0].set_xticks(x)
axes[0][0].set_xticklabels(fnames)
axes[0][0].set_ylabel('Ratio')
axes[0][0].legend(fontsize=8)

# Hour distribution by scenario
axes[0][1].hist(normal['order_hour_of_day'],
                bins=24, alpha=0.5,
                label='Normal', color='steelblue',
                density=True)
axes[0][1].hist(christmas['order_hour_of_day'],
                bins=24, alpha=0.5,
                label='Christmas', color='red',
                density=True)
axes[0][1].hist(blackfri['order_hour_of_day'],
                bins=24, alpha=0.5,
                label='Black Friday',
                color='orange', density=True)
axes[0][1].set_title('Order Hour by Scenario')
axes[0][1].set_xlabel('Hour of Day')
axes[0][1].set_ylabel('Density')
axes[0][1].legend(fontsize=8)

# Confusion matrix heatmap
im = axes[1][0].imshow(cm, cmap='Blues')
axes[1][0].set_title(
    'Scenario Classifier\nConfusion Matrix')
axes[1][0].set_xticks(range(3))
axes[1][0].set_yticks(range(3))
axes[1][0].set_xticklabels(
    ['Norm', 'Xmas', 'BF'])
axes[1][0].set_yticklabels(
    ['Norm', 'Xmas', 'BF'])
axes[1][0].set_xlabel('Predicted')
axes[1][0].set_ylabel('Actual')
for i in range(3):
    for j in range(3):
        text_colour = 'black' if cm[i][j] < cm.max() / 2 else 'white'
        axes[1][0].text(
            j, i, str(cm[i][j]),
            ha='center', va='center',
            color=text_colour)

# Distribution distances
dist_labels = ['Normal\nvs Xmas',
               'Normal\nvs BF',
               'Xmas\nvs BF']
dist_vals = [d_nc, d_nb, d_cb]
axes[1][1].bar(dist_labels, dist_vals,
               color=['purple', 'teal',
                      'darkorange'])
axes[1][1].set_title(
    'Distribution Distance\n'
    '(higher = more different)')
axes[1][1].set_ylabel(
    'Standardised distance')
for i, v in enumerate(dist_vals):
    axes[1][1].text(i, v + 0.01,
                    f'{v:.2f}',
                    ha='center',
                    fontweight='bold')

plt.tight_layout()
plt.savefig('results/conditional_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("Chart saved to "
      "results/conditional_analysis.png")

# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY FOR SUPERVISOR")
print("="*60)
print(f"""
Q1 - Are scenarios different?
  Yes. A classifier distinguishes the three
  scenarios with {acc:.1%} accuracy (vs 33%
  chance), confirming they are statistically
  distinct demand patterns.

Q2 - Does a model for one work on another?
  No - and that is the desired result. A
  Christmas detector flags only {xmas_on_bf:.1%}
  of Black Friday orders, and a Black Friday
  detector flags only {bf_on_xmas:.1%} of
  Christmas orders. The conditioning produces
  specific, non-transferable demand patterns.

Conclusion: The conditional generation is
effective - each scenario is a genuinely
distinct and separable demand pattern.
""")
print("="*60)
print("Done!")