import pandas as pd
import matplotlib.pyplot as plt
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ──
print("Loading data...")
real_df      = pd.read_csv('data/clean_orders.csv').sample(
                   n=10000, random_state=42)
synthetic_df = pd.read_csv('data/synthetic_orders.csv')
baseline_df  = pd.read_csv('data/baseline_orders.csv')

# Use same columns for fair comparison
cols = ['aisle_id', 'department_id',
        'order_dow', 'order_hour_of_day']

real_df      = real_df[cols].reset_index(drop=True)
synthetic_df = synthetic_df[cols].reset_index(drop=True)
baseline_df  = baseline_df[cols].reset_index(drop=True)

print("Real data shape:      ", real_df.shape)
print("Synthetic data shape: ", synthetic_df.shape)
print("Baseline data shape:  ", baseline_df.shape)

# ── METADATA ──
print("\nDefining metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_df)

metadata.update_column(
    column_name='aisle_id',       sdtype='categorical')
metadata.update_column(
    column_name='department_id',  sdtype='categorical')
metadata.update_column(
    column_name='order_dow',      sdtype='categorical')
metadata.update_column(
    column_name='order_hour_of_day', sdtype='categorical')

# ── EVALUATE CTGAN ──
print("\nEvaluating CTGAN synthetic data quality...")
ctgan_quality = evaluate_quality(
    real_data=real_df,
    synthetic_data=synthetic_df,
    metadata=metadata,
    verbose=True
)

# ── EVALUATE BASELINE ──
print("\nEvaluating Baseline synthetic data quality...")
baseline_quality = evaluate_quality(
    real_data=real_df,
    synthetic_data=baseline_df,
    metadata=metadata,
    verbose=True
)

# ── PRINT SCORES ──
print("\n" + "="*50)
print("SYNTHETIC DATA QUALITY SCORES")
print("="*50)

ctgan_score    = ctgan_quality.get_score()
baseline_score = baseline_quality.get_score()

print(f"CTGAN Overall Score:    {ctgan_score:.4f}")
print(f"Baseline Overall Score: {baseline_score:.4f}")
print("="*50)

# ── DETAILED SCORES ──
print("\nCTGAN Detailed Scores:")
ctgan_details = ctgan_quality.get_details(
    property_name='Column Shapes')
print(ctgan_details)

print("\nBaseline Detailed Scores:")
baseline_details = baseline_quality.get_details(
    property_name='Column Shapes')
print(baseline_details)

# ── PLOT SCORES ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Synthetic Data Quality Evaluation',
             fontsize=14)

# Overall scores bar chart
scores  = [ctgan_score, baseline_score]
labels  = ['CTGAN', 'Baseline']
colours = ['green', 'orange']

axes[0].bar(labels, scores, color=colours)
axes[0].set_title('Overall Quality Score\n'
                  '(higher is better)')
axes[0].set_ylabel('Score (0-1)')
axes[0].set_ylim(0, 1)
for i, v in enumerate(scores):
    axes[0].text(i, v + 0.01,
                 f'{v:.4f}',
                 ha='center', fontsize=12)

# Column scores comparison
ctgan_cols    = ctgan_quality.get_details(
    property_name='Column Shapes')
baseline_cols = baseline_quality.get_details(
    property_name='Column Shapes')

col_names = ctgan_cols['Column'].tolist()
ctgan_col_scores    = ctgan_cols['Score'].tolist()
baseline_col_scores = baseline_cols['Score'].tolist()

x = range(len(col_names))
width = 0.35

axes[1].bar([i - width/2 for i in x],
            ctgan_col_scores,
            width, label='CTGAN',    color='green',
            alpha=0.8)
axes[1].bar([i + width/2 for i in x],
            baseline_col_scores,
            width, label='Baseline', color='orange',
            alpha=0.8)

axes[1].set_title('Quality Score per Column\n'
                  '(higher is better)')
axes[1].set_ylabel('Score (0-1)')
axes[1].set_ylim(0, 1)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(col_names, rotation=15)
axes[1].legend()

plt.tight_layout()
plt.savefig('data/quality_evaluation.png')
plt.show()

print("\nQuality evaluation chart saved to"
      " data/quality_evaluation.png")

# ── SUMMARY ──
print("\n" + "="*50)
print("SUMMARY FOR DISSERTATION")
print("="*50)
print(f"CTGAN achieved an overall quality score of "
      f"{ctgan_score:.4f} compared to the shuffling "
      f"baseline score of {baseline_score:.4f}.")
if ctgan_score > baseline_score:
    diff = ctgan_score - baseline_score
    print(f"CTGAN outperformed the baseline by "
          f"{diff:.4f} points, demonstrating that it "
          f"learned meaningful statistical structure "
          f"from the real order data beyond simple "
          f"marginal distributions.")
else:
    print("The baseline matched or outperformed CTGAN "
          "on column shape metrics. This suggests "
          "further CTGAN tuning may be beneficial.")
print("="*50)
print("\nDone!")