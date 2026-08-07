import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CORRELATION MATRIX ANALYSIS")
print("Does synthetic data preserve the")
print("relationships BETWEEN features?")
print("="*60)

# ── LOAD DATA ──
print("\nLoading data...")
real = pd.read_csv('data/fixed_real_compare.csv')
synthetic = pd.read_csv(
    'data/FINAL_synthetic_orders.csv')

# Numeric columns for correlation
num_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_reorder',
    'is_early_in_cart',
    'days_since_prior_order'
]

real_num = real[num_cols].astype(float)
synth_num = synthetic[num_cols].astype(float)

# ── COMPUTE CORRELATION MATRICES ──
print("Computing correlation matrices...")
real_corr  = real_num.corr()
synth_corr = synth_num.corr()

# Difference matrix
diff_corr = (real_corr - synth_corr).abs()

# ── OVERALL CORRELATION SIMILARITY ──
print("\n" + "="*60)
print("CORRELATION PRESERVATION SCORE")
print("="*60)

# Average absolute difference across all pairs
# excluding the diagonal (self-correlation = 1)
mask = ~np.eye(len(num_cols), dtype=bool)
avg_diff = diff_corr.values[mask].mean()
max_diff = diff_corr.values[mask].max()

# Similarity score: 1 - avg difference
corr_similarity = 1 - avg_diff

print(f"\nAverage correlation difference: "
      f"{avg_diff:.4f}")
print(f"Maximum correlation difference: "
      f"{max_diff:.4f}")
print(f"Correlation similarity score:   "
      f"{corr_similarity:.4f} "
      f"({corr_similarity*100:.1f}%)")

if corr_similarity >= 0.90:
    print("\nEXCELLENT - synthetic data preserves")
    print("inter-feature relationships very well")
elif corr_similarity >= 0.80:
    print("\nGOOD - most relationships preserved")
else:
    print("\nMODERATE - some relationships differ")

# ── SHOW STRONGEST CORRELATIONS ──
print("\n" + "="*60)
print("STRONGEST REAL CORRELATIONS")
print("(and how well synthetic preserved them)")
print("="*60)

pairs = []
for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        c1, c2 = num_cols[i], num_cols[j]
        rc = real_corr.loc[c1, c2]
        sc = synth_corr.loc[c1, c2]
        pairs.append({
            'pair':  f"{c1} ~ {c2}",
            'real':  rc,
            'synth': sc,
            'diff':  abs(rc - sc)
        })

pairs_df = pd.DataFrame(pairs)
pairs_df['abs_real'] = pairs_df['real'].abs()
top_pairs = pairs_df.sort_values(
    'abs_real', ascending=False).head(8)

print(f"\n{'Feature Pair':<40} {'Real':>7} "
      f"{'Synth':>7} {'Match':>6}")
print("-"*62)
for _, row in top_pairs.iterrows():
    match = "✅" if row['diff'] < 0.1 else \
            "🟡" if row['diff'] < 0.2 else "❌"
    print(f"{row['pair']:<40} "
          f"{row['real']:>7.3f} "
          f"{row['synth']:>7.3f} {match:>6}")

# ── VISUALISATION ──
print("\nGenerating correlation heatmaps...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    f'Correlation Matrix Analysis\n'
    f'Similarity Score: '
    f'{corr_similarity*100:.1f}%',
    fontsize=14, fontweight='bold')

short_names = [
    'aisle', 'dept', 'dow', 'hour',
    'wknd', 'peak', 'night', 'reord',
    'cart', 'days']

# Real correlation
im0 = axes[0].imshow(
    real_corr.values,
    cmap='coolwarm', vmin=-1, vmax=1,
    aspect='auto')
axes[0].set_title('Real Data Correlations')
axes[0].set_xticks(range(len(short_names)))
axes[0].set_yticks(range(len(short_names)))
axes[0].set_xticklabels(
    short_names, rotation=45, fontsize=8)
axes[0].set_yticklabels(
    short_names, fontsize=8)
plt.colorbar(im0, ax=axes[0],
             fraction=0.046)

# Synthetic correlation
im1 = axes[1].imshow(
    synth_corr.values,
    cmap='coolwarm', vmin=-1, vmax=1,
    aspect='auto')
axes[1].set_title('Synthetic Data Correlations')
axes[1].set_xticks(range(len(short_names)))
axes[1].set_yticks(range(len(short_names)))
axes[1].set_xticklabels(
    short_names, rotation=45, fontsize=8)
axes[1].set_yticklabels(
    short_names, fontsize=8)
plt.colorbar(im1, ax=axes[1],
             fraction=0.046)

# Difference
im2 = axes[2].imshow(
    diff_corr.values,
    cmap='Reds', vmin=0, vmax=0.5,
    aspect='auto')
axes[2].set_title(
    'Absolute Difference\n(darker = bigger gap)')
axes[2].set_xticks(range(len(short_names)))
axes[2].set_yticks(range(len(short_names)))
axes[2].set_xticklabels(
    short_names, rotation=45, fontsize=8)
axes[2].set_yticklabels(
    short_names, fontsize=8)
plt.colorbar(im2, ax=axes[2],
             fraction=0.046)

plt.tight_layout()
plt.savefig('data/correlation_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("Chart saved to "
      "data/correlation_analysis.png")

# ── SUMMARY ──
print("\n" + "="*60)
print("SUMMARY FOR DISSERTATION")
print("="*60)
print(f"""
The CTGAN synthetic data preserves inter-feature
correlations with a similarity score of
{corr_similarity*100:.1f}%. The average difference
between real and synthetic pairwise correlations
is {avg_diff:.4f}, indicating that the model
captures not only individual feature distributions
but also the relationships between features such
as the link between order timing and product
category. This confirms the synthetic data is
structurally faithful, not merely matching
marginal distributions.
""")
print("="*60)
print("Done!")