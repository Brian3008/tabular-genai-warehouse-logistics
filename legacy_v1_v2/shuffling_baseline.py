import pandas as pd
import matplotlib.pyplot as plt

# Load clean real orders
print("Loading clean orders...")
df = pd.read_csv('data/clean_orders.csv')

# Sample same size as synthetic for fair comparison
print("Sampling 10,000 rows...")
sample = df.sample(n=10000, random_state=42)

# Shuffling baseline
# Shuffle each column independently
# This preserves individual column distributions
# but destroys correlations between columns
print("\nCreating shuffling baseline...")
baseline = sample.copy()

# Shuffle each column independently
for col in ['aisle_id', 'department_id',
            'order_dow', 'order_hour_of_day']:
    baseline[col] = sample[col].sample(
        frac=1, random_state=42).values

print("Shuffling baseline created!")
print("\nBaseline sample:")
print(baseline.head(10))

# Save baseline
baseline.to_csv('data/baseline_orders.csv', index=False)
print("\nSaved baseline_orders.csv to data folder")

# Compare stats
print("\n--- REAL DATA STATS ---")
print(sample[['order_dow',
              'order_hour_of_day',
              'aisle_id']].describe())

print("\n--- BASELINE STATS ---")
print(baseline[['order_dow',
                'order_hour_of_day',
                'aisle_id']].describe())

# Plot comparison
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Real vs Baseline vs Synthetic Orders',
             fontsize=14)

# Load synthetic for comparison
synthetic = pd.read_csv('data/synthetic_orders.csv')

cols = ['order_dow', 'order_hour_of_day', 'aisle_id']
titles = ['Day of Week', 'Hour of Day', 'Aisle ID']

for i, (col, title) in enumerate(zip(cols, titles)):
    # Real
    sample[col].value_counts().sort_index().plot(
        kind='bar', ax=axes[i][0], color='steelblue')
    axes[i][0].set_title(f'Real - {title}')
    axes[i][0].set_xlabel('')

    # Baseline
    baseline[col].value_counts().sort_index().plot(
        kind='bar', ax=axes[i][1], color='orange')
    axes[i][1].set_title(f'Baseline - {title}')
    axes[i][1].set_xlabel('')

    # Synthetic
    synthetic[col].value_counts().sort_index().plot(
        kind='bar', ax=axes[i][2], color='green')
    axes[i][2].set_title(f'CTGAN - {title}')
    axes[i][2].set_xlabel('')

plt.tight_layout()
plt.savefig('data/comparison_plots.png')
plt.show()

print("\nComparison chart saved to data/comparison_plots.png")
print("Done!")