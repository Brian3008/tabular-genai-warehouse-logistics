import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings('ignore')

print("Loading enriched dataset...")
df = pd.read_csv('data/clean_orders_v2.csv')

# Sample 200k rows - more data = better CTGAN learning
print("Sampling 200,000 rows for training...")
sample = df.sample(n=200000, random_state=42)

print("Sample shape:", sample.shape)
print("\nColumn types:")
print(sample.dtypes)

# ── METADATA ──
print("\nDefining metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(sample)

# Categorical columns
cat_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'time_of_day',
    'is_reorder', 'order_frequency',
    'is_early_in_cart', 'aisle_popularity'
]

# Numerical columns
num_cols = ['days_since_prior_order']

for col in cat_cols:
    metadata.update_column(
        column_name=col, sdtype='categorical')

for col in num_cols:
    metadata.update_column(
        column_name=col, sdtype='numerical')

# order_id is just an id
metadata.update_column(
    column_name='order_id', sdtype='id')

print("Metadata defined!")

# ── TRAIN CTGAN ──
print("\nTraining CTGAN v2 with 300 epochs...")
print("This will take 20-30 minutes, please wait...")
print("You will see generator and discriminator")
print("loss values updating every epoch.")
print("Loss values stabilising = good training!\n")

model = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    verbose=True
)

model.fit(sample)

print("\nCTGAN v2 training complete!")

# Save model
model.save('data/ctgan_model_v2.pkl')
print("Model saved to data/ctgan_model_v2.pkl")

# ── GENERATE SYNTHETIC DATA ──
print("\nGenerating 50,000 synthetic orders...")
synthetic = model.sample(num_rows=50000)

print("Synthetic data shape:", synthetic.shape)
print("\nSynthetic sample:")
print(synthetic.head(10))

# ── COMPARE DISTRIBUTIONS ──
print("\n--- REAL vs SYNTHETIC COMPARISON ---")

compare_cols = [
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'aisle_popularity', 'time_of_day',
    'order_frequency'
]

for col in compare_cols:
    real_mode = sample[col].mode()[0]
    synt_mode = synthetic[col].mode()[0]
    real_mean = sample[col].apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    ).mean()
    synt_mean = synthetic[col].apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    ).mean()
    print(f"\n{col}:")
    print(f"  Real mode: {real_mode} | "
          f"Synthetic mode: {synt_mode}")
    if not pd.isna(real_mean):
        print(f"  Real mean: {real_mean:.3f} | "
              f"Synthetic mean: {synt_mean:.3f}")

# Key distribution comparisons
print("\n--- WEEKEND DISTRIBUTION ---")
print("Real:     ",
      sample['is_weekend'].value_counts(
          normalize=True).round(3).to_dict())
print("Synthetic:",
      synthetic['is_weekend'].value_counts(
          normalize=True).round(3).to_dict())

print("\n--- PEAK HOUR DISTRIBUTION ---")
print("Real:     ",
      sample['is_peak_hour'].value_counts(
          normalize=True).round(3).to_dict())
print("Synthetic:",
      synthetic['is_peak_hour'].value_counts(
          normalize=True).round(3).to_dict())

print("\n--- REORDER DISTRIBUTION ---")
print("Real:     ",
      sample['is_reorder'].value_counts(
          normalize=True).round(3).to_dict())
print("Synthetic:",
      synthetic['is_reorder'].value_counts(
          normalize=True).round(3).to_dict())

print("\n--- TIME OF DAY DISTRIBUTION ---")
print("Real:     ",
      sample['time_of_day'].value_counts(
          normalize=True).round(3).to_dict())
print("Synthetic:",
      synthetic['time_of_day'].value_counts(
          normalize=True).round(3).to_dict())

# Save synthetic data
synthetic.to_csv(
    'data/synthetic_orders_v2.csv', index=False)
print("\nSaved synthetic_orders_v2.csv")

# ── GENERATE SHUFFLING BASELINE V2 ──
print("\nGenerating shuffling baseline v2...")
baseline_v2 = sample.copy().head(50000)

for col in cat_cols + num_cols:
    baseline_v2[col] = sample[col].sample(
        frac=1, random_state=42
    ).values[:50000]

baseline_v2.to_csv(
    'data/baseline_orders_v2.csv', index=False)
print("Saved baseline_orders_v2.csv")

print("\nAll done!")
print("\nFiles saved:")
print("  data/ctgan_model_v2.pkl")
print("  data/synthetic_orders_v2.csv")
print("  data/baseline_orders_v2.csv")