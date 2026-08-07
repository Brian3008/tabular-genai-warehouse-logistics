import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("="*60)
print("TRAINING CTGAN ON STRATIFIED 10% SAMPLE")
print("Simulating a realistic Locus-size dataset")
print("="*60)

# ── LOAD FULL DATA ──
print("\nLoading full dataset...")
df = pd.read_csv('data/clean_orders_v2.csv')
print(f"Full dataset: {len(df):,} rows")

# ── STRATIFIED 10% SAMPLE ──
print("\nCreating stratified 10% sample...")

# Build a stratification key from the most
# important features so the sample perfectly
# represents the full data
df['strat_key'] = (
    df['is_weekend'].astype(str) + '_' +
    df['is_peak_hour'].astype(str) + '_' +
    df['is_night'].astype(str) + '_' +
    df['is_reorder'].astype(str) + '_' +
    df['time_of_day'].astype(str)
)

# Remove very rare combinations that can't
# be stratified
counts = df['strat_key'].value_counts()
valid = counts[counts >= 10].index
df_valid = df[df['strat_key'].isin(valid)].copy()

# Take stratified 10%
target_size = 320000
_, sample = train_test_split(
    df_valid,
    test_size=target_size,
    stratify=df_valid['strat_key'],
    random_state=SEED
)
sample = sample.drop(columns=['strat_key'])
df = df.drop(columns=['strat_key'])

print(f"Sample size: {len(sample):,} rows "
      f"(10% of full data)")

# ── VERIFY SAMPLE IS REPRESENTATIVE ──
print("\nVerifying sample matches full data:")
print(f"{'Metric':<20}{'Full':>10}{'Sample':>10}")
print("-"*40)
for col in ['is_weekend','is_peak_hour',
            'is_night','is_reorder',
            'order_hour_of_day','order_dow']:
    fv = df[col].astype(float).mean()
    sv = sample[col].astype(float).mean()
    print(f"{col:<20}{fv:>10.3f}{sv:>10.3f}")

# ── DEFINE METADATA ──
print("\nDefining metadata...")
train_cols = [
    'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day',
    'is_weekend', 'is_peak_hour',
    'is_night', 'is_reorder',
    'is_early_in_cart', 'time_of_day',
    'order_frequency', 'aisle_popularity',
    'days_since_prior_order'
]
train_data = sample[train_cols].copy()

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)
# Set categorical columns explicitly
for col in ['aisle_id','department_id',
            'order_dow','order_hour_of_day',
            'is_weekend','is_peak_hour',
            'is_night','is_reorder',
            'is_early_in_cart','time_of_day',
            'order_frequency','aisle_popularity']:
    metadata.update_column(
        column_name=col, sdtype='categorical')
metadata.update_column(
    column_name='days_since_prior_order',
    sdtype='numerical')

print("Metadata defined!")

# ── TRAIN CTGAN ──
print("\n" + "="*60)
print("TRAINING CTGAN")
print("Config: batch=300, lr=0.00005,")
print("disc_steps=3, 300 epochs")
print("Estimated time: 4 hours")
print("="*60)

model = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=300,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=0.00005,
    discriminator_lr=0.00005,
    discriminator_steps=3,
    verbose=True,
    cuda=True
)

model.fit(train_data)
print("\nTraining complete!")

# ── SAVE MODEL ──
model.save('data/ctgan_10percent_300.pkl')
print("Saved ctgan_10percent_300.pkl")

# ── EVALUATE ──
print("\nGenerating 50,000 synthetic orders...")
synthetic = model.sample(num_rows=50000)
synthetic.to_csv(
    'data/synthetic_10percent_300.csv', index=False)

# Quality against the sample
quality = evaluate_quality(
    real_data=train_data.sample(
        n=10000, random_state=SEED),
    synthetic_data=synthetic[
        train_cols].sample(
        n=10000, random_state=SEED),
    metadata=metadata,
    verbose=True
)
score = quality.get_score()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Quality score: {score:.4f}")

# Compare key metrics
print(f"\n{'Metric':<20}{'Full-data':>10}"
      f"{'10%-model':>12}")
print("-"*42)
for col in ['is_weekend','is_peak_hour',
            'is_night','is_reorder',
            'is_early_in_cart']:
    rv = df[col].astype(float).mean()
    sv = synthetic[col].astype(float).mean()
    print(f"{col:<20}{rv:>10.3f}{sv:>12.3f}")

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"10% model quality: {score:.4f}")
print("Compare to full-data model: 0.9347")
print("\nIf similar, it confirms the method works")
print("on realistic Locus-size data volumes.")
print("Saved: data/ctgan_10percent_300.pkl")
print("       data/synthetic_10percent_300.csv")