import pandas as pd
import numpy as np
import random
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*60)
print("REGENERATING SEASONAL DATA")
print("Using best model: ctgan_final.pkl")
print("="*60)

# Load best model
print("\nLoading best model...")
model = CTGANSynthesizer.load('data/ctgan_final.pkl')
print("Model loaded!")

NUM_ROWS = 10000

# ── NORMAL WEEKDAY ──
print("\nGenerating Normal weekday orders...")
normal_cond = Condition(
    num_rows=NUM_ROWS,
    column_values={
        'is_weekend':       0,
        'is_peak_hour':     1,
        'is_night':         0,
        'time_of_day':      'morning',
        'is_reorder':       1,
        'order_frequency':  'monthly',
        'aisle_popularity': 'medium'
    }
)
normal = model.sample_from_conditions(
    conditions=[normal_cond])
normal['scenario'] = 'Normal'
normal.to_csv('data/normal_orders.csv', index=False)
print(f"  Normal: {normal.shape}")

# ── CHRISTMAS ──
print("Generating Christmas orders...")
christmas_cond = Condition(
    num_rows=NUM_ROWS,
    column_values={
        'is_weekend':       1,
        'is_peak_hour':     1,
        'is_night':         0,
        'time_of_day':      'morning',
        'is_reorder':       1,
        'order_frequency':  'weekly',
        'aisle_popularity': 'high'
    }
)
christmas = model.sample_from_conditions(
    conditions=[christmas_cond])
christmas['scenario'] = 'Christmas'
christmas.to_csv(
    'data/christmas_orders.csv', index=False)
print(f"  Christmas: {christmas.shape}")

# ── BLACK FRIDAY ──
print("Generating Black Friday orders...")
blackfriday_cond = Condition(
    num_rows=NUM_ROWS,
    column_values={
        'is_weekend':       0,
        'is_peak_hour':     1,
        'is_night':         0,
        'time_of_day':      'afternoon',
        'is_reorder':       0,
        'order_frequency':  'weekly',
        'aisle_popularity': 'high'
    }
)
blackfriday = model.sample_from_conditions(
    conditions=[blackfriday_cond])
blackfriday['scenario'] = 'Black Friday'
blackfriday.to_csv(
    'data/blackfriday_orders.csv', index=False)
print(f"  Black Friday: {blackfriday.shape}")

print("\n" + "="*60)
print("All seasonal data regenerated!")
print("Files saved:")
print("  data/normal_orders.csv")
print("  data/christmas_orders.csv")
print("  data/blackfriday_orders.csv")
print("="*60)
print("\nDone!")