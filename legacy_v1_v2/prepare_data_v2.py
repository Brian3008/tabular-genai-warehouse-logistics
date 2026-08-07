import pandas as pd
import numpy as np

print("Loading raw data...")
orders        = pd.read_csv('data/orders.csv')
products      = pd.read_csv('data/products.csv')
aisles        = pd.read_csv('data/aisles.csv')
departments   = pd.read_csv('data/departments.csv')
order_products = pd.read_csv(
    'data/order_products__prior.csv')

print("Raw data loaded successfully!")

# ── STEP 1: Merge everything ──
print("\nMerging datasets...")
df = order_products.merge(
    products[['product_id', 'aisle_id', 'department_id']],
    on='product_id', how='left')

df = df.merge(aisles,       on='aisle_id',       how='left')
df = df.merge(departments,  on='department_id',  how='left')
df = df.merge(
    orders[['order_id', 'user_id', 'order_dow',
            'order_hour_of_day',
            'days_since_prior_order',
            'order_number']],
    on='order_id', how='left')

print("Merged shape:", df.shape)

# ── STEP 2: Feature Engineering ──
print("\nEngineering features...")

# Time based features
df['is_weekend'] = df['order_dow'].apply(
    lambda x: 1 if x in [0, 1] else 0)

df['is_peak_hour'] = df['order_hour_of_day'].apply(
    lambda x: 1 if 9 <= x <= 15 else 0)

df['is_night'] = df['order_hour_of_day'].apply(
    lambda x: 1 if x < 6 or x > 21 else 0)

df['time_of_day'] = pd.cut(
    df['order_hour_of_day'],
    bins=[-1, 6, 12, 17, 21, 24],
    labels=['night', 'morning',
            'afternoon', 'evening', 'late'])

# Order behaviour features
df['is_reorder'] = df['reordered'].astype(int)

df['is_first_order'] = df['order_number'].apply(
    lambda x: 1 if x == 1 else 0)

df['days_since_prior_order'] = \
    df['days_since_prior_order'].fillna(0)

df['order_frequency'] = pd.cut(
    df['days_since_prior_order'],
    bins=[-1, 0, 7, 14, 30, 999],
    labels=['first', 'weekly',
            'biweekly', 'monthly', 'infrequent'])

# Cart position feature
df['is_early_in_cart'] = df['add_to_cart_order'].apply(
    lambda x: 1 if x <= 3 else 0)

# Aisle popularity bins
aisle_counts = df['aisle_id'].value_counts()
df['aisle_popularity'] = df['aisle_id'].map(
    aisle_counts).apply(
    lambda x: 'high'   if x > 2000000
    else 'medium' if x > 500000
    else 'low')

print("Features engineered successfully!")
print("\nNew columns added:")
new_cols = ['is_weekend', 'is_peak_hour', 'is_night',
            'time_of_day', 'is_reorder', 'is_first_order',
            'days_since_prior_order', 'order_frequency',
            'is_early_in_cart', 'aisle_popularity']
for c in new_cols:
    print(f"  - {c}")

# ── STEP 3: Select final columns for CTGAN ──
print("\nSelecting final columns...")
final_cols = [
    'order_id',
    'aisle_id',
    'department_id',
    'order_dow',
    'order_hour_of_day',
    'is_weekend',
    'is_peak_hour',
    'is_night',
    'time_of_day',
    'is_reorder',
    'days_since_prior_order',
    'order_frequency',
    'is_early_in_cart',
    'aisle_popularity'
]

clean_v2 = df[final_cols].dropna()
print("Final shape:", clean_v2.shape)
print("\nSample:")
print(clean_v2.head(10))

# ── STEP 4: Statistics ──
print("\n--- DATASET STATISTICS ---")
print(f"Total orders:       "
      f"{clean_v2['order_id'].nunique():,}")
print(f"Total rows:         "
      f"{len(clean_v2):,}")
print(f"Unique aisles:      "
      f"{clean_v2['aisle_id'].nunique()}")
print(f"Unique departments: "
      f"{clean_v2['department_id'].nunique()}")
print(f"\nWeekend orders:     "
      f"{clean_v2['is_weekend'].sum():,} "
      f"({clean_v2['is_weekend'].mean()*100:.1f}%)")
print(f"Peak hour orders:   "
      f"{clean_v2['is_peak_hour'].sum():,} "
      f"({clean_v2['is_peak_hour'].mean()*100:.1f}%)")
print(f"Reorders:           "
      f"{clean_v2['is_reorder'].sum():,} "
      f"({clean_v2['is_reorder'].mean()*100:.1f}%)")

print(f"\nTime of day distribution:")
print(clean_v2['time_of_day'].value_counts())

print(f"\nOrder frequency distribution:")
print(clean_v2['order_frequency'].value_counts())

print(f"\nAisle popularity distribution:")
print(clean_v2['aisle_popularity'].value_counts())

# ── STEP 5: Save ──
clean_v2.to_csv('data/clean_orders_v2.csv', index=False)
print("\nSaved to data/clean_orders_v2.csv")
print("Done!")