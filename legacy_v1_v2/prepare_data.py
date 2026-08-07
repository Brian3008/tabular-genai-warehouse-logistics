import pandas as pd

# Load all files
print("Loading data...")
orders = pd.read_csv('data/orders.csv')
products = pd.read_csv('data/products.csv')
aisles = pd.read_csv('data/aisles.csv')
order_products = pd.read_csv('data/order_products__prior.csv')

print("Merging data...")

# Merge order_products with products to get aisle info
order_products = order_products.merge(products[['product_id', 'aisle_id', 'department_id']],
                                       on='product_id',
                                       how='left')

# Merge with aisles to get aisle names
order_products = order_products.merge(aisles,
                                       on='aisle_id',
                                       how='left')

# Merge with orders to get timing info
merged = order_products.merge(orders[['order_id', 'user_id',
                                       'order_dow', 'order_hour_of_day',
                                       'days_since_prior_order']],
                               on='order_id',
                               how='left')

print("Merged data shape:", merged.shape)
print("\nFirst few rows:")
print(merged.head())
print("\nColumns:", merged.columns.tolist())

# Keep only the columns we need for CTGAN and simulator
clean = merged[['order_id', 'product_id', 'aisle_id',
                'department_id', 'order_dow',
                'order_hour_of_day']].copy()

# Drop any rows with missing values
clean = clean.dropna()

print("\nClean data shape:", clean.shape)
print("\nClean data sample:")
print(clean.head(10))

# Check aisle distribution
print("\nTop 10 most ordered aisles:")
print(clean['aisle_id'].value_counts().head(10))

# Save to CSV for use in next steps
clean.to_csv('data/clean_orders.csv', index=False)
print("\nSaved clean_orders.csv to data folder")
print("Done!")