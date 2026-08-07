import pandas as pd
import matplotlib.pyplot as plt

# Load the data
print("Loading data...")
orders = pd.read_csv('data/orders.csv')
products = pd.read_csv('data/products.csv')
aisles = pd.read_csv('data/aisles.csv')
order_products = pd.read_csv('data/order_products__prior.csv')

print("Data loaded successfully!")
print("\n--- ORDERS ---")
print(orders.head())
print("Shape:", orders.shape)

print("\n--- PRODUCTS ---")
print(products.head())
print("Shape:", products.shape)

print("\n--- AISLES ---")
print(aisles.head())
print("Shape:", aisles.shape)

# Order patterns
print("\n--- ORDER PATTERNS ---")
print("Orders by day of week:")
print(orders['order_dow'].value_counts().sort_index())

print("\nOrders by hour of day:")
print(orders['order_hour_of_day'].value_counts().sort_index())

# Plot order patterns
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

orders['order_dow'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Orders by Day of Week')
axes[0].set_xlabel('Day (0=Sunday)')
axes[0].set_ylabel('Number of Orders')

orders['order_hour_of_day'].value_counts().sort_index().plot(
    kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title('Orders by Hour of Day')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Number of Orders')

plt.tight_layout()
plt.savefig('data/order_patterns.png')
plt.show()

print("\nDone! Chart saved to data/order_patterns.png")