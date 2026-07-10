import pandas as pd

print("Loading seasonal datasets...")
normal    = pd.read_csv('data/normal_orders.csv')
christmas = pd.read_csv('data/christmas_orders.csv')
blackfri  = pd.read_csv('data/blackfriday_orders.csv')

print(f"\nNormal orders shape:      {normal.shape}")
print(f"Christmas orders shape:   {christmas.shape}")
print(f"Black Friday orders shape:{blackfri.shape}")

print("\n--- SCENARIO COMPARISON ---")
print(f"{'Metric':<25} {'Normal':>10} "
      f"{'Christmas':>12} {'BlackFriday':>12}")
print("-"*62)

cols = [
    ('Weekend ratio',    'is_weekend'),
    ('Peak hour ratio',  'is_peak_hour'),
    ('Reorder ratio',    'is_reorder'),
    ('Avg aisle',        'aisle_id'),
    ('Avg hour',         'order_hour_of_day'),
    ('Avg day',          'order_dow'),
]

for name, col in cols:
    nv = normal[col].astype(float).mean()
    cv = christmas[col].astype(float).mean()
    bv = blackfri[col].astype(float).mean()
    print(f"{name:<25} {nv:>10.3f} "
          f"{cv:>12.3f} {bv:>12.3f}")

print("\nDone!")