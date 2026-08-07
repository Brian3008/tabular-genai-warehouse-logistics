import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import matplotlib.pyplot as plt
import gymnasium as gym
import rware
import warnings
warnings.filterwarnings('ignore')

# ── LOAD TRAINED MODEL ──
print("Loading trained CTGAN v2 model...")
model = CTGANSynthesizer.load('data/ctgan_final.pkl')
print("Model loaded successfully!")

# ── DEFINE SEASONAL CONDITIONS ──
# These reflect real behavioural differences
# between normal days and seasonal peak events

NUM_ROWS = 10000

print("\nDefining seasonal conditions...")

# Normal weekday conditions
# Average day, mixed hours, moderate reorder rate
normal_conditions = Condition(
    num_rows=NUM_ROWS,
    column_values={
        'is_weekend':     0,
        'is_peak_hour':   1,
        'is_night':       0,
        'time_of_day':    'morning',
        'is_reorder':     1,
        'order_frequency': 'monthly',
        'is_early_in_cart': 0,
        'aisle_popularity': 'medium'
    }
)

# Christmas conditions
# Weekend, high reorder (familiar products),
# morning peak, weekly ordering, popular aisles
christmas_conditions = Condition(
    num_rows=NUM_ROWS,
    column_values={
        'is_weekend':     1,
        'is_peak_hour':   1,
        'is_night':       0,
        'time_of_day':    'morning',
        'is_reorder':     1,
        'order_frequency': 'weekly',
        'is_early_in_cart': 1,
        'aisle_popularity': 'high'
    }
)

# Black Friday conditions
# Weekday, peak hours, new products (not reorders),
# high volume, popular aisles
blackfriday_conditions = Condition(
    num_rows=NUM_ROWS,
    column_values={
        'is_weekend':     0,
        'is_peak_hour':   1,
        'is_night':       0,
        'time_of_day':    'afternoon',
        'is_reorder':     0,
        'order_frequency': 'weekly',
        'is_early_in_cart': 1,
        'aisle_popularity': 'high'
    }
)

# ── GENERATE CONDITIONED SYNTHETIC DATA ──
print("\nGenerating conditioned synthetic orders...")

print("  Generating Normal weekday orders...")
normal_orders = model.sample_from_conditions(
    conditions=[normal_conditions]
)
normal_orders['scenario'] = 'Normal'
print(f"  Normal orders shape: {normal_orders.shape}")

print("  Generating Christmas orders...")
christmas_orders = model.sample_from_conditions(
    conditions=[christmas_conditions]
)
christmas_orders['scenario'] = 'Christmas'
print(f"  Christmas orders shape: "
      f"{christmas_orders.shape}")

print("  Generating Black Friday orders...")
blackfriday_orders = model.sample_from_conditions(
    conditions=[blackfriday_conditions]
)
blackfriday_orders['scenario'] = 'Black Friday'
print(f"  Black Friday orders shape: "
      f"{blackfriday_orders.shape}")

# ── COMPARE DISTRIBUTIONS ──
print("\n--- SCENARIO COMPARISON ---")
print(f"\n{'Metric':<25} {'Normal':>10} "
      f"{'Christmas':>12} {'BlackFriday':>12}")
print("-" * 62)

metrics = [
    ('Weekend ratio',
     'is_weekend', 'mean'),
    ('Peak hour ratio',
     'is_peak_hour', 'mean'),
    ('Reorder ratio',
     'is_reorder', 'mean'),
    ('Avg aisle_id',
     'aisle_id', 'mean'),
    ('Avg hour',
     'order_hour_of_day', 'mean'),
    ('Avg day',
     'order_dow', 'mean'),
]

for name, col, agg in metrics:
    if agg == 'mean':
        nv = normal_orders[col].mean()
        cv = christmas_orders[col].mean()
        bv = blackfriday_orders[col].mean()
        print(f"{name:<25} {nv:>10.3f} "
              f"{cv:>12.3f} {bv:>12.3f}")

# ── SAVE ALL THREE ──
print("\nSaving conditioned datasets...")
normal_orders.to_csv(
    'data/normal_orders.csv', index=False)
christmas_orders.to_csv(
    'data/christmas_orders.csv', index=False)
blackfriday_orders.to_csv(
    'data/blackfriday_orders.csv', index=False)
print("All three datasets saved!")

# ── RUN SIMULATOR ON EACH SCENARIO ──
print("\nRunning simulator on each scenario...")

def run_sim(orders_df, label, n_orders=200):
    env = gym.make("rware-small-4ag-v2")
    obs, info = env.reset()

    n_shelves = 32
    orders_df = orders_df.copy()
    orders_df['shelf_id'] = \
        orders_df['aisle_id'] % n_shelves

    unique_orders = orders_df.groupby(
        'order_id').agg(
        n_items=('shelf_id', 'count'),
        shelves=('shelf_id', list)
    ).reset_index().head(n_orders)

    total_steps   = 0
    total_rewards = 0
    total_picks   = 0
    steps_list    = []

    for _, order in unique_orders.iterrows():
        order_steps = 0
        max_steps   = min(
            150 * order['n_items'], 800)

        for step in range(max_steps):
            actions = env.action_space.sample()
            obs, rewards, term, trunc, info = \
                env.step(actions)

            if isinstance(rewards, (list, tuple)):
                r = sum(rewards)
            else:
                r = float(rewards)

            total_rewards += r
            total_steps   += 1
            order_steps   += 1

            if r > 0:
                total_picks += 1

            if term or trunc:
                obs, info = env.reset()
                break

        steps_list.append(order_steps)

    env.close()

    throughput = (total_picks / total_steps
                  if total_steps > 0 else 0)
    avg_steps  = np.mean(steps_list)

    print(f"\n  {label}:")
    print(f"    Orders processed: "
          f"{len(unique_orders)}")
    print(f"    Total steps:      {total_steps}")
    print(f"    Successful picks: {total_picks}")
    print(f"    Avg steps/order:  {avg_steps:.2f}")
    print(f"    Throughput:       {throughput:.6f}")

    return {
        'label':        label,
        'total_steps':  total_steps,
        'total_picks':  total_picks,
        'avg_steps':    avg_steps,
        'throughput':   throughput,
        'total_rewards': total_rewards
    }

print("\nThis will take several minutes...")
r_normal = run_sim(
    normal_orders,     "Normal Weekday")
r_xmas   = run_sim(
    christmas_orders,  "Christmas")
r_bf     = run_sim(
    blackfriday_orders, "Black Friday")

# ── FINAL COMPARISON TABLE ──
print("\n" + "="*65)
print("SEASONAL FLEET PERFORMANCE COMPARISON")
print("="*65)
print(f"{'Metric':<25} {'Normal':>10} "
      f"{'Christmas':>12} {'BlackFriday':>12}")
print("-"*65)

all_results = [r_normal, r_xmas, r_bf]
row_metrics = [
    ('Total Steps',     'total_steps',   '{:>10.0f}'),
    ('Successful Picks','total_picks',   '{:>10.0f}'),
    ('Avg Steps/Order', 'avg_steps',     '{:>10.2f}'),
    ('Throughput',      'throughput',    '{:>10.6f}'),
    ('Total Rewards',   'total_rewards', '{:>10.2f}'),
]

for name, key, fmt in row_metrics:
    vals = [fmt.format(r[key]) for r in all_results]
    print(f"{name:<25} {vals[0]:>10} "
          f"{vals[1]:>12} {vals[2]:>12}")
print("="*65)

# ── PLOT RESULTS ──
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    'Seasonal Demand: Fleet Performance Comparison',
    fontsize=14)

labels  = ['Normal', 'Christmas', 'Black Friday']
colours = ['steelblue', 'red', 'orange']

plot_items = [
    ('total_steps',   'Total Steps',      'Steps'),
    ('total_picks',   'Successful Picks', 'Count'),
    ('avg_steps',     'Avg Steps/Order',  'Steps'),
    ('throughput',    'Throughput',
     'Picks per Step'),
]

for i, (key, title, ylabel) in enumerate(plot_items):
    ax = axes[i // 2][i % 2]
    vals = [r[key] for r in all_results]
    bars = ax.bar(labels, vals, color=colours)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.01,
            f'{val:.4f}' if val < 1
            else f'{val:.0f}',
            ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('data/seasonal_comparison.png')
plt.show()

print("\nSeasonal comparison chart saved!")
print("\nConditional CTGAN complete!")
print("\nFiles saved:")
print("  data/normal_orders.csv")
print("  data/christmas_orders.csv")
print("  data/blackfriday_orders.csv")
print("  data/seasonal_comparison.png")