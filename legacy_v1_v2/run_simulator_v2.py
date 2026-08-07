import pandas as pd
import numpy as np
import gymnasium as gym
import rware
import matplotlib.pyplot as plt


def run_simulation(orders_df, label=""):
    """
    Run RWARE simulation with given order stream.
    Measures fleet performance based on rewards received.
    """
    print(f"\nRunning simulation: {label}")

    # Setup environment
    env = gym.make("rware-tiny-2ag-v2")
    obs, info = env.reset()

    # Map aisle_id to RWARE shelf range
    n_shelves = 8
    orders_df = orders_df.copy()
    orders_df['shelf_id'] = orders_df['aisle_id'] % n_shelves

    # Get unique orders - limit to 500 for speed
    unique_orders = orders_df.groupby('order_id').agg(
        n_items=('shelf_id', 'count'),
        hour=('order_hour_of_day', 'first'),
        dow=('order_dow', 'first')
    ).reset_index().head(500)

    print(f"Processing {len(unique_orders)} unique orders...")
    print(f"Average items per order: "
          f"{unique_orders['n_items'].mean():.2f}")

    # Metrics
    total_steps        = 0
    total_rewards      = 0
    successful_picks   = 0
    steps_per_order    = []
    rewards_per_order  = []

    for idx, order in unique_orders.iterrows():
        order_steps   = 0
        order_rewards = 0
        n_items       = order['n_items']

        # Run steps proportional to order size
        max_steps = min(200 * n_items, 1000)

        for step in range(max_steps):
            # Sample random actions for each agent
            actions = env.action_space.sample()
            obs, rewards, terminated, truncated, info = \
                env.step(actions)

            # Count reward
            if isinstance(rewards, (list, tuple)):
                step_reward = sum(rewards)
            else:
                step_reward = float(rewards)

            order_rewards += step_reward
            total_rewards += step_reward

            if step_reward > 0:
                successful_picks += 1

            order_steps   += 1
            total_steps   += 1

            if terminated or truncated:
                obs, info = env.reset()
                break

        steps_per_order.append(order_steps)
        rewards_per_order.append(order_rewards)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}"
                  f"/{len(unique_orders)} orders...")

    env.close()

    # Calculate metrics
    avg_steps          = np.mean(steps_per_order)
    avg_reward         = np.mean(rewards_per_order)
    throughput         = successful_picks / total_steps \
                         if total_steps > 0 else 0

    print(f"\n--- Results: {label} ---")
    print(f"  Total steps:         {total_steps}")
    print(f"  Total rewards:       {total_rewards:.2f}")
    print(f"  Successful picks:    {successful_picks}")
    print(f"  Avg steps/order:     {avg_steps:.2f}")
    print(f"  Avg reward/order:    {avg_reward:.4f}")
    print(f"  Throughput:          {throughput:.6f}")

    return {
        'label':            label,
        'total_steps':      total_steps,
        'total_rewards':    total_rewards,
        'successful_picks': successful_picks,
        'avg_steps':        avg_steps,
        'avg_reward':       avg_reward,
        'throughput':       throughput,
        'steps_per_order':  steps_per_order,
        'rewards_per_order': rewards_per_order
    }


# ── MAIN ──
print("Loading datasets...")
real_df = pd.read_csv('data/clean_orders.csv').sample(
    n=50000, random_state=42)
baseline_df  = pd.read_csv('data/baseline_orders.csv')
synthetic_df = pd.read_csv('data/synthetic_orders.csv')

# Run all three simulations
real_r = run_simulation(real_df,
                        label="Real Orders")
base_r = run_simulation(baseline_df,
                        label="Shuffling Baseline")
synt_r = run_simulation(synthetic_df,
                        label="CTGAN Synthetic")

# ── COMPARISON TABLE ──
print("\n" + "="*60)
print("FLEET PERFORMANCE COMPARISON")
print("="*60)
print(f"{'Metric':<30} {'Real':>8} "
      f"{'Baseline':>10} {'CTGAN':>8}")
print("-"*60)

metrics = [
    ('Total Steps',       'total_steps',      '{:>8.0f}'),
    ('Total Rewards',     'total_rewards',    '{:>8.2f}'),
    ('Successful Picks',  'successful_picks', '{:>8.0f}'),
    ('Avg Steps/Order',   'avg_steps',        '{:>8.2f}'),
    ('Avg Reward/Order',  'avg_reward',       '{:>8.4f}'),
    ('Throughput',        'throughput',       '{:>8.6f}'),
]

for name, key, fmt in metrics:
    rv = fmt.format(real_r[key])
    bv = fmt.format(base_r[key])
    sv = fmt.format(synt_r[key])
    print(f"{name:<30} {rv:>8} {bv:>10} {sv:>8}")

print("="*60)

# ── PLOTS ──
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Warehouse Fleet Performance: '
             'Real vs Baseline vs CTGAN', fontsize=13)

labels  = ['Real', 'Baseline', 'CTGAN']
colors  = ['steelblue', 'orange', 'green']
results = [real_r, base_r, synt_r]

plot_metrics = [
    ('total_steps',      'Total Steps',       'Steps'),
    ('total_rewards',    'Total Rewards',     'Reward'),
    ('successful_picks', 'Successful Picks',  'Count'),
    ('avg_steps',        'Avg Steps/Order',   'Steps'),
    ('avg_reward',       'Avg Reward/Order',  'Reward'),
    ('throughput',       'Throughput',        'Picks/Step'),
]

for i, (key, title, ylabel) in enumerate(plot_metrics):
    ax = axes[i // 3][i % 3]
    ax.bar(labels,
           [r[key] for r in results],
           color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

plt.tight_layout()
plt.savefig('data/fleet_performance_v2.png')
plt.show()

# ── DISTRIBUTION PLOTS ──
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('Steps per Order Distribution', fontsize=12)

axes2[0].hist(real_r['steps_per_order'],
              bins=30, alpha=0.5,
              color='steelblue', label='Real')
axes2[0].hist(base_r['steps_per_order'],
              bins=30, alpha=0.5,
              color='orange', label='Baseline')
axes2[0].hist(synt_r['steps_per_order'],
              bins=30, alpha=0.5,
              color='green', label='CTGAN')
axes2[0].set_title('Steps per Order')
axes2[0].set_xlabel('Steps')
axes2[0].legend()

axes2[1].hist(real_r['rewards_per_order'],
              bins=30, alpha=0.5,
              color='steelblue', label='Real')
axes2[1].hist(base_r['rewards_per_order'],
              bins=30, alpha=0.5,
              color='orange', label='Baseline')
axes2[1].hist(synt_r['rewards_per_order'],
              bins=30, alpha=0.5,
              color='green', label='CTGAN')
axes2[1].set_title('Rewards per Order')
axes2[1].set_xlabel('Reward')
axes2[1].legend()

plt.tight_layout()
plt.savefig('data/distribution_comparison.png')
plt.show()

print("\nAll charts saved to data folder")
print("Project complete!")