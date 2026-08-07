import pandas as pd
import numpy as np
import gymnasium as gym
import rware
import matplotlib.pyplot as plt

def run_simulation(orders_df, n_agents=2, max_steps=500, label=""):
    """
    Run RWARE simulation with given order stream.
    Returns performance metrics.
    """
    print(f"\nRunning simulation: {label}")
    print(f"Orders to process: {len(orders_df)}")

    # Setup RWARE environment
    env = gym.make("rware-small-2ag-v2")
    obs, info = env.reset()

    # Metrics to track
    total_steps = 0
    completed_orders = 0
    agent_actions = []
    steps_per_order = []

    # Convert aisle_ids to shelf requests
    # RWARE has limited shelves so we map aisle_id to shelf index
    n_shelves = 8  # rware-small has 8 shelves
    orders_df = orders_df.copy()
    orders_df['shelf_id'] = orders_df['aisle_id'] % n_shelves

    # Get unique orders
    unique_orders = orders_df.groupby('order_id').agg(
        shelves=('shelf_id', list),
        hour=('order_hour_of_day', 'first'),
        dow=('order_dow', 'first')
    ).reset_index()

    # Limit to first 200 orders for simulation speed
    unique_orders = unique_orders.head(200)

    print(f"Processing {len(unique_orders)} unique orders...")

    for idx, order in unique_orders.iterrows():
        order_start_step = total_steps
        shelves_needed = order['shelves']

        for shelf in shelves_needed:
            steps_taken = 0

            for step in range(max_steps):
                # Simple random policy for agents
                # In real fleet management this would be
                # optimised routing
                actions = env.action_space.sample()
                obs, rewards, terminated, truncated, info = \
                    env.step(actions)

                total_steps += 1
                steps_taken += 1
                agent_actions.extend(
                    [a for a in actions]
                    if hasattr(actions, '__iter__')
                    else [actions]
                )

                if terminated or truncated:
                    obs, info = env.reset()
                    break

            steps_per_order.append(steps_taken)

        completed_orders += 1

        if completed_orders % 50 == 0:
            print(f"  Completed {completed_orders}"
                  f"/{len(unique_orders)} orders...")

    env.close()

    # Calculate metrics
    avg_steps = np.mean(steps_per_order)
    std_steps = np.std(steps_per_order)
    total_completed = completed_orders

    print(f"\nResults for {label}:")
    print(f"  Total orders completed: {total_completed}")
    print(f"  Average steps per item: {avg_steps:.2f}")
    print(f"  Std steps per item:     {std_steps:.2f}")
    print(f"  Total steps taken:      {total_steps}")

    return {
        'label': label,
        'completed_orders': total_completed,
        'avg_steps': avg_steps,
        'std_steps': std_steps,
        'total_steps': total_steps,
        'steps_per_order': steps_per_order
    }


# ── MAIN ──
print("Loading order datasets...")
real_df      = pd.read_csv('data/clean_orders.csv').sample(
                   n=50000, random_state=42)
baseline_df  = pd.read_csv('data/baseline_orders.csv')
synthetic_df = pd.read_csv('data/synthetic_orders.csv')

print("Real orders shape:",      real_df.shape)
print("Baseline orders shape:",  baseline_df.shape)
print("Synthetic orders shape:", synthetic_df.shape)

# Run simulation for all three conditions
real_results      = run_simulation(real_df,
                                   label="Real Orders")
baseline_results  = run_simulation(baseline_df,
                                   label="Shuffling Baseline")
synthetic_results = run_simulation(synthetic_df,
                                   label="CTGAN Synthetic")

# ── COMPARISON TABLE ──
print("\n" + "="*55)
print("FLEET PERFORMANCE COMPARISON")
print("="*55)
print(f"{'Metric':<30} {'Real':>7} "
      f"{'Baseline':>10} {'CTGAN':>7}")
print("-"*55)
print(f"{'Orders Completed':<30} "
      f"{real_results['completed_orders']:>7} "
      f"{baseline_results['completed_orders']:>10} "
      f"{synthetic_results['completed_orders']:>7}")
print(f"{'Avg Steps per Item':<30} "
      f"{real_results['avg_steps']:>7.2f} "
      f"{baseline_results['avg_steps']:>10.2f} "
      f"{synthetic_results['avg_steps']:>7.2f}")
print(f"{'Std Steps per Item':<30} "
      f"{real_results['std_steps']:>7.2f} "
      f"{baseline_results['std_steps']:>10.2f} "
      f"{synthetic_results['std_steps']:>7.2f}")
print(f"{'Total Steps':<30} "
      f"{real_results['total_steps']:>7} "
      f"{baseline_results['total_steps']:>10} "
      f"{synthetic_results['total_steps']:>7}")
print("="*55)

# ── PLOT RESULTS ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Warehouse Fleet Performance Comparison',
             fontsize=14)

labels = ['Real', 'Baseline', 'CTGAN']
colors = ['steelblue', 'orange', 'green']
results = [real_results, baseline_results, synthetic_results]

# Average steps per item
axes[0].bar(labels,
            [r['avg_steps'] for r in results],
            color=colors)
axes[0].set_title('Average Steps per Item')
axes[0].set_ylabel('Steps')

# Total steps
axes[1].bar(labels,
            [r['total_steps'] for r in results],
            color=colors)
axes[1].set_title('Total Steps Taken')
axes[1].set_ylabel('Steps')

# Orders completed
axes[2].bar(labels,
            [r['completed_orders'] for r in results],
            color=colors)
axes[2].set_title('Orders Completed')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('data/fleet_performance.png')
plt.show()

print("\nFleet performance chart saved to"
      " data/fleet_performance.png")
print("\nAll simulations complete!")