import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRID = 10
N_SHELVES = GRID * GRID
N_ROBOTS = 4
MAX_ORDERS = 200

print("="*60)
print("FLEET STRATEGY COMPARISON")
print("Do different fleet strategies suit")
print("different demand periods?")
print("="*60)

def shelf_cell(shelf):
    s = shelf % N_SHELVES
    return (s // GRID, s % GRID)

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def build_tasks(df):
    df = df.copy()
    df['shelf'] = df['aisle_id'] % N_SHELVES
    grouped = df.groupby('order_id')['shelf'].apply(
        list).tolist()[:MAX_ORDERS]
    tasks = []
    for order in grouped:
        for s in order:
            tasks.append(s)
    return tasks

# ══════════════════════════════════════════
# THREE FLEET STRATEGIES
# Each returns total steps to clear all tasks
# ══════════════════════════════════════════

def strat_nearest(tasks):
    """Nearest-task: each free robot takes the
    task whose shelf is closest to it."""
    robots = [[(i*2) % GRID, (i*3) % GRID]
              for i in range(N_ROBOTS)]
    remaining = [shelf_cell(t) for t in tasks]
    total_steps = 0
    # assign one task to each robot at a time
    robot_target = [None] * N_ROBOTS
    while remaining or any(robot_target):
        for i in range(N_ROBOTS):
            if robot_target[i] is None and remaining:
                # pick nearest remaining task
                pos = tuple(robots[i])
                idx = min(range(len(remaining)),
                          key=lambda k: manhattan(
                              pos, remaining[k]))
                robot_target[i] = remaining.pop(idx)
        # step all robots one cell
        for i in range(N_ROBOTS):
            if robot_target[i] is not None:
                tr, tc = robot_target[i]
                if robots[i][1] < tc:
                    robots[i][1] += 1
                elif robots[i][1] > tc:
                    robots[i][1] -= 1
                elif robots[i][0] < tr:
                    robots[i][0] += 1
                elif robots[i][0] > tr:
                    robots[i][0] -= 1
                total_steps += 1
                if tuple(robots[i]) == robot_target[i]:
                    robot_target[i] = None
    return total_steps

def strat_roundrobin(tasks):
    """Round-robin: tasks handed out in order,
    ignoring distance."""
    robots = [[(i*2) % GRID, (i*3) % GRID]
              for i in range(N_ROBOTS)]
    queue = [shelf_cell(t) for t in tasks]
    qi = 0
    robot_target = [None] * N_ROBOTS
    total_steps = 0
    while qi < len(queue) or any(robot_target):
        for i in range(N_ROBOTS):
            if robot_target[i] is None and qi < len(queue):
                robot_target[i] = queue[qi]
                qi += 1
        for i in range(N_ROBOTS):
            if robot_target[i] is not None:
                tr, tc = robot_target[i]
                if robots[i][1] < tc:
                    robots[i][1] += 1
                elif robots[i][1] > tc:
                    robots[i][1] -= 1
                elif robots[i][0] < tr:
                    robots[i][0] += 1
                elif robots[i][0] > tr:
                    robots[i][0] -= 1
                total_steps += 1
                if tuple(robots[i]) == robot_target[i]:
                    robot_target[i] = None
    return total_steps

def strat_zoned(tasks):
    """Zoned: grid split into regions, each robot
    owns a region and only serves tasks there."""
    robots = [[(i*2) % GRID, (i*3) % GRID]
              for i in range(N_ROBOTS)]
    # assign each task to the robot whose zone
    # (vertical strips) it falls into
    zones = [[] for _ in range(N_ROBOTS)]
    for t in tasks:
        r, c = shelf_cell(t)
        zone = min(c // (GRID // N_ROBOTS),
                   N_ROBOTS - 1)
        zones[zone].append((r, c))
    robot_target = [None] * N_ROBOTS
    total_steps = 0
    while any(zones[i] for i in range(N_ROBOTS)) \
            or any(robot_target):
        for i in range(N_ROBOTS):
            if robot_target[i] is None and zones[i]:
                robot_target[i] = zones[i].pop(0)
        for i in range(N_ROBOTS):
            if robot_target[i] is not None:
                tr, tc = robot_target[i]
                if robots[i][1] < tc:
                    robots[i][1] += 1
                elif robots[i][1] > tc:
                    robots[i][1] -= 1
                elif robots[i][0] < tr:
                    robots[i][0] += 1
                elif robots[i][0] > tr:
                    robots[i][0] -= 1
                total_steps += 1
                if tuple(robots[i]) == robot_target[i]:
                    robot_target[i] = None
    return total_steps

# ══════════════════════════════════════════
# RUN ALL STRATEGIES ON ALL SCENARIOS
# ══════════════════════════════════════════
scenarios = {
    'Normal':       pd.read_csv('data/normal_orders.csv'),
    'Christmas':    pd.read_csv('data/christmas_orders.csv'),
    'Black Friday': pd.read_csv('data/blackfriday_orders.csv'),
}

strategies = {
    'Nearest-task': strat_nearest,
    'Round-robin':  strat_roundrobin,
    'Zoned':        strat_zoned,
}

print("\nRunning each strategy on each scenario...")
print("(steps = total robot moves to clear all")
print(" orders; lower = more efficient)\n")

results = {}  # scenario -> {strategy: steps}
for sname, sdf in scenarios.items():
    tasks = build_tasks(sdf)
    results[sname] = {}
    for stratname, stratfn in strategies.items():
        steps = stratfn(tasks)
        results[sname][stratname] = steps

# ── TABLE ──
print(f"{'Scenario':<15}", end='')
for st in strategies:
    print(f"{st:>14}", end='')
print(f"{'Best':>14}")
print("-"*72)

for sname in scenarios:
    print(f"{sname:<15}", end='')
    best_strat = min(results[sname],
                     key=results[sname].get)
    for st in strategies:
        val = results[sname][st]
        mark = "*" if st == best_strat else " "
        print(f"{val:>13}{mark}", end='')
    print(f"{best_strat:>14}")

print("\n(* = best strategy for that scenario)")

# ══════════════════════════════════════════
# KEY QUESTION: does best strategy change?
# ══════════════════════════════════════════
print("\n" + "="*60)
print("DOES THE BEST STRATEGY CHANGE BY PERIOD?")
print("="*60)

best_by_scenario = {
    s: min(results[s], key=results[s].get)
    for s in scenarios}

print()
for s, best in best_by_scenario.items():
    print(f"  {s:<15} -> best: {best}")

unique_best = set(best_by_scenario.values())
print()
if len(unique_best) > 1:
    print("  RESULT: The optimal fleet strategy")
    print("  DOES change depending on the demand")
    print("  period. This directly supports adapting")
    print("  fleet management to forecasted demand.")
else:
    print(f"  RESULT: {list(unique_best)[0]} is best")
    print("  across all scenarios, but the margin")
    print("  of advantage differs by period (see")
    print("  percentages below).")

# Margins
print("\nEfficiency gap between best and worst")
print("strategy in each scenario:")
for s in scenarios:
    vals = results[s]
    best_v = min(vals.values())
    worst_v = max(vals.values())
    gap = (worst_v - best_v) / worst_v * 100
    print(f"  {s:<15} {gap:.1f}% "
          f"(choosing the right strategy matters "
          f"{'a lot' if gap > 15 else 'moderately' if gap > 5 else 'little'})")

# ══════════════════════════════════════════
# CHART
# ══════════════════════════════════════════
print("\nGenerating chart...")
fig, ax = plt.subplots(figsize=(12, 6))

snames = list(scenarios.keys())
stratnames = list(strategies.keys())
x = np.arange(len(snames))
w = 0.25
colours = ['steelblue', 'orange', 'green']

for i, st in enumerate(stratnames):
    vals = [results[s][st] for s in snames]
    ax.bar(x + (i-1)*w, vals, w,
           label=st, color=colours[i])

ax.set_title(
    'Fleet Strategy Efficiency by Demand Scenario\n'
    '(lower steps = more efficient)',
    fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(snames)
ax.set_ylabel('Total steps to clear all orders')
ax.legend(title='Strategy')

plt.tight_layout()
plt.savefig('results/fleet_strategy_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved results/fleet_strategy_comparison.png")

# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY FOR SUPERVISOR")
print("="*60)
print(f"""
Three fleet strategies (nearest-task, round-robin,
zoned) were tested across the three demand
scenarios generated by the conditional model.

Best strategy per scenario:""")
for s, best in best_by_scenario.items():
    print(f"  {s}: {best}")
print(f"""
This addresses the question of whether fleet
management style should adapt to the demand
period. The conditional generator provides the
demand forecasts; this analysis shows how the
choice of strategy interacts with them.
""")
print("="*60)
print("Done!")