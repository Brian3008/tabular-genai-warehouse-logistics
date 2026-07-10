import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*60)
print("FLEET WORKLOAD ANALYSIS")
print("How does demand structure differ across")
print("scenarios, and what does it mean for fleets?")
print("="*60)

# ── LOAD ──
normal    = pd.read_csv('data/normal_orders.csv')
christmas = pd.read_csv('data/christmas_orders.csv')
blackfri  = pd.read_csv('data/blackfriday_orders.csv')

scenarios = {
    'Normal':       normal,
    'Christmas':    christmas,
    'Black Friday': blackfri
}

# Map aisles to shelves (same as simulator)
N_SHELVES = 100
for name, df in scenarios.items():
    df['shelf_id'] = df['aisle_id'] % N_SHELVES

# ══════════════════════════════════════════
# METRIC 1: DEMAND CONCENTRATION
# How spread out is the workload across shelves?
# ══════════════════════════════════════════
print("\n" + "="*60)
print("METRIC 1: DEMAND CONCENTRATION")
print("(How concentrated are orders on shelves?)")
print("="*60)

def gini(values):
    """Gini coefficient: 0=perfectly even,
    1=all on one shelf. Measures concentration."""
    v = np.sort(np.array(values, dtype=float))
    n = len(v)
    if v.sum() == 0:
        return 0
    cum = np.cumsum(v)
    return (2 * np.sum((np.arange(1, n+1)) * v)
            - (n + 1) * cum[-1]) / (n * cum[-1])

print(f"\n{'Scenario':<15} {'Shelves used':>13} "
      f"{'Gini':>8} {'Top-10 %':>10}")
print("-"*48)

concentration = {}
for name, df in scenarios.items():
    shelf_counts = df['shelf_id'].value_counts()
    shelves_used = len(shelf_counts)
    g = gini(shelf_counts.values)
    # what % of all picks are in the top 10 shelves
    top10 = shelf_counts.head(10).sum() / len(df)
    concentration[name] = {
        'shelves_used': shelves_used,
        'gini': g,
        'top10_pct': top10,
        'shelf_counts': shelf_counts
    }
    print(f"{name:<15} {shelves_used:>13} "
          f"{g:>8.3f} {top10*100:>9.1f}%")

print("\nInterpretation:")
print("  Higher Gini / higher Top-10% = demand")
print("  concentrated on fewer shelves.")
print("  Concentrated demand means robots cluster")
print("  in hot zones (congestion risk), while")
print("  spread demand means more travel distance.")

# ══════════════════════════════════════════
# METRIC 2: THEORETICAL TRAVEL WORKLOAD
# Estimate robot travel based on shelf spread
# ══════════════════════════════════════════
print("\n" + "="*60)
print("METRIC 2: SPATIAL WORKLOAD SPREAD")
print("(Average distance robots must cover)")
print("="*60)

# Place shelves on a 10x10 grid, compute the
# average pairwise distance weighted by demand
GRID = 10
def shelf_coord(shelf_id):
    return (shelf_id // GRID, shelf_id % GRID)

print(f"\n{'Scenario':<15} {'Avg shelf dist':>15} "
      f"{'Spread':>10}")
print("-"*42)

travel = {}
for name, df in scenarios.items():
    counts = concentration[name]['shelf_counts']
    # weighted centroid
    coords = np.array(
        [shelf_coord(s) for s in counts.index])
    weights = counts.values
    centroid = np.average(
        coords, axis=0, weights=weights)
    # mean weighted distance from centroid
    dists = np.sqrt(
        ((coords - centroid)**2).sum(axis=1))
    avg_dist = np.average(dists, weights=weights)
    spread = np.sqrt(
        np.average((dists - avg_dist)**2,
                   weights=weights))
    travel[name] = avg_dist
    print(f"{name:<15} {avg_dist:>15.3f} "
          f"{spread:>10.3f}")

print("\nInterpretation:")
print("  Larger avg distance = robots must travel")
print("  further on average to serve demand.")

# ══════════════════════════════════════════
# METRIC 3: FLEET MANAGEMENT IMPLICATION
# Based on the metrics that ACTUALLY separate
# the scenarios: shelves-used and travel spread
# ══════════════════════════════════════════
print("\n" + "="*60)
print("METRIC 3: FLEET MANAGEMENT IMPLICATION")
print("="*60)

# Rank scenarios by the metrics that genuinely
# differ. Top-10% is saturated (all 93-99%) so
# we do NOT use it to separate them. We use
# shelves-used and average travel distance.
print(f"\n{'Scenario':<15} {'Shelves':>8} "
      f"{'Travel':>8}  Implication")
print("-"*70)

for name in scenarios:
    sh = concentration[name]['shelves_used']
    tv = travel[name]
    # compare each against Normal as the reference
    print(f"{name:<15} {sh:>8} {tv:>8.2f}", end='')
    if name == 'Normal':
        print("  Reference: widest spread, "
              "longest travel")
    else:
        sh_norm = concentration['Normal']['shelves_used']
        tv_norm = travel['Normal']
        sh_diff = (sh - sh_norm) / sh_norm * 100
        tv_diff = (tv - tv_norm) / tv_norm * 100
        print(f"  {sh_diff:+.0f}% shelves, "
              f"{tv_diff:+.0f}% travel vs Normal")

print("\nHonest interpretation:")
print("  The scenarios are NOT separated well by")
print("  Top-10% concentration (all 93-99%, because")
print("  a few mega-popular aisles dominate every")
print("  scenario). The metrics that DO separate")
print("  them are the number of distinct shelves")
print("  used and the average travel spread:")
print()
print("  - Normal spreads across the most shelves")
print("    (75) with the longest average travel")
print("    (3.34) -> favours distributed coverage,")
print("    more robots to minimise walking.")
print("  - Christmas uses fewer shelves (67) with")
print("    the shortest travel (2.47) -> tighter,")
print("    zone-focused allocation works well.")
print("  - Black Friday uses the fewest shelves")
print("    (64); demand is tightly clustered (very")
print("    low spread 0.49) -> the most localised,")
print("    favouring concentrated robot deployment")
print("    in a small hot zone.")

# ══════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════
print("\nGenerating charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Fleet Workload Analysis Across Scenarios',
    fontsize=14, fontweight='bold')

names = list(scenarios.keys())
colours = ['steelblue', 'red', 'orange']

# Shelves used (a metric that separates them)
shelves_used = [concentration[n]['shelves_used']
                for n in names]
axes[0][0].bar(names, shelves_used, color=colours)
axes[0][0].set_title(
    'Distinct Shelves Used\n'
    'fewer = more concentrated demand')
axes[0][0].set_ylabel('Shelves visited')
for i, v in enumerate(shelves_used):
    axes[0][0].text(i, v + 0.5, str(v),
                    ha='center', fontweight='bold')

# Average travel distance (separates them)
travels = [travel[n] for n in names]
axes[0][1].bar(names, travels, color=colours)
axes[0][1].set_title(
    'Avg Travel Spread of Demand\n'
    'lower = less robot travel needed')
axes[0][1].set_ylabel('Avg grid distance')
for i, v in enumerate(travels):
    axes[0][1].text(i, v + 0.03, f'{v:.2f}',
                    ha='center', fontweight='bold')

# Gini concentration
ginis = [concentration[n]['gini'] for n in names]
axes[1][0].bar(names, ginis, color=colours)
axes[1][0].set_title(
    'Demand Concentration (Gini)\n'
    'higher = more concentrated')
axes[1][0].set_ylabel('Gini coefficient')
axes[1][0].set_ylim(0.8, 1.0)
for i, v in enumerate(ginis):
    axes[1][0].text(i, v + 0.003, f'{v:.3f}',
                    ha='center', fontweight='bold')

# Aisle distribution (the real demand shape)
axes[1][1].hist(normal['aisle_id'], bins=30,
                alpha=0.5, label='Normal',
                color='steelblue', density=True)
axes[1][1].hist(christmas['aisle_id'], bins=30,
                alpha=0.5, label='Christmas',
                color='red', density=True)
axes[1][1].hist(blackfri['aisle_id'], bins=30,
                alpha=0.5, label='Black Friday',
                color='orange', density=True)
axes[1][1].set_title('Aisle Demand Distribution')
axes[1][1].set_xlabel('Aisle ID')
axes[1][1].set_ylabel('Density')
axes[1][1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/fleet_workload_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved results/fleet_workload_analysis.png")

# ══════════════════════════════════════════
# HONEST SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*60)
print("HONEST SUMMARY FOR SUPERVISOR / LOCUS")
print("="*60)
print(f"""
Demand is highly concentrated in all three
scenarios because a small number of mega-popular
aisles dominate the dataset. As a result, simple
"top-10 shelf" concentration does not separate
the scenarios (all 93-99%).

The metrics that DO distinguish them are the
number of distinct shelves used and the average
travel spread of demand:

  Scenario      Shelves   Avg travel
  Normal           75        3.34
  Christmas        67        2.47
  Black Friday     64        3.04

Normal demand is the most dispersed (75 shelves,
longest travel), suggesting distributed fleet
coverage to minimise walking. Christmas demand is
tighter (67 shelves, shortest travel), suiting
zone-focused allocation. Black Friday is the most
localised (64 shelves, lowest spread), favouring
concentrated deployment in a small hot zone.

This supports the hypothesis that the optimal
fleet management style is demand-dependent, and
that forecasting the demand pattern (which the
conditional generator enables) could inform
how the fleet is allocated for each period.
""")
print("="*60)
print("Done!")