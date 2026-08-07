"""
Locus-style Warehouse Congestion Simulation
A* pathfinding + racking layout + deadlock breaking
+ congestion heatmaps (mirrors LocusView).
"""

import heapq
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRID_W    = 20
GRID_H    = 20
N_ROBOTS  = 8
MAX_ORDERS = 150

print("="*60)
print("LOCUS-STYLE CONGESTION SIMULATION")
print("A* pathfinding + deadlock breaking + heatmap")
print("="*60)

# ── LAYOUT ──
def build_layout():
    shelf = np.zeros((GRID_H, GRID_W), dtype=bool)
    for c in range(GRID_W):
        if c % 3 == 2:
            continue
        for r in range(1, GRID_H - 1):
            shelf[r][c] = True
    return shelf

is_shelf = build_layout()

def walkable(r, c):
    if r < 0 or r >= GRID_H: return False
    if c < 0 or c >= GRID_W: return False
    return not is_shelf[r][c]

shelf_list  = []
shelf_aisle = {}
for r in range(GRID_H):
    for c in range(GRID_W):
        if not is_shelf[r][c]:
            continue
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            if walkable(r+dr, c+dc):
                shelf_list.append((r, c))
                shelf_aisle[(r,c)] = (r+dr, c+dc)
                break

n_slots = len(shelf_list)
stations_raw = [(0,2),(0,17),(19,2),(19,17)]
stations = [s for s in stations_raw if walkable(*s)]

print(f"\nLayout: {GRID_W}x{GRID_H} grid")
print(f"Shelf slots:  {n_slots}")
print(f"Pick stations: {stations}")

def nearest_station(pos):
    return min(stations, key=lambda s:
               abs(s[0]-pos[0]) + abs(s[1]-pos[1]))

def slot(aisle_id):
    return int(aisle_id) % n_slots

# ── A* ──
_path_cache = {}
def astar(start, goal, blocked=None):
    if start == goal:
        return [start]
    if blocked is None:
        key = (start, goal)
        if key in _path_cache:
            return list(_path_cache[key])
    heap = [(0, start)]
    came = {start: None}
    g = {start: 0}
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur); cur = came[cur]
            path.reverse()
            if blocked is None:
                _path_cache[(start, goal)] = path
            return list(path)
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nb = (cur[0]+dr, cur[1]+dc)
            if not walkable(*nb):
                continue
            if blocked and nb in blocked and nb != goal:
                continue
            ng = g[cur] + 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng; came[nb] = cur
                heapq.heappush(heap, (ng + h(nb), nb))
    return []

# ── ROBOT ──
class Robot:
    def __init__(self, rid, pos):
        self.rid = rid
        self.pos = pos
        self.path = []
        self.stage = 'idle'
        self.picks = 0
        self.travel = 0
        self.waited = 0
        self.blocked_for = 0

def build_tasks(df):
    d = df.copy()
    d['s'] = d['aisle_id'].apply(slot)
    groups = (d.groupby('order_id')['s']
               .apply(list).tolist()[:MAX_ORDERS])
    return [s for order in groups for s in order]

# ── RUN ──
def run(df, strategy, heat=None):
    random.seed(SEED)
    tasks = build_tasks(df)
    n_tasks = len(tasks)
    remaining = list(tasks)
    rr_idx = 0
    zone_q = [[] for _ in range(N_ROBOTS)]
    if strategy == 'zoned':
        for t in tasks:
            ac = shelf_aisle[shelf_list[t]]
            band = min(ac[1]*N_ROBOTS//GRID_W, N_ROBOTS-1)
            zone_q[band].append(t)

    robots = [Robot(i, stations[i % len(stations)])
              for i in range(N_ROBOTS)]

    def get_task(rob):
        nonlocal rr_idx, remaining
        if strategy == 'nearest':
            if not remaining: return None
            pos = rob.pos
            window = min(len(remaining), 10)
            idx = min(range(window), key=lambda k:
                abs(shelf_aisle[shelf_list[remaining[k]]][0]-pos[0]) +
                abs(shelf_aisle[shelf_list[remaining[k]]][1]-pos[1]))
            return remaining.pop(idx)
        elif strategy == 'roundrobin':
            if rr_idx >= n_tasks: return None
            t = tasks[rr_idx]; rr_idx += 1; return t
        else:
            band = rob.rid % N_ROBOTS
            if zone_q[band]: return zone_q[band].pop(0)
            for b in range(N_ROBOTS):
                if zone_q[b]: return zone_q[b].pop(0)
            return None

    completed = 0
    total_travel = 0
    total_wait = 0

    for tick in range(60000):
        for rob in robots:
            if rob.stage == 'idle' and not rob.path:
                t = get_task(rob)
                if t is not None:
                    goal = shelf_aisle[shelf_list[t]]
                    rob.path = astar(rob.pos, goal)[1:]
                    rob.stage = 'shelf'

        all_done = all(r.stage == 'idle' and not r.path
                       for r in robots)
        tasks_done = (
            (strategy == 'nearest' and not remaining) or
            (strategy == 'roundrobin' and rr_idx >= n_tasks) or
            (strategy == 'zoned' and
             not any(zone_q[b] for b in range(N_ROBOTS))))
        if all_done and tasks_done:
            break

        occupied = {r.pos for r in robots}
        for rob in robots:
            if not rob.path:
                continue
            nxt = rob.path[0]
            if nxt in (occupied - {rob.pos}):
                rob.waited += 1
                total_wait += 1
                rob.blocked_for += 1
                if heat is not None:
                    heat[rob.pos[0]][rob.pos[1]] += 1
                if rob.blocked_for >= 5:
                    goal = rob.path[-1]
                    blockers = (occupied - {rob.pos})
                    new_path = astar(rob.pos, goal,
                                     blocked=blockers)
                    if len(new_path) > 1:
                        rob.path = new_path[1:]
                    rob.blocked_for = 0
            else:
                rob.blocked_for = 0
                occupied.discard(rob.pos)
                occupied.add(nxt)
                rob.pos = nxt
                rob.path.pop(0)
                rob.travel += 1
                total_travel += 1
                if heat is not None:
                    heat[nxt[0]][nxt[1]] += 0.1

            if not rob.path:
                if rob.stage == 'shelf':
                    st = nearest_station(rob.pos)
                    rob.path = astar(rob.pos, st)[1:]
                    rob.stage = 'station'
                elif rob.stage == 'station':
                    rob.picks += 1
                    completed += 1
                    rob.stage = 'idle'

    tot = total_travel + total_wait
    return {
        'completed': completed,
        'travel': total_travel,
        'wait': total_wait,
        'cong_pct': total_wait/tot*100 if tot > 0 else 0,
        'throughput': completed/tot if tot > 0 else 0,
    }

# ── MAIN ──
scenarios = {
    'Normal':       pd.read_csv('data/normal_orders.csv'),
    'Christmas':    pd.read_csv('data/christmas_orders.csv'),
    'Black Friday': pd.read_csv('data/blackfriday_orders.csv'),
}
strats = ['nearest','roundrobin','zoned']
strat_nm = {'nearest':'Nearest-task',
            'roundrobin':'Round-robin','zoned':'Zoned'}

print("\n" + "="*60)
print(f"RESULTS  ({N_ROBOTS} robots, up to {MAX_ORDERS} orders)")
print("="*60)
print(f"\n{'Scenario':<14}{'Strategy':<14}"
      f"{'Done':>5}{'Travel':>8}{'Wait':>7}"
      f"{'Cong%':>7}{'Thru':>9}")
print("-"*64)

results = {}
for sname, sdf in scenarios.items():
    results[sname] = {}
    for strat in strats:
        m = run(sdf, strat)
        results[sname][strat] = m
        print(f"{sname:<14}{strat_nm[strat]:<14}"
              f"{m['completed']:>5}{m['travel']:>8}"
              f"{m['wait']:>7}{m['cong_pct']:>6.1f}%"
              f"{m['throughput']:>9.4f}")
    print()

print("="*60)
print("BEST STRATEGY PER SCENARIO (by throughput)")
print("="*60)
bests = {}
for sname in scenarios:
    best = max(results[sname],
               key=lambda s: results[sname][s]['throughput'])
    bests[sname] = best
    m = results[sname][best]
    print(f"  {sname:<14} -> {strat_nm[best]:<14}"
          f"  done={m['completed']:>3}, cong={m['cong_pct']:.1f}%")

unique = set(bests.values())
print()
if len(unique) > 1:
    print("  Best strategy CHANGES by period.")
    print("  Congestion shows WHY: concentrated demand")
    print("  jams aisles; the right strategy avoids it.")
else:
    b = list(unique)[0]
    print(f"  {strat_nm[b]} leads overall, but")
    print("  congestion cost differs by period.")

print("\nGenerating congestion heatmaps...")
_path_cache.clear()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Congestion Heatmaps — Nearest-task, {N_ROBOTS} robots\n'
             'Red = robots waiting (mirrors LocusView)',
             fontsize=13, fontweight='bold')
for ax, (sname, sdf) in zip(axes, scenarios.items()):
    heat = np.zeros((GRID_H, GRID_W))
    run(sdf, 'nearest', heat=heat)
    disp = np.ma.masked_where(is_shelf, heat)
    im = ax.imshow(disp, cmap='hot', interpolation='nearest', vmin=0)
    ax.imshow(np.ma.masked_where(~is_shelf, np.ones_like(heat)*0.5),
              cmap='Greys', vmin=0, vmax=1, alpha=0.5)
    ax.set_title(sname, fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label='congestion')
plt.tight_layout()
plt.savefig('results/congestion_heatmaps.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved results/congestion_heatmaps.png")

fig2, ax2 = plt.subplots(figsize=(12, 6))
slist = list(scenarios.keys())
x = np.arange(len(slist)); w = 0.25
cls = ['steelblue','orange','green']
for i, strat in enumerate(strats):
    vals = [results[s][strat]['cong_pct'] for s in slist]
    ax2.bar(x+(i-1)*w, vals, w, label=strat_nm[strat],
            color=cls[i], alpha=0.85)
ax2.set_title('Congestion % by Strategy and Scenario\n(lower = better)',
              fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(slist)
ax2.set_ylabel('% of steps waiting')
ax2.legend(title='Strategy')
plt.tight_layout()
plt.savefig('results/congestion_by_strategy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved results/congestion_by_strategy.png")

fig3, ax3 = plt.subplots(figsize=(12, 6))
for i, strat in enumerate(strats):
    vals = [results[s][strat]['throughput'] for s in slist]
    ax3.bar(x+(i-1)*w, vals, w, label=strat_nm[strat],
            color=cls[i], alpha=0.85)
ax3.set_title('Throughput by Strategy and Scenario\n(higher = better)',
              fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels(slist)
ax3.set_ylabel('Throughput (orders/step)')
ax3.legend(title='Strategy')
plt.tight_layout()
plt.savefig('results/fleet_throughput.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved results/fleet_throughput.png")

print("\n" + "="*60)
print("Done!")