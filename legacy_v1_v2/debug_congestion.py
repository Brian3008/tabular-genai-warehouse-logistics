import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── SAME LAYOUT AS warehouse_congestion.py ──
GRID_W = 20
GRID_H = 20

def build_layout():
    is_shelf = np.zeros((GRID_H, GRID_W), dtype=bool)
    for c in range(GRID_W):
        if c % 3 == 2:
            continue
        for r in range(GRID_H):
            if r == 0 or r == GRID_H - 1:
                continue
            is_shelf[r][c] = True
    return is_shelf

is_shelf = build_layout()

def walkable(r, c):
    if r < 0 or r >= GRID_H or \
       c < 0 or c >= GRID_W:
        return False
    return not is_shelf[r][c]

# Build shelf slots
shelf_cells = []
for r in range(GRID_H):
    for c in range(GRID_W):
        if is_shelf[r][c]:
            for dr, dc in [(0,1),(0,-1),
                           (1,0),(-1,0)]:
                if walkable(r+dr, c+dc):
                    shelf_cells.append(
                        ((r,c),(r+dr,c+dc)))
                    break

n_shelf_slots = len(shelf_cells)

# Pick stations
stations = []
for (sr, sc) in [
    (GRID_H//2, 2),
    (GRID_H//2, GRID_W-2),
    (1, GRID_W//2),
    (GRID_H-2, GRID_W//2)]:
    if walkable(sr, sc):
        stations.append((sr, sc))
    else:
        for dc in range(GRID_W):
            if walkable(sr, dc):
                stations.append((sr, dc))
                break

def nearest_station(pos):
    return min(stations,
               key=lambda s:
               abs(s[0]-pos[0]) +
               abs(s[1]-pos[1]))

def aisle_to_slot(aisle_id):
    return aisle_id % n_shelf_slots

print("="*60)
print("DIAGNOSTIC: ONE ROBOT, FEW ORDERS")
print("Step by step behaviour trace")
print("="*60)
print(f"\nGrid: {GRID_W}x{GRID_H}")
print(f"Shelf slots: {n_shelf_slots}")
print(f"Stations: {stations}")

# ── PRINT LAYOUT (small slice) ──
print("\nLayout (S=shelf, .=aisle, P=station):")
for r in range(GRID_H):
    row = ""
    for c in range(GRID_W):
        if (r,c) in stations:
            row += "P "
        elif is_shelf[r][c]:
            row += "S "
        else:
            row += ". "
    print(row)

# ── LOAD A FEW ORDERS ──
print("\nLoading 5 orders from normal_orders.csv...")
df = pd.read_csv('data/normal_orders.csv')
df['slot'] = df['aisle_id'].apply(aisle_to_slot)
orders = df.groupby('order_id')['slot']\
    .apply(list).tolist()[:5]

# Flatten to task list
tasks = []
for order in orders:
    for s in order:
        tasks.append(s)
print(f"Tasks to complete: {len(tasks)}")

# Show what each task means
print("\nTask details:")
for i, t in enumerate(tasks):
    shelf_cell, aisle_cell = shelf_cells[t]
    st = nearest_station(aisle_cell)
    print(f"  Task {i}: slot {t} -> "
          f"shelf {shelf_cell}, "
          f"aisle cell {aisle_cell}, "
          f"station {st}")

# ── GREEDY STEP FUNCTION ──
def greedy_step(pos, target, occupied=set()):
    r, c = pos
    tr, tc = target
    options = []
    if c < tc: options.append((r, c+1))
    if c > tc: options.append((r, c-1))
    if r < tr: options.append((r+1, c))
    if r > tr: options.append((r-1, c))
    for nxt in options:
        if walkable(*nxt) and nxt not in occupied:
            return nxt
    # nudge
    neigh = [(r,c+1),(r,c-1),(r+1,c),(r-1,c)]
    random.shuffle(neigh)
    for nxt in neigh:
        if walkable(*nxt) and nxt not in occupied:
            return nxt
    return None

# ── RUN ONE ROBOT, TRACE EVERY STEP ──
print("\n" + "="*60)
print("STEP-BY-STEP ROBOT TRACE")
print("="*60)

robot_pos = stations[0]
task_idx = 0
stage = 'to_shelf'
target = shelf_cells[tasks[0]][1]
completed = 0
max_steps = 500
step = 0
stuck_count = 0
prev_pos = None

print(f"\nRobot starts at {robot_pos}")
print(f"First target (aisle cell): {target}")
print(f"First shelf cell: "
      f"{shelf_cells[tasks[0]][0]}")
print()

while step < max_steps and task_idx < len(tasks):
    step += 1
    nxt = greedy_step(robot_pos, target)

    # detect stuck
    if nxt is None or nxt == prev_pos:
        stuck_count += 1
    else:
        stuck_count = 0

    prev_pos = robot_pos

    if nxt is None:
        print(f"Step {step:>3}: pos={robot_pos} "
              f"target={target} "
              f"BLOCKED - no valid move! "
              f"stage={stage}")
        # try nudge in all directions
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr,nc = robot_pos[0]+dr, robot_pos[1]+dc
            if walkable(nr,nc):
                print(f"          Nudge available: "
                      f"({nr},{nc})")
        break
    else:
        moved = nxt != robot_pos
        robot_pos = nxt

        # check arrival
        if robot_pos == target:
            if stage == 'to_shelf':
                print(f"Step {step:>3}: "
                      f"pos={robot_pos} "
                      f"ARRIVED at shelf aisle cell "
                      f"-> going to station")
                target = nearest_station(robot_pos)
                stage = 'to_station'
            else:
                completed += 1
                print(f"Step {step:>3}: "
                      f"pos={robot_pos} "
                      f"DELIVERED at station! "
                      f"Task {task_idx} complete "
                      f"({completed} total)")
                task_idx += 1
                if task_idx < len(tasks):
                    target = shelf_cells[
                        tasks[task_idx]][1]
                    stage = 'to_shelf'
                    print(f"          Next target: "
                          f"{target} "
                          f"(shelf: "
                          f"{shelf_cells[tasks[task_idx]][0]})")
        else:
            if step <= 80 or step % 20 == 0:
                dist = abs(robot_pos[0]-target[0]) \
                     + abs(robot_pos[1]-target[1])
                print(f"Step {step:>3}: "
                      f"pos={robot_pos} -> "
                      f"target={target} "
                      f"dist={dist} "
                      f"stage={stage}")

        if stuck_count > 10:
            print(f"\n!! ROBOT STUCK for "
                  f"{stuck_count} steps !!")
            print(f"   pos={robot_pos} "
                  f"target={target}")
            print(f"   walkable neighbours:")
            for dr,dc in [(0,1),(0,-1),
                          (1,0),(-1,0)]:
                nr = robot_pos[0]+dr
                nc = robot_pos[1]+dc
                w = walkable(nr,nc)
                print(f"     ({nr},{nc}) "
                      f"walkable={w}")
            break

print(f"\n{'='*60}")
print(f"DIAGNOSTIC SUMMARY")
print(f"{'='*60}")
print(f"Steps taken:     {step}")
print(f"Tasks completed: {completed} "
      f"/ {len(tasks)}")
print(f"Final position:  {robot_pos}")
print(f"Final target:    {target}")
print(f"Final stage:     {stage}")
if completed == len(tasks):
    print("\nSUCCESS - robot completed all tasks!")
    print("Pathing works. Bug is elsewhere.")
elif step >= max_steps:
    print("\nFAILED - hit step limit.")
    print("Robot is getting lost/stuck.")
else:
    print("\nFAILED - robot got stuck.")
    print("This is the bug to fix.")
print("="*60)