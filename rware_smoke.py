"""
rware_smoke.py - environment + bridge smoke test.

Verifies the FACTS the design rests on, empirically. Nothing here is
taken from arithmetic on the layout formula; every number is measured
from the built environment.

Reads:  data/v3_compare.csv, data/v3_train_order_ids.csv
Writes: results/rware/smoke.json, data/rware/aisle_shelf_map_s*.json
"""
import json
import os
import sys
import time

import numpy as np

import rware_bridge as rb
from rware.warehouse import Action

os.makedirs(rb.DATA_DIR, exist_ok=True)
os.makedirs(rb.RESULTS_DIR, exist_ok=True)

out = {}
print("=" * 68)
print("RWARE SMOKE TEST")
print("=" * 68)

# ── [1] environment ────────────────────────────────────────
env = rb.make_env()
env.reset(seed=0)
h, w = env.grid_size
n_shelves = len(env.shelfs)
n_highway = int(env.highways.sum())
print(f"\n[1] ENVIRONMENT (medium layout, "
      f"{rb.FLEET['n_agents']} agents, queue "
      f"{rb.FLEET['request_queue_size']})")
print(f"    grid            {h} x {w}  = {h*w} cells")
print(f"    storage cells   {n_shelves}   (need >= {rb.N_AISLES})")
print(f"    highway cells   {n_highway}")
print(f"    goals           {env.goals}")
print(f"    agents spawned  {len(env.agents)}")
ok_shelves = n_shelves >= rb.N_AISLES
print(f"    -> storage >= aisles: {'PASS' if ok_shelves else 'FAIL'}")
out["grid"] = [int(h), int(w)]
out["n_storage_cells"] = int(n_shelves)
out["n_highway_cells"] = n_highway
out["goals"] = [[int(a), int(b)] for a, b in env.goals]
out["storage_covers_aisles"] = bool(ok_shelves)

# every non-highway cell really does hold a shelf (the fact the
# loaded-agent path rule depends on)
occupied = {(s.x, s.y) for s in env.shelfs}
nonhw = {(x, y) for y in range(h) for x in range(w)
         if not env.highways[y, x]}
out["every_storage_cell_has_shelf"] = bool(occupied == nonhw)
print(f"    every non-highway cell holds a shelf: "
      f"{'PASS' if occupied == nonhw else 'FAIL'}   "
      f"(loaded agents therefore travel highways only)")

# ── [2] bijection gate ─────────────────────────────────────
print("\n[2] BIJECTION GATE (aisle -> shelf)")
fps = {}
for ms in (11, 22, 33):
    amap = rb.build_aisle_shelf_map(env, ms)
    p = os.path.join(rb.DATA_DIR, f"aisle_shelf_map_s{ms}.json")
    fps[ms] = rb.save_aisle_shelf_map(amap, ms, p)
    print(f"    map_seed {ms:>3}  total={len(amap)}  "
          f"injective={len(set(amap.values())) == len(amap)}  "
          f"fingerprint={fps[ms]}")
# reproducibility: same seed -> same fingerprint, on a fresh env
env2 = rb.make_env()
env2.reset(seed=999)
repro = rb.map_fingerprint(rb.build_aisle_shelf_map(env2, 11)) == fps[11]
distinct = len(set(fps.values())) == 3
print(f"    same map_seed reproduces on a fresh env: "
      f"{'PASS' if repro else 'FAIL'}")
print(f"    three map_seeds give three distinct maps: "
      f"{'PASS' if distinct else 'FAIL'}")
out["map_fingerprints"] = fps
out["map_reproducible"] = bool(repro)
out["maps_distinct"] = bool(distinct)

# ── [3] path planner ───────────────────────────────────────
print("\n[3] PATH PLANNER")
amap = rb.build_aisle_shelf_map(env, 11)
rng = np.random.RandomState(0)
unloaded_ok = loaded_ok = 0
TRIALS = 60
for _ in range(TRIALS):
    a = int(rng.randint(1, rb.N_AISLES + 1))
    sh = env.shelfs[amap[a]]
    st = (int(rng.randint(w)), int(rng.randint(h)))
    if not env.highways[st[1], st[0]]:
        st = (st[0], st[1])
    p = rb.plan_path(env, st, env.agents[0].dir, (sh.x, sh.y),
                     carrying=False)
    unloaded_ok += p is not None
    gx, gy = env.goals[0]
    p2 = rb.plan_path(env, (sh.x, sh.y), env.agents[0].dir, (gx, gy),
                      carrying=True)
    loaded_ok += p2 is not None
print(f"    unloaded routes solved  {unloaded_ok}/{TRIALS}")
print(f"    loaded shelf->goal      {loaded_ok}/{TRIALS}  "
      f"(highways only)")
out["plan_unloaded"] = [unloaded_ok, TRIALS]
out["plan_loaded"] = [loaded_ok, TRIALS]

# ── [4] a real mini-episode ────────────────────────────────
print("\n[4] MINI-EPISODE (real held-out baskets)")
orders = rb.load_real_orders("data/v3_compare.csv",
                             "data/v3_train_order_ids.csv")
print(f"    real orders loaded (small+large only): {len(orders):,}")
sizes = [len(o["aisles"]) for o in orders]
print(f"    items {sum(sizes):,}   mean basket "
      f"{np.mean(sizes):.2f}   max {max(sizes)}")
out["n_real_orders"] = len(orders)
out["n_real_items"] = int(sum(sizes))

sched = rb.build_schedule(orders, 25, np.random.RandomState(0))
sA = rb.stream_from_schedule_real(orders, sched)
t0 = time.time()
res = rb.run_episode(sA, amap, env_seed=0, policy="nearest",
                     step_budget=60_000, inactivity_limit=2_000)
el = time.time() - t0
print(f"    valid={res['valid']}  deadlock={res['deadlock']} "
      f"({res['reason']})")
print(f"    {res['deliveries']}/{res['n_requested']} delivered in "
      f"{res['steps']:,} steps")
print(f"    steps/delivery {res['steps_per_delivery']:.3f}   "
      f"throughput {res['throughput_per_1k']:.2f} per 1k steps")
print(f"    deferrals {res['deferrals']}  perturbations "
      f"{res['perturbations']}  replans {res['replans']}  "
      f"env_asserts {res['env_assert_events']}")
print(f"    wall clock {el:.1f}s  -> {el/max(res['steps'],1)*1000:.3f} "
      f"ms/step")
out["mini_episode"] = res
out["sec_per_1k_steps"] = round(el / max(res["steps"], 1) * 1000, 3)

# ── [5] budget sizing (MEASURED, not guessed) ──────────────
if res["valid"]:
    spd = res["steps_per_delivery"]
    per_item_s = el / max(res["deliveries"], 1)
    print("\n[5] RUN SIZING (measured from the mini-episode)")
    for n_ord in (100, 200, 300, 500):
        items = n_ord * np.mean(sizes)
        print(f"    {n_ord:>4} orders ~ {items:>6,.0f} items  "
              f"~ {items*spd:>9,.0f} steps  "
              f"~ {items*per_item_s:>6.0f} s/run")
    out["measured_steps_per_delivery"] = spd
    out["measured_sec_per_delivery"] = round(per_item_s, 4)

with open(os.path.join(rb.RESULTS_DIR, "smoke.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"\n-> {os.path.join(rb.RESULTS_DIR, 'smoke.json')}")
