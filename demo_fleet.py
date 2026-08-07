"""demo_fleet.py - the warehouse simulator, running live.

Runs in the RWARE env:   .venv_rware\\Scripts\\python.exe demo_fleet.py

Runs two short episodes through the SAME warehouse, same robots, same
schedule - changing only one thing:

  A  real customer baskets, kept whole
  B  the same items, shuffled into random baskets

Everything else is identical, so any difference is caused purely by
whether the items that belong together stayed together. That is the
finding neither generator can reproduce, because their output has no
order number at all.

Imports rware_bridge UNMODIFIED. Reads data/v3_compare.csv and
data/v3_train_order_ids.csv. Writes nothing.
"""
import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rware_bridge as rb                                # noqa: E402

WATCH = "--watch" in sys.argv
FRAME_EVERY = 2          # render every Nth step, so it stays watchable

if WATCH:
    # Show the robots moving. rware 2.0.0 has a bug: reset() calls render()
    # before it creates the shelves, so render_mode cannot be passed to the
    # constructor - it is switched on AFTER reset instead.
    # rware_bridge is NOT modified; make_env is rebound at runtime only.
    _make_env = rb.make_env

    def _watched_make_env(*a, **kw):
        env = _make_env(*a, **kw)
        _reset, _step = env.reset, env.step
        state = {"n": 0}

        def reset(*ra, **rkw):
            out = _reset(*ra, **rkw)
            env.render_mode = "human"
            env.render()
            return out

        def step(*sa, **skw):
            out = _step(*sa, **skw)
            state["n"] += 1
            if state["n"] % FRAME_EVERY == 0:
                env.render()
            return out

        env.reset, env.step = reset, step
        return env

    rb.make_env = _watched_make_env

N_ORDERS = int(sys.argv[sys.argv.index("--orders") + 1]) \
    if "--orders" in sys.argv else 60
# Everything below is seeded, so a given --seed reproduces exactly.
# Change it to draw a different sample and see the number move: this is
# one draw, not a measurement.
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) \
    if "--seed" in sys.argv else 42
MAP_SEED = 20260728


def rule(ch="=", n=68):
    print(ch * n)


rule()
print("  THE WAREHOUSE SIMULATOR - RUNNING LIVE")
rule()
print(f"""
  Two runs through one warehouse. Same robots, same layout, same
  {N_ORDERS} orders arriving in the same sequence.

  The ONLY difference is whether each customer's items stay
  together as a basket, or get shuffled in with everyone else's.
""")

print("  building the warehouse ...")
env = rb.make_env(seed=SEED)
env.reset(seed=SEED)
amap = rb.build_aisle_shelf_map(env, MAP_SEED)
print(f"    grid {env.grid_size}  ·  {len(env.shelfs)} shelves  ·  "
      f"{len(env.agents)} robots")
print(f"    aisle-to-shelf map fingerprint {rb.map_fingerprint(amap)[:16]}")

print("\n  loading real held-out orders ...")
orders = rb.load_real_orders("data/v3_compare.csv",
                             "data/v3_train_order_ids.csv")
pool = rb.real_item_pool(orders)
rng = np.random.RandomState(SEED)
schedule = rb.build_schedule(orders, N_ORDERS, rng)
n_items = sum(s["size"] for s in schedule)
print(f"    {len(schedule)} orders  ·  {n_items} items  ·  identical for both runs")

streams = {
    "A  real baskets, kept whole":
        rb.stream_from_schedule_real(orders, schedule),
    "B  same items, shuffled together":
        rb.stream_from_schedule_pool(pool, schedule,
                                     np.random.RandomState(SEED + 1)),
}

results = {}
for label, stream in streams.items():
    print(f"\n  running:  {label}")
    t0 = time.time()
    r = rb.run_episode(stream, amap, env_seed=SEED, policy="nearest")
    dt = time.time() - t0
    assert not r.get("deadlock"), "  ! deadlock - run is invalid"
    results[label] = r
    print(f"    {r['deliveries']} deliveries in {r['steps']:,} steps "
          f"({dt:.0f}s wall clock, no deadlock)")

# within-basket repeats: the mechanism, measured on these exact streams
dup = {}
for label, stream in streams.items():
    tot = rep = 0
    for o in stream:
        seen = set()
        for a in o["aisles"]:
            tot += 1
            if a in seen:
                rep += 1
            seen.add(a)
    dup[label] = 100.0 * rep / max(tot, 1)

rule()
print("  RESULT")
rule()
A, B = list(results)
ra, rb_ = results[A], results[B]

RECORDED_DUP = {"A": 28.7, "B": 18.3}       # the full recorded run
print(f"\n  THE MECHANISM  -  measured on these exact two streams\n")
print(f"  {'stream':<36} {'THIS RUN':>10} {'recorded':>10}")
for lab, key in ((A, "A"), (B, "B")):
    print(f"  {lab:<36} {dup[lab]:>9.1f}% {RECORDED_DUP[key]:>9.1f}%")
print(f"\n  Real baskets repeat locations about half again as often. The two")
print(f"  columns differ because this is a {N_ORDERS}-order sample and the recorded")
print(f"  figures come from the full run - but the SEPARATION is stable, and")
print(f"  it holds on every draw I have run.")

print(f"\n  {'stream':<36} {'steps/delivery':>15}")
for lab in (A, B):
    r = results[lab]
    print(f"  {lab:<36} {r['steps'] / max(r['deliveries'], 1):>15.2f}")

if "mean_order_completion" in ra:
    da, db = ra["mean_order_completion"], rb_["mean_order_completion"]
    print(f"\n  How long ONE order waits (THIS DRAW ONLY):")
    print(f"    A  real baskets      {da:>10,.0f} steps")
    print(f"    B  shuffled          {db:>10,.0f} steps")
    print(f"    difference           {da - db:>+10,.0f} steps "
          f"({100 * (da - db) / max(db, 1):+.1f}%)")
    print(f"\n  >> DO NOT read that last line as the finding. At {N_ORDERS} orders")
    print(f"     it is noise - across five seeds I measured -15, +408, -17,")
    print(f"     +172 and +437 steps. It changes SIGN. The latency effect")
    print(f"     needs the full run to resolve, and the recorded result is")
    print(f"     +1,041 steps firing on 26 of 30 draws at 300 orders each.")

print(f"""
  Real baskets repeat locations more often - people buy several
  things from one aisle. In this simulator one aisle is one shelf,
  so a repeat has to queue, and the order waits longer.

  A REAL warehouse picks both items in a single visit, so there the
  same structure would SAVE time instead. The mechanism is real; the
  direction depends on the picking model. That is the one thing I
  would want operational data to settle.

  What matters here: this is the largest effect in the whole fleet
  experiment, and it is caused by basket membership alone - the one
  property CTGAN and TabSyn do not model at all.

  IMPORTANT - this single short run is an ILLUSTRATION, not evidence.
  One run of {N_ORDERS} orders proves nothing on its own; the number
  above moves with the draw, and at this scale it can even change
  sign. The evidence is the recorded experiment: 180 episodes at 300
  orders each, +1,041 steps, firing on 26 of 30 independent draws.
""")
rule()
