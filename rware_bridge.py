"""
rware_bridge.py - core bridge from an order stream to RWARE requests.

NO SIDE EFFECTS ON IMPORT. Every entry point is a function.

WHAT THIS IS
------------
RWARE (uoe-agents/robotic-warehouse) generates its own shelf requests
uniformly at random (warehouse.py:799-803 on reset, :918-920 on
delivery). It cannot read an external order file. This module drives
the request stream from OUR order data instead, WITHOUT modifying the
installed package: we call env.step() normally, then diff
env.request_queue against what we last wrote. Any slot that changed is
a delivery; we log it and overwrite the slot with the next request
from our stream.

That is safe because the scripted policy reads env.request_queue
directly and never consumes the observation vector, so the one-step
stale random refill inside step() is never acted upon.

STATED LIMITATION (must appear in every output file)
----------------------------------------------------
The aisle -> shelf mapping is a MODELING ABSTRACTION, not a real
warehouse layout. The comparison's validity rests on the mapping being
IDENTICAL across real and synthetic runs - NOT on the mapping being
physically realistic. Same standing as the travel-distance proxy.

Because aisle geometry is relabelling-sensitive (tabsyn_conditional_
geometry.py measured the travel gap ranging 2.94 aisles = 4.6x the
noise bar across aisle relabellings), a SINGLE arbitrary map could
manufacture or erase a fleet difference. Every verdict must therefore
be a fire rate across MULTIPLE independent mappings.
"""

import hashlib
import json
import os
import sys
from collections import deque

import numpy as np
import pandas as pd

from rware.warehouse import (
    Action,
    Direction,
    RewardType,
    Warehouse,
    _LAYER_AGENTS,
    _LAYER_SHELFS,
)

# UTF-8 stdout guard - project convention. Three scripts previously
# crashed a default cp1252 Windows console on a non-ASCII glyph.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# FIXED EXPERIMENT CONSTANTS (approved: medium layout,
# 8 agents, queue 8, no fleet-size sweep)
# ══════════════════════════════════════════════════════════
FLEET = dict(
    shelf_columns=5,
    shelf_rows=2,
    column_height=8,
    n_agents=8,
    request_queue_size=8,
)

N_AISLES = 134          # verified: real/TabSyn/CTGAN all use IDs 1..134
SMALL_MAX = 10          # screened buckets (screen_results.json "60/75")
LARGE_MIN = 14

DATA_DIR = os.path.join("data", "rware")
RESULTS_DIR = os.path.join("results", "rware")


def make_env(seed=None, max_steps=None):
    """Build the fixed fleet. max_steps/max_inactivity are OURS to
    manage (the registered presets hard-code max_steps=500, which
    would silently truncate a run); we pass None and enforce the
    budget in run_episode so a truncation can never masquerade as a
    completed run."""
    env = Warehouse(
        shelf_columns=FLEET["shelf_columns"],
        shelf_rows=FLEET["shelf_rows"],
        column_height=FLEET["column_height"],
        n_agents=FLEET["n_agents"],
        msg_bits=0,
        sensor_range=1,
        request_queue_size=FLEET["request_queue_size"],
        max_inactivity_steps=None,
        max_steps=max_steps,
        reward_type=RewardType.INDIVIDUAL,
    )
    return env


# ══════════════════════════════════════════════════════════
# MAPPING 1: aisle_id -> shelf  (the fixed bijection)
# ══════════════════════════════════════════════════════════
def build_aisle_shelf_map(env, map_seed):
    """A seeded bijection aisle_id (1..134) -> index into env.shelfs.

    env.shelfs is deterministic (warehouse.py:774-781 walks
    np.indices in fixed order), so a given map_seed reproduces the
    same map on every process and every condition."""
    n_shelves = len(env.shelfs)
    assert n_shelves >= N_AISLES, (
        f"FATAL: layout has {n_shelves} storage cells < {N_AISLES} "
        f"aisles - cannot build a bijection")
    rng = np.random.RandomState(map_seed)
    chosen = rng.permutation(n_shelves)[:N_AISLES]
    amap = {int(a): int(s) for a, s in
            zip(range(1, N_AISLES + 1), chosen)}

    # bijection gate
    assert len(amap) == N_AISLES, "FATAL: map is not total"
    assert len(set(amap.values())) == N_AISLES, \
        "FATAL: map is not injective (two aisles share a shelf)"
    return amap


def map_fingerprint(amap):
    """Stable hash - proves every condition used the SAME map."""
    blob = json.dumps({str(k): amap[k] for k in sorted(amap)},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def save_aisle_shelf_map(amap, map_seed, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "map_seed": map_seed,
        "n_aisles": len(amap),
        "fingerprint": map_fingerprint(amap),
        "LIMITATION": (
            "Modeling abstraction, NOT a real warehouse layout. "
            "Validity rests on this map being IDENTICAL across real "
            "and synthetic runs, not on physical realism."),
        "aisle_to_shelf_index": {str(k): amap[k] for k in sorted(amap)},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload["fingerprint"]


# ══════════════════════════════════════════════════════════
# MAPPING 2: items -> baskets  (the assembly rule)
# ══════════════════════════════════════════════════════════
def load_real_orders(csv_path, train_ids_path=None):
    """Real baskets, restricted to small+large.

    'mid' is dropped on EVERY stream because CTGAN/TabSyn were trained
    on small+large only - handling caveat 3 (population mismatch) at
    source instead of discovering it afterwards."""
    df = pd.read_csv(csv_path)
    if train_ids_path is not None:
        tr = set(pd.read_csv(train_ids_path)["order_id"])
        overlap = set(df["order_id"]) & tr
        assert not overlap, (
            f"FATAL: {len(overlap)} training orders leaked into "
            f"{csv_path}")

    size = df.groupby("order_id")["aisle_id"].size()
    grp = np.where(size <= SMALL_MAX, "small",
                   np.where(size >= LARGE_MIN, "large", "mid"))
    keep = size.index[grp != "mid"]
    df = df[df["order_id"].isin(set(keep))]

    orders = []
    for oid, g in df.groupby("order_id", sort=True):
        aisles = [int(a) for a in g["aisle_id"].values]
        orders.append({
            "order_id": int(oid),
            "grp": "small" if len(aisles) <= SMALL_MAX else "large",
            "aisles": aisles,
        })
    return orders


def load_item_pool(csv_path):
    """Flat item pool keyed by order_size_grp.

    CTGAN and TabSyn emit INDEPENDENT ITEM ROWS with no order_id -
    they do not model basket membership at all. This is why an
    assembly rule is unavoidable, and why stream B exists to price
    it."""
    df = pd.read_csv(csv_path)
    assert "order_id" not in df.columns, (
        f"{csv_path} has order_id - use load_real_orders for real "
        f"baskets")
    pool = {}
    for g in ("small", "large"):
        sel = df[df["order_size_grp"] == g]
        pool[g] = sel["aisle_id"].astype(int).values
        assert len(pool[g]) > 0, f"FATAL: empty {g} pool in {csv_path}"
    return pool


def real_item_pool(orders):
    """Stream B's pool: the SAME real items, but stripped of basket
    membership so the assembly rule can be applied identically."""
    pool = {"small": [], "large": []}
    for o in orders:
        pool[o["grp"]].extend(o["aisles"])
    return {g: np.asarray(v, dtype=int) for g, v in pool.items()}


def build_schedule(orders, n_orders, rng):
    """The (grp, size) schedule, taken from REAL orders and REUSED
    verbatim by streams B, C and D.

    This is what makes the comparison clean: order count, order sizes
    and total item count are IDENTICAL across all four streams, so the
    only thing that varies is which aisles fill the slots."""
    idx = rng.choice(len(orders), size=n_orders, replace=False)
    return [{"order_id": orders[i]["order_id"],
             "grp": orders[i]["grp"],
             "size": len(orders[i]["aisles"])} for i in idx]


def stream_from_schedule_real(orders, schedule):
    """Stream A: the true held-out baskets for the scheduled orders."""
    by_id = {o["order_id"]: o for o in orders}
    return [{"order_id": s["order_id"], "grp": s["grp"],
             "aisles": list(by_id[s["order_id"]]["aisles"])}
            for s in schedule]


def stream_from_schedule_pool(pool, schedule, rng):
    """Streams B/C/D: same schedule, aisles drawn from an item pool.

    Drawn WITH replacement, i.e. n i.i.d. items from the stream's
    aisle distribution.

    Replacement is not a detail. 28.6% of real items repeat an aisle
    already in their own basket (large baskets average 7.0 repeats,
    small 1.1), and a repeat costs real fleet time because one aisle
    is one shelf and the duplicate has to be deferred and re-issued.
    Drawing DISTINCT aisles would emit zero repeats - a distortion
    falling hardest on large baskets, which is precisely the axis
    under test. Sampling i.i.d. keeps repeats present; the fact that
    i.i.d. draws under-produce them relative to real baskets (which
    are positively correlated within a basket) is exactly the
    assembly artifact stream B exists to price."""
    out = []
    for s in schedule:
        p = pool[s["grp"]]
        vals, counts = np.unique(p, return_counts=True)
        probs = counts / counts.sum()
        aisles = rng.choice(vals, size=s["size"], replace=True, p=probs)
        out.append({"order_id": s["order_id"], "grp": s["grp"],
                    "aisles": [int(a) for a in aisles]})
    return out


# ══════════════════════════════════════════════════════════
# PATH PLANNING  (BFS over (x, y, dir); cost 1 per action)
# ══════════════════════════════════════════════════════════
_DELTA = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}
_WRAP = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]


def _turn(d, action):
    i = _WRAP.index(d)
    return _WRAP[(i + 1) % 4] if action == Action.RIGHT \
        else _WRAP[(i - 1) % 4]


def plan_path(env, start, start_dir, goal, carrying, avoid=None):
    """Shortest action sequence from (start, start_dir) to goal.

    Passability - this is the one rule that actually matters:
    EVERY non-highway cell holds a shelf (warehouse.py:774-781 makes a
    Shelf for every non-highway cell), and a carrying agent cannot
    enter a cell with a standing shelf (:832-846). So a LOADED agent
    can only travel on highways - plus its own start cell (it is
    standing there) and its goal cell (which is empty precisely
    because it is carrying that cell's shelf).

    Other agents ARE avoided when `avoid` is supplied (their current
    cells are treated as blocked). This is what prevents the head-on
    standoff: RWARE refuses to commit a 2-cycle swap (:858-861), so
    two agents facing each other in a one-wide corridor would never
    move. Callers retry with avoid=None if avoidance makes the goal
    unreachable, so a blocked corridor degrades to "try anyway and
    let the stall detector deal with it" rather than to a NOOP."""
    h, w = env.grid_size
    avoid = avoid or set()

    def passable(x, y):
        if not (0 <= x < w and 0 <= y < h):
            return False
        if (x, y) in avoid and (x, y) != start:
            return False
        if not carrying:
            return True
        if (x, y) == start or (x, y) == goal:
            return True
        return bool(env.highways[y, x])

    if not passable(*goal):
        return None

    src = (start[0], start[1], start_dir)
    seen = {src}
    q = deque([(src, [])])
    while q:
        (x, y, d), path = q.popleft()
        if (x, y) == goal:
            return path
        if len(path) > 4 * h * w:
            return None
        dx, dy = _DELTA[d]
        nxt = (x + dx, y + dy)
        if passable(*nxt):
            s = (nxt[0], nxt[1], d)
            if s not in seen:
                seen.add(s)
                q.append((s, path + [Action.FORWARD]))
        for a in (Action.LEFT, Action.RIGHT):
            s = (x, y, _turn(d, a))
            if s not in seen:
                seen.add(s)
                q.append((s, path + [a]))
    return None


# ══════════════════════════════════════════════════════════
# THE SCRIPTED FLEET  (no RL - this project is not doing
# multi-agent RL)
# ══════════════════════════════════════════════════════════
# Task cycle per agent:
#   IDLE -> TO_SHELF -> (load) -> TO_GOAL -> (delivered by env)
#        -> TO_HOME  -> (unload) -> IDLE
#
# RETURN-TO-ORIGIN is deliberate. RWARE lets an agent drop a shelf on
# ANY free storage cell, which would slowly scramble the layout and
# inject condition-dependent noise into the geometry we are measuring.
# Returning every shelf to its own cell keeps the warehouse stationary
# and IDENTICAL across all four streams - fixed slotting.

IDLE, TO_SHELF, TO_GOAL, TO_HOME = "idle", "to_shelf", "to_goal", "to_home"


class ScriptedFleet:
    """Scripted (non-RL) fleet. Two assignment styles:
      nearest - each free agent takes the closest unclaimed request
      random  - each free agent takes a random unclaimed request

    Replans from scratch every step. That is deliberate: a cached path
    desynchronises the moment RWARE cancels a move, and BFS over 320
    cells x 4 headings is cheap next to the cost of acting on a stale
    plan.
    """

    STALL_DETOUR = 4      # steps stuck in one cell before detouring
    DETOUR_HOLD = 6       # steps a detour target stays active

    def __init__(self, env, policy="nearest", seed=0):
        assert policy in ("nearest", "random"), policy
        self.env = env
        self.policy = policy
        self.rng = np.random.RandomState(seed)
        n = env.n_agents
        self.state = [IDLE] * n
        self.target = [None] * n          # shelf object
        self.home = [None] * n            # (x, y) of that shelf
        self.claimed = set()              # id(shelf) in progress
        self.ignore = set()               # id(shelf) not from our stream
        self.stall = [0] * n
        self.last_xy = [None] * n
        self.detour = [None] * n          # (cell, expires_at)
        self.idle_dest = [None] * n       # idle staging target
        self.detours = 0
        self.perturbations = 0
        self.replans = 0
        self.unreachable = 0
        self.t = 0

    # -- assignment ---------------------------------------------------
    def _free_requests(self):
        return [s for s in self.env.request_queue
                if s is not None and id(s) not in self.claimed
                and id(s) not in self.ignore]

    def _assign(self, i):
        agent = self.env.agents[i]
        cands = self._free_requests()
        if not cands:
            return False
        if self.policy == "random":
            pick = cands[self.rng.randint(len(cands))]
        else:
            d = [abs(s.x - agent.x) + abs(s.y - agent.y) for s in cands]
            pick = cands[int(np.argmin(d))]
        self.target[i] = pick
        self.home[i] = (int(pick.x), int(pick.y))
        self.claimed.add(id(pick))
        self.state[i] = TO_SHELF
        return True

    def _release(self, i):
        if self.target[i] is not None:
            self.claimed.discard(id(self.target[i]))
        self.target[i] = None
        self.home[i] = None
        self.state[i] = IDLE

    def _goal_cell(self, agent):
        gs = [(int(gx), int(gy)) for gx, gy in self.env.goals]
        d = [abs(gx - agent.x) + abs(gy - agent.y) for gx, gy in gs]
        return gs[int(np.argmin(d))]

    def _pick_detour(self, i):
        """DEADLOCK GUARD layer 3: send a stuck agent to a random
        nearby free highway cell.

        A rotation-only 'perturbation' does NOT break a head-on
        standoff - RWARE refuses to commit a 2-cycle swap, so the
        agent has to physically vacate the corridor. Hence a real
        destination, held for several steps."""
        env = self.env
        h, w = env.grid_size
        occupied = {(a.x, a.y) for j, a in enumerate(env.agents)
                    if j != i}
        agent = env.agents[i]
        cells = [(x, y) for y in range(h) for x in range(w)
                 if env.highways[y, x] and (x, y) not in occupied]
        if not cells:
            return None
        near = [c for c in cells
                if 1 <= abs(c[0] - agent.x) + abs(c[1] - agent.y) <= 6]
        pool = near or cells
        return pool[self.rng.randint(len(pool))]

    def _drift(self, i, xy, agent):
        """Idle staging move: head for a free highway cell, re-picking
        on arrival or expiry. Returns the action to take."""
        d = self.idle_dest[i]
        if d is None or xy == d[0] or self.t >= d[1]:
            cell = self._pick_detour(i)
            d = (cell, self.t + 15) if cell else None
            self.idle_dest[i] = d
        if d is None:
            return Action.NOOP.value
        positions = {(a.x, a.y) for j, a in enumerate(self.env.agents)
                     if j != i}
        path = plan_path(self.env, xy, agent.dir, d[0], False,
                         positions)
        if not path:
            path = plan_path(self.env, xy, agent.dir, d[0], False)
        return path[0].value if path else Action.NOOP.value

    # -- one decision per agent per step -------------------------------
    def act(self):
        env = self.env
        self.t += 1
        actions = []
        positions = {(a.x, a.y) for a in env.agents}

        for i, agent in enumerate(env.agents):
            xy = (agent.x, agent.y)
            carrying = agent.carrying_shelf is not None

            # Stall bookkeeping is POSITION-based. Keying it on
            # heading as well let an agent mask a standoff by
            # rotating back and forth for ever.
            if xy == self.last_xy[i]:
                self.stall[i] += 1
            else:
                self.stall[i] = 0
            self.last_xy[i] = xy

            if self.state[i] == IDLE and not carrying:
                if not self._assign(i):
                    # An idle agent must NOT park where it stands. A
                    # stationary robot is a wall: idle agents sitting
                    # in corridors deadlocked the fleet outright
                    # (5 of 8 idle at stall ~3000, boxing in the
                    # working agents). Idle agents drift to a free
                    # highway cell instead - staging behaviour, and
                    # applied identically in every condition.
                    actions.append(self._drift(i, xy, agent))
                    continue

            # ---- arrival / transition ----
            if self.state[i] == TO_SHELF:
                if carrying:
                    self.state[i] = TO_GOAL
                elif xy == self.home[i]:
                    if env.grid[_LAYER_SHELFS, agent.y, agent.x]:
                        actions.append(Action.TOGGLE_LOAD.value)
                        continue
                    self._release(i)          # shelf gone: give up
                    actions.append(Action.NOOP.value)
                    continue
            elif self.state[i] == TO_GOAL:
                if not carrying:
                    self.state[i] = TO_SHELF
                elif xy in [(int(gx), int(gy)) for gx, gy in env.goals]:
                    self.state[i] = TO_HOME
            elif self.state[i] == TO_HOME:
                if xy == self.home[i]:
                    if carrying:
                        # release BEFORE going idle: setting state
                        # without dropping the claim leaks one shelf
                        # id per delivery, and once every queue slot
                        # holds a leaked id nothing can be assigned
                        # again - a silent, total stall.
                        actions.append(Action.TOGGLE_LOAD.value)
                        self._release(i)
                        continue
                    self._release(i)
                    actions.append(Action.NOOP.value)
                    continue

            # ---- destination for this step ----
            if self.state[i] == TO_SHELF:
                dest = self.home[i]
            elif self.state[i] == TO_GOAL:
                dest = self._goal_cell(agent)
            elif self.state[i] == TO_HOME:
                dest = self.home[i]
            else:
                actions.append(Action.NOOP.value)
                continue

            # ---- detour override (deadlock guard) ----
            if self.detour[i] is not None:
                cell, expires = self.detour[i]
                if self.t >= expires or xy == cell:
                    self.detour[i] = None
                else:
                    dest = cell
            elif self.stall[i] >= self.STALL_DETOUR:
                cell = self._pick_detour(i)
                if cell is not None:
                    self.detour[i] = (cell, self.t + self.DETOUR_HOLD)
                    self.detours += 1
                    self.stall[i] = 0
                    dest = cell

            # ---- plan: avoid other agents, else try anyway ----
            avoid = positions - {xy}
            path = plan_path(env, xy, agent.dir, dest, carrying, avoid)
            if not path:
                path = plan_path(env, xy, agent.dir, dest, carrying)
                if not path:
                    self.unreachable += 1
                    actions.append(Action.NOOP.value)
                    continue
                self.perturbations += 1
            self.replans += 1
            actions.append(path[0].value)
        return actions

    def notify_delivered(self, shelf):
        """A tagged shelf reached a goal: its carrier switches to
        TO_HOME so the layout is restored (fixed slotting)."""
        for i, t in enumerate(self.target):
            if t is shelf and self.state[i] == TO_GOAL:
                self.state[i] = TO_HOME


# ══════════════════════════════════════════════════════════
# THE EPISODE RUNNER  (stream injection + deadlock guard)
# ══════════════════════════════════════════════════════════
def run_episode(stream, amap, env_seed, policy="nearest",
                step_budget=200_000, inactivity_limit=3_000,
                verbose=False):
    """Run one order stream through the fleet.

    DEADLOCK GUARD - four layers, none of them silent:
      1. hard step budget
      2. inactivity limit (no delivery for N steps -> abort)
      3. per-agent stall detector (in ScriptedFleet: replan at 3,
         bounded random perturbation at 6)
      4. completion assertion - a run that ends with
         deliveries < requested is flagged deadlock=True and its
         metrics are marked INVALID for aggregation.
    """
    env = make_env()
    env.reset(seed=env_seed)
    shelves = env.shelfs

    # flatten the stream into tagged requests, in order
    pending = deque()
    for o in stream:
        for a in o["aisles"]:
            pending.append((o["order_id"], int(a)))
    n_requested = len(pending)
    assert n_requested > 0, "FATAL: empty stream"

    fleet = ScriptedFleet(env, policy=policy, seed=env_seed)

    qsize = env.request_queue_size
    expected = [None] * qsize
    slot_tag = [None] * qsize
    deferrals = 0

    def next_request(occupied):
        """Pull the next request whose shelf is not already queued.
        Under 1 aisle = 1 shelf, an aisle cannot be requested twice
        concurrently, so a repeat is DEFERRED and re-issued later."""
        nonlocal deferrals
        for _ in range(len(pending)):
            oid, aisle = pending.popleft()
            sh = shelves[amap[aisle]]
            if id(sh) in occupied:
                pending.append((oid, aisle))
                deferrals += 1
                continue
            return oid, aisle, sh
        return None

    # initial fill
    occupied = set()
    for i in range(qsize):
        got = next_request(occupied)
        if got is None:
            break
        oid, aisle, sh = got
        expected[i] = sh
        slot_tag[i] = (oid, aisle)
        occupied.add(id(sh))
    env.request_queue = [s for s in expected if s is not None]
    expected = list(env.request_queue)
    slot_tag = [t for t in slot_tag if t is not None]

    def top_up(now):
        """Fill every untagged slot we can from the stream.

        Called EVERY step, not only after a delivery. A slot goes
        untagged when the stream is exhausted, or when the next
        pending request names an aisle already in the queue (one aisle
        = one shelf, so it must wait). Those must keep being retried.

        The subtle case, found by a fixture run that stalled at
        1187/1188: RWARE's own random refill can drop the very shelf a
        pending request wants into an untagged slot. The request then
        blocks on ITSELF - the shelf is 'in the queue', but in a slot
        no agent will serve, so no delivery ever happens and no retry
        is ever triggered. The fix is to ADOPT that slot rather than
        defer: if the shelf we want is already sitting in a free slot,
        tag it where it is."""
        nonlocal deferrals
        free_slots = [i for i in range(len(expected))
                      if slot_tag[i][0] is None]
        if not free_slots or not pending:
            return
        slot_by_shelf = {id(env.request_queue[i]): i for i in free_slots}
        occ = {id(s) for s in env.request_queue}
        budget = 2 * len(pending)
        while pending and free_slots and budget > 0:
            budget -= 1
            oid, aisle = pending.popleft()
            sh = shelves[amap[aisle]]
            if id(sh) in slot_by_shelf:
                i = slot_by_shelf.pop(id(sh))     # adopt in place
                free_slots.remove(i)
            elif id(sh) in occ:
                pending.append((oid, aisle))      # busy elsewhere
                deferrals += 1
                continue
            else:
                i = free_slots.pop(0)
                slot_by_shelf.pop(id(env.request_queue[i]), None)
                occ.discard(id(env.request_queue[i]))
                env.request_queue[i] = sh
                occ.add(id(sh))
            expected[i] = sh
            slot_tag[i] = (oid, aisle)
            fleet.ignore.discard(id(sh))
            first_seen.setdefault(oid, now)

    first_seen, last_done, done_count = {}, {}, {}
    per_order_size = {o["order_id"]: len(o["aisles"]) for o in stream}
    for t in slot_tag:
        first_seen.setdefault(t[0], 0)

    steps = 0
    delivered = 0
    since_delivery = 0
    env_assert_events = 0
    deadlock = False
    reason = "completed"

    while delivered < n_requested:
        if steps >= step_budget:
            deadlock, reason = True, "step_budget_exhausted"
            break
        if since_delivery >= inactivity_limit:
            deadlock, reason = True, "inactivity_limit"
            break

        actions = fleet.act()
        # INVARIANT: at most one claimed shelf per agent. A claim that
        # is never released is invisible in the metrics until the run
        # stalls outright, so it is asserted every step rather than
        # inferred afterwards.
        assert len(fleet.claimed) <= env.n_agents, (
            f"FATAL: claim leak - {len(fleet.claimed)} claims for "
            f"{env.n_agents} agents at step {steps}")
        try:
            env.step(actions)
        except AssertionError:
            # RWARE's movement-commit assert (warehouse.py:878) can
            # fire on a rare multi-agent configuration. Re-step with
            # all-NOOP: step() reassigns req_action from the action
            # list at :812, so the retry is clean and deterministic.
            env_assert_events += 1
            env.step([Action.NOOP.value] * env.n_agents)
        steps += 1

        # ---- DIFF THE QUEUE: this is the injection point ----
        hits = []
        for i, sh in enumerate(env.request_queue):
            if i < len(expected) and sh is not expected[i]:
                hits.append(i)
        if hits:
            progressed = False
            for i in hits:
                oid, aisle = slot_tag[i]
                fleet.notify_delivered(expected[i])
                # Untagged slot = RWARE's own random refill, NOT one of
                # our requests. Counting it would let `delivered`
                # overshoot the stream and silently inflate throughput.
                if oid is not None:
                    delivered += 1
                    progressed = True
                    done_count[oid] = done_count.get(oid, 0) + 1
                    if done_count[oid] == per_order_size.get(oid, 0):
                        last_done[oid] = steps
                expected[i] = env.request_queue[i]
                slot_tag[i] = (None, None)
                fleet.ignore.add(id(env.request_queue[i]))
            since_delivery = 0 if progressed else since_delivery + 1
        else:
            since_delivery += 1
        top_up(steps)

    # On a deadlock, capture WHY. A stalled run must be diagnosable
    # after the fact, not just labelled.
    diagnostic = None
    if deadlock:
        diagnostic = {
            "pending_unissued": len(pending),
            "queue_tagged": sum(t[0] is not None for t in slot_tag),
            "queue_size": len(env.request_queue),
            "claimed": len(fleet.claimed),
            "ignored": len(fleet.ignore),
            "agents": [
                {"state": fleet.state[i],
                 "xy": [int(a.x), int(a.y)],
                 "carrying": a.carrying_shelf is not None,
                 "stall": fleet.stall[i],
                 "has_target": fleet.target[i] is not None}
                for i, a in enumerate(env.agents)],
        }

    completions = [last_done[o] - first_seen.get(o, 0)
                   for o in last_done]
    valid = (not deadlock) and delivered == n_requested

    return {
        "valid": bool(valid),
        "deadlock": bool(deadlock),
        "reason": reason,
        "policy": policy,
        "env_seed": int(env_seed),
        "n_orders": len(stream),
        "n_requested": int(n_requested),
        "deliveries": int(delivered),
        "steps": int(steps),
        # PRIMARY fleet metric
        "steps_per_delivery": (steps / delivered) if delivered else None,
        "throughput_per_1k": (1000.0 * delivered / steps) if steps else None,
        "mean_order_completion": (float(np.mean(completions))
                                  if completions else None),
        "orders_completed": len(last_done),
        "deferrals": int(deferrals),
        "perturbations": int(fleet.perturbations),
        "replans": int(fleet.replans),
        "env_assert_events": int(env_assert_events),
        "diagnostic": diagnostic,
    }
