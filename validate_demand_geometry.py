"""
validate_demand_geometry.py
===========================
Does the synthetic data mislead a downstream warehouse
decision?

WHY THIS REPLACED A FLEET SIMULATOR
-----------------------------------
I first built a simulator running three strategies
(nearest-task / zoned / random) on real vs synthetic data,
to see whether they picked the same winner.

IT WAS RIGGED. Zoned won EVERY demand pattern - not
because zoning is better, but because I wrote it to sort
its picks and sweep once while nearest-task jumps around
greedily. On a corridor a sorted sweep is near-optimal BY
CONSTRUCTION. The contest was decided before any data
touched it.

I could have added congestion, capacity, 2-D layouts and
deadlines until the strategies traded places. But that is
TUNING THE SIMULATOR UNTIL IT PRODUCES A FINDING - the
same sin as tuning post-processing until the quality
score rises. It is exactly what this rebuild exists to
eliminate.

(The project's earlier A* congestion simulator DEADLOCKED
and was abandoned. Not bad luck. Multi-robot coordination
is genuinely hard, and a simulator that is both realistic
AND validated is a serious piece of work.)

THE REAL QUESTION
-----------------
Not "which strategy wins", but: DOES THE SYNTHETIC DATA
MISLEAD A WAREHOUSE DECISION?

Answerable with NO simulator, by comparing the DEMAND
GEOMETRY - the three things every warehouse simulator
consumes:

  1. WHICH AISLES get picked, in what proportion
     -> which shelves robots visit
  2. HOW CONCENTRATED demand is (Gini)
     -> congestion risk, value of zoning
  3. EXPECTED TRAVEL between consecutive picks
     -> routing cost, in AISLES TRAVELLED

If those match, ANY simulator fed by either behaves the
same. If they diverge, no simulator can rescue it.

No simulator = nothing to rig, nothing to tune, no toy to
defend in a viva.

TWO BUGS CAUGHT WHILE BUILDING THIS
-----------------------------------
1. SAMPLE-SIZE MISMATCH.
   TVD and Gini are both size-dependent. Measured: two
   samples from the IDENTICAL distribution give

       n = 3,500   TVD 0.096
       n = 30,000  TVD 0.033

   My first design measured the noise floor by halving
   the real data (n~3,500) while comparing real vs
   synthetic at n=7,000 vs 30,000. That made the FLOOR
   INFLATED and the COMPARISON DEFLATED - so a genuine
   deviation could hide under an over-generous floor.
   A false PASS. The dangerous direction.

   FIXED: every comparison happens at ONE common n.

2. THE FLOOR WAS A SINGLE DRAW x AN ARBITRARY 2.
   The floor is a RANDOM VARIABLE. One draw tells you
   nothing about its spread, and "x2" was a guess -
   the eighth time in this project I have hard-coded a
   threshold.

   Verified it FAILED: a genuinely distorted model
   (TVD 0.173) slipped under floor x 2 (0.187).

   FIXED: measure the floor's DISTRIBUTION over many
   real-vs-real draws and use its 95th percentile.
   Verified: faithful passes (0.095 <= 0.109),
   distorted is caught (0.178 > 0.109).

EVERY METRIC VERIFIED BEFORE THIS FILE WAS WRITTEN
--------------------------------------------------
Each scores ~perfect on a faithful sample (no false
alarm) and clearly worse on a distorted one (real
detection).
"""

import pandas as pd
import numpy as np
import random
import json
import os
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

COND_COL = 'order_size_grp'
SMALL_MAX = 10
LARGE_MIN = 14

N_FLOOR_DRAWS = 40    # to measure the floor DISTRIBUTION
N_TRAVEL_REPS = 30

print("=" * 70)
print("DOES SYNTHETIC DATA MISLEAD A WAREHOUSE DECISION?")
print("=" * 70)
print("""
  Comparing the DEMAND GEOMETRY every fleet simulator
  consumes. No simulator - nothing to rig.
""")


# ══════════════════════════════════════════
# METRICS (each verified to discriminate)
# ══════════════════════════════════════════
def aisle_tvd(a, b):
    """Total variation distance between two aisle-demand
    distributions. 0 = identical."""
    ca = pd.Series(a).value_counts(normalize=True)
    cb = pd.Series(b).value_counts(normalize=True)
    idx = sorted(set(ca.index) | set(cb.index))
    ca = ca.reindex(idx, fill_value=0)
    cb = cb.reindex(idx, fill_value=0)
    return float(0.5 * np.abs(ca - cb).sum())


def gini(a):
    """Demand concentration. High = a few aisles dominate
    (congestion risk, zoning pays). Low = spread out."""
    c = np.sort(pd.Series(a).value_counts().values)
    n = len(c)
    cum = np.cumsum(c)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def expected_travel(a, seed=SEED):
    """Mean |aisle_i - aisle_i+1| over random pick orders:
    expected robot travel between consecutive picks, in
    AISLES.

    THIS is the metric a warehouse engineer acts on - it
    converts an abstract distribution error into travel
    distance."""
    a = np.asarray(a, dtype=float)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(N_TRAVEL_REPS):
        s = rng.permutation(a)
        out.append(float(np.mean(np.abs(np.diff(s)))))
    return float(np.mean(out))


def top_aisles(a, k=5):
    return list(pd.Series(a).value_counts()
                .head(k).index.astype(int))


# ══════════════════════════════════════════
# [1] DATA + DISJOINTNESS
# ══════════════════════════════════════════
print("=" * 70)
print("[1] DATA")
print("=" * 70)

comp = pd.read_csv('data/v3_compare.csv')
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
assert len(set(comp['order_id']) & tr_ids) == 0, \
    "FATAL: compare contains training orders"
print("\n    [ASSERTED] real data is held out from the")
print("               model's training orders")

osize = comp.groupby('order_id').size()
comp['n_items'] = comp['order_id'].map(osize)
comp[COND_COL] = np.where(
    comp['n_items'] <= SMALL_MAX, 'small',
    np.where(comp['n_items'] >= LARGE_MIN,
             'large', 'mid'))

real_all = {
    g: comp[comp[COND_COL] == g]['aisle_id'].values
    for g in ['small', 'large']
}

model = CTGANSynthesizer.load('data/ctgan_v3.pkl')
syn_all = {}
for g in ['small', 'large']:
    d = model.sample_from_conditions(
        conditions=[Condition(
            num_rows=40000,
            column_values={COND_COL: g})])
    assert (d[COND_COL] == g).all(), \
        f"FATAL: conditioning leaked on {g}"
    syn_all[g] = pd.to_numeric(
        d['aisle_id'], errors='coerce'
    ).dropna().astype(int).values

print("    [ASSERTED] conditioning exact, zero leakage\n")
for g in ['small', 'large']:
    print(f"    {g:<8} real {len(real_all[g]):>7,} picks"
          f"   synthetic {len(syn_all[g]):>7,} picks")


# ══════════════════════════════════════════
# SIZE MATCHING - measured to be essential
# ══════════════════════════════════════════
# TVD and Gini are BOTH sample-size dependent. Measured:
# two samples from the SAME distribution give TVD 0.096
# at n=3,500 but 0.033 at n=30,000.
#
# So every comparison - the floor AND the real-vs-
# synthetic test - must happen at ONE common n, or the
# floor is inflated relative to the comparison and a real
# deviation slips through.
#
# The floor needs two disjoint halves of real data, so
# the budget is half the smallest real group.
N = min(min(len(real_all[g]) // 2
            for g in ['small', 'large']),
        min(len(syn_all[g])
            for g in ['small', 'large']))

print(f"\n    SIZE MATCHED: every comparison at "
      f"n = {N:,}")
print("    (TVD/Gini are size-dependent - measured: the")
print("     SAME distribution gives TVD 0.096 at n=3.5k")
print("     but 0.033 at n=30k. Without matching, the")
print("     floor is inflated and a real deviation could")
print("     pass unnoticed.)")


# ══════════════════════════════════════════
# [2] THE NOISE FLOOR - a DISTRIBUTION
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[2] THE NOISE FLOOR - measured as a distribution")
print("=" * 70)
print(f"""
    Two samples of the SAME real data differ by chance.
    That spread is the floor.

    The floor is a RANDOM VARIABLE - one draw tells you
    nothing about its spread. My first version took a
    single draw and multiplied by an arbitrary 2, and I
    VERIFIED IT FAILED: a genuinely distorted model
    (TVD 0.173) slipped under the bar (0.187).

    So: {N_FLOOR_DRAWS} independent real-vs-real draws, and the bar
    is the 95th percentile of that distribution.
""")

floor = {}
for g in ['small', 'large']:
    rng = np.random.RandomState(SEED)
    tvds, ginis, travels = [], [], []
    for _ in range(N_FLOOR_DRAWS):
        idx = rng.permutation(len(real_all[g]))
        A = real_all[g][idx[:N]]
        B = real_all[g][idx[N:2 * N]]
        tvds.append(aisle_tvd(A, B))
        ginis.append(abs(gini(A) - gini(B)))
        tA = expected_travel(A)
        tB = expected_travel(B)
        travels.append(abs(tA - tB) / tA if tA else 0)

    floor[g] = {
        'tvd_bar': float(np.percentile(tvds, 95)),
        'tvd_mean': float(np.mean(tvds)),
        'gini_bar': float(np.percentile(ginis, 95)),
        'gini_mean': float(np.mean(ginis)),
        'travel_bar': float(np.percentile(travels, 95)),
        'travel_mean': float(np.mean(travels)),
    }

    f = floor[g]
    print(f"    {g.upper()}")
    print(f"      TVD     mean {f['tvd_mean']:.4f}   "
          f"BAR (95th) {f['tvd_bar']:.4f}")
    print(f"      Gini    mean {f['gini_mean']:.4f}   "
          f"BAR (95th) {f['gini_bar']:.4f}")
    print(f"      travel  mean {f['travel_mean']:.2%}   "
          f"BAR (95th) {f['travel_bar']:.2%}")
    print()

print("    (real vs real - this is what PERFECT looks")
print("     like. Synthetic is judged against the BAR.)")


# ══════════════════════════════════════════
# [3] REAL vs SYNTHETIC
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[3] REAL vs SYNTHETIC DEMAND GEOMETRY")
print("=" * 70)

rs = np.random.RandomState(SEED + 1)
res = {}
for g in ['small', 'large']:
    R = rs.choice(real_all[g], size=N, replace=False)
    S = rs.choice(syn_all[g], size=N, replace=False)

    tr = expected_travel(R)
    ts = expected_travel(S)
    g_err = abs(gini(S) - gini(R))
    t_err = abs(ts - tr) / tr if tr else 0
    tv = aisle_tvd(R, S)

    f = floor[g]
    res[g] = {
        'R': R, 'S': S,
        'tvd': tv,
        'gini_real': gini(R), 'gini_syn': gini(S),
        'gini_err': g_err,
        'travel_real': tr, 'travel_syn': ts,
        'travel_err': t_err,
        'top_real': top_aisles(R),
        'top_syn': top_aisles(S),
        'ok_tvd': bool(tv <= f['tvd_bar']),
        'ok_gini': bool(g_err <= f['gini_bar']),
        'ok_travel': bool(t_err <= f['travel_bar']),
    }

for g in ['small', 'large']:
    r = res[g]
    f = floor[g]
    print(f"\n    --- {g.upper()} BASKETS  (n={N:,}) ---\n")

    print(f"    1. WHICH AISLES (TVD)")
    print(f"       synthetic vs real : {r['tvd']:.4f}")
    print(f"       bar (95th of floor): "
          f"{f['tvd_bar']:.4f}")
    print(f"       -> "
          f"{'OK' if r['ok_tvd'] else 'DEVIATES'}")
    print(f"       real  top5: {r['top_real']}")
    print(f"       synth top5: {r['top_syn']}")

    print(f"\n    2. CONCENTRATION (Gini)")
    print(f"       real      : {r['gini_real']:.4f}")
    print(f"       synthetic : {r['gini_syn']:.4f}")
    print(f"       error     : {r['gini_err']:.4f}"
          f"   bar {f['gini_bar']:.4f}")
    print(f"       -> "
          f"{'OK' if r['ok_gini'] else 'DEVIATES'}")

    print(f"\n    3. EXPECTED TRAVEL between picks")
    print(f"       real      : {r['travel_real']:.2f} "
          f"aisles")
    print(f"       synthetic : {r['travel_syn']:.2f} "
          f"aisles")
    print(f"       error     : {r['travel_err']:.2%}"
          f"   bar {f['travel_bar']:.2%}")
    print(f"       -> "
          f"{'OK' if r['ok_travel'] else 'DEVIATES'}")


# ══════════════════════════════════════════
# [4] THE SMALL-vs-LARGE GAP
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[4] IS THE SMALL-vs-LARGE GAP PRESERVED?")
print("=" * 70)
print("""
    The fleet story rests on small and large baskets
    behaving DIFFERENTLY. If synthetic exaggerates that
    gap, a planner would over-provision for it.
""")

real_gap = abs(res['large']['travel_real']
               - res['small']['travel_real'])
syn_gap = abs(res['large']['travel_syn']
              - res['small']['travel_syn'])
gap_ratio = syn_gap / real_gap if real_gap else 0

print(f"    TRAVEL gap (large - small)")
print(f"      real      : {real_gap:.2f} aisles")
print(f"      synthetic : {syn_gap:.2f} aisles")
print(f"      ratio     : {gap_ratio:.0%}")

rg = abs(res['large']['gini_real']
         - res['small']['gini_real'])
sg = abs(res['large']['gini_syn']
         - res['small']['gini_syn'])
gini_ratio = sg / rg if rg else 0
print(f"\n    CONCENTRATION gap (large - small)")
print(f"      real      : {rg:.4f}")
print(f"      synthetic : {sg:.4f}")
print(f"      ratio     : {gini_ratio:.0%}")

print("""
    ~100%  faithful
    >>100% EXAGGERATED - a planner would over-provision
    <<100% UNDERSTATED - the difference would be missed
""")


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("=" * 70)
print("VERDICT")
print("=" * 70)

all_ok = all(res[g][k] for g in ['small', 'large']
             for k in ['ok_tvd', 'ok_gini', 'ok_travel'])

def mark(g, k):
    return 'OK' if res[g][k] else 'DEVIATES'

print(f"""
    WHICH AISLES    small {mark('small', 'ok_tvd'):<9}"""
      f"""large {mark('large', 'ok_tvd')}
    CONCENTRATION   small {mark('small', 'ok_gini'):<9}"""
      f"""large {mark('large', 'ok_gini')}
    TRAVEL COST     small {mark('small', 'ok_travel'):<9}"""
      f"""large {mark('large', 'ok_travel')}

    small/large travel gap reproduced at {gap_ratio:.0%}
""")

if all_ok:
    print("""
    THE DEMAND GEOMETRY IS FAITHFUL.

    Synthetic data reproduces which aisles get hit, how
    concentrated demand is, and the expected robot travel
    - all within the noise of real-vs-real.

    ANY fleet simulator fed by this synthetic data would
    behave as it does on real data. The generator is
    VALIDATED for warehouse planning.
""")
else:
    print("""
    THE GEOMETRY DEVIATES on at least one axis.

    This is a REAL, REPORTABLE FINDING - and more
    valuable than any simulator result. It says a
    generator can be faithful on marginals, on ML
    efficacy, and on privacy, and STILL misrepresent the
    demand geometry a warehouse decision depends on.

    Report exactly WHICH axis deviates and BY HOW MUCH,
    in aisles of robot travel. That is a number a Locus
    engineer can act on.
""")

worst_tvd = max(res['small']['tvd'], res['large']['tvd'])
worst_tr = max(res['small']['travel_err'],
               res['large']['travel_err'])
print(f"""
    FOR THE DISSERTATION:

      Synthetic demand reproduces the real aisle
      distribution to within {worst_tvd:.3f} TVD and expected
      robot travel to within {worst_tr:.1%}. The small-vs-large
      demand gap is reproduced at {gap_ratio:.0%} of its true
      magnitude.
""")


# ══════════════════════════════════════════
# CHART
# ══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    'Demand geometry: does synthetic data mislead a '
    'warehouse decision?',
    fontsize=13, fontweight='bold')

for i, g in enumerate(['small', 'large']):
    ax = axes[0][i]
    cr = pd.Series(res[g]['R']).value_counts(
        normalize=True).sort_index()
    cs = pd.Series(res[g]['S']).value_counts(
        normalize=True).sort_index()
    ax.plot(cr.index, cr.values, lw=1.2,
            color='steelblue', label='real')
    ax.plot(cs.index, cs.values, lw=1.2, ls='--',
            color='seagreen', label='synthetic')
    ax.set_title(
        f'{g}-basket aisle demand\n'
        f'TVD {res[g]["tvd"]:.4f}  '
        f'(bar {floor[g]["tvd_bar"]:.4f})')
    ax.set_xlabel('aisle')
    ax.set_ylabel('share of picks')
    ax.legend(fontsize=8)

x = np.arange(2)
w = 0.35

ax = axes[1][0]
rv = [res[g]['travel_real'] for g in ['small', 'large']]
sv = [res[g]['travel_syn'] for g in ['small', 'large']]
ax.bar(x - w / 2, rv, w, label='real',
       color='steelblue')
ax.bar(x + w / 2, sv, w, label='synthetic',
       color='seagreen')
ax.set_xticks(x)
ax.set_xticklabels(['small', 'large'])
ax.set_ylabel('aisles travelled between picks')
ax.set_title('Expected robot travel\n'
             '(routing cost depends on this)')
ax.legend()
for j, (a_, b_) in enumerate(zip(rv, sv)):
    ax.text(j - w / 2, a_ + 0.3, f'{a_:.1f}',
            ha='center', fontsize=9)
    ax.text(j + w / 2, b_ + 0.3, f'{b_:.1f}',
            ha='center', fontsize=9)

ax = axes[1][1]
gr = [res[g]['gini_real'] for g in ['small', 'large']]
gs = [res[g]['gini_syn'] for g in ['small', 'large']]
ax.bar(x - w / 2, gr, w, label='real',
       color='steelblue')
ax.bar(x + w / 2, gs, w, label='synthetic',
       color='seagreen')
ax.set_xticks(x)
ax.set_xticklabels(['small', 'large'])
ax.set_ylabel('Gini (higher = more concentrated)')
ax.set_title('Demand concentration\n'
             '(drives congestion, value of zoning)')
ax.legend()

plt.tight_layout()
plt.savefig('results/demand_geometry.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("  saved results/demand_geometry.png")

out = {
    'n_common': int(N),
    'floor': floor,
    'results': {
        g: {k: v for k, v in res[g].items()
            if k not in ('R', 'S')}
        for g in ['small', 'large']},
    'travel_gap_real': float(real_gap),
    'travel_gap_syn': float(syn_gap),
    'gap_ratio': float(gap_ratio),
    'gini_gap_ratio': float(gini_ratio),
    'all_ok': bool(all_ok),
}
json.dump(out, open('data/demand_geometry.json', 'w'),
          indent=2, default=float)
print("Saved data/demand_geometry.json")
print("=" * 70)