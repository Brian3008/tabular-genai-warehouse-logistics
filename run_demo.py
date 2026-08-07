"""
demo_for_locus.py
=================
A LIVE demonstration of the Gen AI, end to end, in ~2 minutes:

  [1] load the trained model (ctgan_v3.pkl)
  [2] generate fresh synthetic orders LIVE - unconditional
      and conditioned on basket size
  [3] verify integrity of the live sample (legal values,
      exact conditioning) - ground truth derived from the
      training data, never assumed
  [4] show real vs generated rows side by side
  [5] show the statistical match on held-out real data
  [6] the WAREHOUSE CONNECTION: measure the demand geometry
      of the live sample (aisle mix / concentration / travel)
      against the noise bars recorded in demand_geometry.json
      - the same measured bars, no thresholds guessed here

WHAT THIS DEMO IS AND IS NOT
----------------------------
It IS a live proof that the generator works, produces only
legal warehouse values, conditions exactly, and matches real
statistics - plus an honest display of where it fails.

It is NOT a robot simulator. The fleet connection is made at
the demand-geometry level (the three quantities any fleet
simulator consumes). The recorded finding this demo replays:
synthetic data exaggerates the real aisle-mix effect and
FABRICATES a travel effect that does not exist in reality
(data/real_fleet_effect.json). Show that plainly - it is the
project's contribution, not a weakness of the demo.

Output: results/demo_for_locus.png + console walkthrough.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # save to file, never block on a window
import matplotlib.pyplot as plt
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/demo_run', exist_ok=True)

SEED = 42
# Pass --free to leave the run unpinned and show draw-to-draw variation.
PINNED = "--free" not in sys.argv
if PINNED:
    random.seed(SEED)
    np.random.seed(SEED)
    import torch as _t                       # sampling RNG lives inside the model
    _t.manual_seed(SEED)
    if _t.cuda.is_available():
        _t.cuda.manual_seed_all(SEED)
    print("[pinned run - identical every time. Use --free for a fresh draw.]")
else:
    print("[FREE run - unpinned, numbers will move between runs.]")

COND_COL = 'order_size_grp'
SMALL_MAX, LARGE_MIN = 10, 14      # the screened buckets
BASE_COLS = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_reorder', 'is_early_in_cart', 'days_since_prior_order',
]
N_FREE = 20000                     # unconditional live sample
N_COND = 10000                     # per conditioned class

# project figure convention: real = steelblue, synthetic =
# seagreen, everywhere, in fixed assignment. Identity is also
# carried by linestyle (real solid, synth dashed) so it never
# rests on color alone.
C_REAL, C_SYN = 'steelblue', 'seagreen'

print("=" * 68)
print("GEN AI LIVE DEMO - Tabular Gen-AI for Warehouse Logistics")
print("=" * 68)

# ══════════════════════════════════════════
# [1] LOAD MODEL + REAL DATA (held out)
# ══════════════════════════════════════════
print("\n[1] Loading the trained Gen AI (data/ctgan_v3.pkl) ...")
t0 = time.time()
model = CTGANSynthesizer.load('data/ctgan_v3.pkl')
print(f"    loaded in {time.time() - t0:.1f}s")

real_eval = pd.read_csv('data/v3_eval.csv')
comp = pd.read_csv('data/v3_compare.csv')
train = pd.read_csv('data/v3_train.csv')
tr_ids = set(pd.read_csv('data/v3_train_order_ids.csv')['order_id'])
assert len(set(real_eval['order_id']) & tr_ids) == 0, \
    "FATAL: eval contains training orders"
assert len(set(comp['order_id']) & tr_ids) == 0, \
    "FATAL: compare contains training orders"
print("    [ASSERTED] the real data shown below is HELD OUT -")
print("               the model has never seen these orders")

# ══════════════════════════════════════════
# [2] GENERATE LIVE
# ══════════════════════════════════════════
print(f"\n[2] Generating {N_FREE:,} fresh orders LIVE "
      f"(unconditional) ...")
t0 = time.time()
synth = model.sample(num_rows=N_FREE)
print(f"    generated in {time.time() - t0:.1f}s")

print(f"\n    Now conditioned: {N_COND:,} small-basket + "
      f"{N_COND:,} large-basket orders ...")
t0 = time.time()
gen = {}
for grp in ['small', 'large']:
    gen[grp] = model.sample_from_conditions(
        conditions=[Condition(num_rows=N_COND,
                              column_values={COND_COL: grp})])
print(f"    generated in {time.time() - t0:.1f}s")

# ══════════════════════════════════════════
# [3] LIVE INTEGRITY CHECK
# ══════════════════════════════════════════
print("\n[3] Integrity of the LIVE sample "
      "(ground truth = training data, not assumptions):")

legal = {c: set(pd.to_numeric(train[c], errors='coerce')
                .dropna().unique())
         for c in BASE_COLS if c != 'days_since_prior_order'}
bad_total = 0
for c, legal_set in legal.items():
    v = pd.to_numeric(synth[c], errors='coerce')
    n_bad = int((~v.isin(legal_set)).sum())
    bad_total += n_bad
print(f"    illegal values in {N_FREE:,} live rows: {bad_total}"
      f"  {'[OK]' if bad_total == 0 else '[FAIL]'}")

leaks = sum(int((gen[g][COND_COL] != g).sum())
            for g in ['small', 'large'])
print(f"    conditioning leaks in {2 * N_COND:,} rows:  {leaks}"
      f"  {'[OK - exact]' if leaks == 0 else '[FAIL]'}")

# ══════════════════════════════════════════
# [4] SIDE BY SIDE ROWS
# ══════════════════════════════════════════
print("\n[4] What the data looks like - 5 rows each:")
print("\n    REAL (held-out):")
print(real_eval[BASE_COLS].sample(5, random_state=SEED)
      .to_string(index=False).replace('\n', '\n    '))
print("\n    GENERATED (just now):")
print(synth[BASE_COLS].sample(5, random_state=SEED)
      .to_string(index=False).replace('\n', '\n    '))

# ══════════════════════════════════════════
# [5] STATISTICAL MATCH
# ══════════════════════════════════════════
print("\n[5] Statistical match (live sample vs held-out real):")
print(f"    {'column':<26}{'real':>9}{'generated':>11}")
print("    " + "-" * 46)
for c in BASE_COLS:
    r = pd.to_numeric(real_eval[c], errors='coerce').mean()
    s = pd.to_numeric(synth[c], errors='coerce').mean()
    print(f"    {c:<26}{r:>9.3f}{s:>11.3f}")
print("    (is_reorder and is_early_in_cart show the two known,")
print("     documented marginal errors - reported, not hidden)")

# ══════════════════════════════════════════
# [6] THE WAREHOUSE CONNECTION - live geometry
# ══════════════════════════════════════════
print("\n[6] WAREHOUSE CONNECTION - demand geometry of the live")
print("    sample, judged against the RECORDED noise bars")
print("    (data/demand_geometry.json - measured from 40")
print("    real-vs-real draws, nothing guessed here):")

geo = json.load(open('data/demand_geometry.json'))
N_GEO = geo['n_common']            # bars are size-dependent -
                                   # must measure at the same n

def aisle_tvd(a, b):
    ca = pd.Series(a).value_counts(normalize=True)
    cb = pd.Series(b).value_counts(normalize=True)
    idx = sorted(set(ca.index) | set(cb.index))
    ca = ca.reindex(idx, fill_value=0)
    cb = cb.reindex(idx, fill_value=0)
    return float(0.5 * np.abs(ca - cb).sum())

def gini(a):
    c = np.sort(pd.Series(a).value_counts().values)
    n = len(c)
    cum = np.cumsum(c)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def expected_travel(a, reps=30, seed=SEED):
    a = np.asarray(a, dtype=float)
    rng = np.random.RandomState(seed)
    return float(np.mean([
        np.mean(np.abs(np.diff(rng.permutation(a))))
        for _ in range(reps)]))

comp = comp.copy()
osz = comp.groupby('order_id').size()
comp['n_items'] = comp['order_id'].map(osz)
comp[COND_COL] = np.where(
    comp['n_items'] <= SMALL_MAX, 'small',
    np.where(comp['n_items'] >= LARGE_MIN, 'large', 'mid'))

rs = np.random.RandomState(SEED + 1)
live = {}
for g in ['small', 'large']:
    R_pool = comp.loc[comp[COND_COL] == g, 'aisle_id'].values
    S_pool = pd.to_numeric(gen[g]['aisle_id'],
                           errors='coerce').dropna().astype(int).values
    n = min(N_GEO, len(R_pool), len(S_pool))
    R = rs.choice(R_pool, size=n, replace=False)
    S = rs.choice(S_pool, size=n, replace=False)
    f = geo['floor'][g]
    rec = geo['results'][g]        # the recorded run of record
    tv = aisle_tvd(R, S)
    ge = abs(gini(S) - gini(R))
    tr_r, tr_s = expected_travel(R), expected_travel(S)
    te = abs(tr_s - tr_r) / tr_r if tr_r else 0
    live[g] = {'R': R, 'S': S, 'tvd': tv, 'gini_err': ge,
               'travel_real': tr_r, 'travel_syn': tr_s,
               'travel_err': te, 'n': n}
    print(f"\n    {g.upper()} baskets (n={n:,})"
          f"   [live | recorded run]:")
    print(f"      aisle-mix TVD  {tv:.4f} | {rec['tvd']:.4f}"
          f"  vs bar {f['tvd_bar']:.4f}"
          f"   -> {'OK' if tv <= f['tvd_bar'] else 'DEVIATES'}")
    print(f"      Gini error     {ge:.4f} | {rec['gini_err']:.4f}"
          f"  vs bar {f['gini_bar']:.4f}"
          f"   -> {'OK' if ge <= f['gini_bar'] else 'DEVIATES'}")
    print(f"      travel error   {te:>6.2%} | {rec['travel_err']:.2%}"
          f"  vs bar {f['travel_bar']:.2%}"
          f"   -> {'OK' if te <= f['travel_bar'] else 'DEVIATES'}")

gap_real = abs(live['large']['travel_real']
               - live['small']['travel_real'])
gap_syn = abs(live['large']['travel_syn']
              - live['small']['travel_syn'])
rec_gap_real = geo['travel_gap_real']
rec_gap_syn = geo['travel_gap_syn']
fleet = json.load(open('data/real_fleet_effect.json'))
noise_bar = fleet['bar']['travel']
print(f"\n    small-vs-large TRAVEL GAP  [live | recorded run]:")
print(f"      real       {gap_real:.2f} | {rec_gap_real:.2f} aisles")
print(f"      generated  {gap_syn:.2f} | {rec_gap_syn:.2f} aisles")
print(f"      noise bar  {noise_bar:.2f} aisles (95th pct of the")
print("                 real-vs-real null, real_fleet_effect.json)")
print("      (live draws vary - the recorded run, the artifact of")
print("       record, is demand_geometry.json. The real gap sits")
print("       under the noise bar: the fixture-verified real-data")
print("       test proved the real travel effect is NOISE. The")
print("       generated gap is FABRICATED - that is the project's")
print("       headline finding.)")

# ══════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle('Gen AI live demo - real vs generated orders '
             '(real data is held out; synthetic generated live)',
             fontsize=13, fontweight='bold')

# (a) aisle demand profile
ax = axes[0][0]
cr = pd.to_numeric(real_eval['aisle_id'], errors='coerce') \
       .value_counts(normalize=True).sort_index()
cs = pd.to_numeric(synth['aisle_id'], errors='coerce') \
       .value_counts(normalize=True).sort_index()
ax.plot(cr.index, cr.values, lw=1.6, color=C_REAL,
        label='real (held-out)')
ax.plot(cs.index, cs.values, lw=1.6, ls='--', color=C_SYN,
        label='generated (live)')
ax.set_title('Which shelves get hit - aisle demand profile')
ax.set_xlabel('aisle')
ax.set_ylabel('share of picks')
ax.legend(fontsize=9)

# (b) hour of day
ax = axes[0][1]
hr = pd.to_numeric(real_eval['order_hour_of_day'],
                   errors='coerce').value_counts(
    normalize=True).sort_index()
hs = pd.to_numeric(synth['order_hour_of_day'],
                   errors='coerce').value_counts(
    normalize=True).sort_index()
ax.plot(hr.index, hr.values, lw=1.6, color=C_REAL,
        label='real (held-out)')
ax.plot(hs.index, hs.values, lw=1.6, ls='--', color=C_SYN,
        label='generated (live)')
ax.set_title('When orders arrive - hour of day')
ax.set_xlabel('hour')
ax.set_ylabel('share of orders')
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=9)

# (c) day of week
ax = axes[1][0]
dr = pd.to_numeric(real_eval['order_dow'], errors='coerce') \
       .value_counts(normalize=True).sort_index()
ds = pd.to_numeric(synth['order_dow'], errors='coerce') \
       .value_counts(normalize=True).sort_index()
x = np.arange(7)
w = 0.38
ax.bar(x - w / 2, [dr.get(d, 0) for d in range(7)], w,
       color=C_REAL, label='real (held-out)')
ax.bar(x + w / 2, [ds.get(d, 0) for d in range(7)], w,
       color=C_SYN, hatch='//', label='generated (live)')
ax.set_title('Orders by day of week')
ax.set_xlabel('day (0 = Sunday in Instacart)')
ax.set_ylabel('share of orders')
ax.set_xticks(x)
ax.legend(fontsize=9)

# (d) the warehouse verdict - plot the GAP itself.
# Absolute travel is ~43 aisles for every group, so plotting
# raw travel makes a 0.3-vs-1.3 gap invisible on a zero-based
# axis. The message IS the gap, so the gap is the mark - with
# the MEASURED noise bar (95th pct of the real-data null,
# data/real_fleet_effect.json) as the reference line. Live
# gaps vary draw to draw, so the recorded run of record
# (demand_geometry.json) is annotated on each bar.
ax = axes[1][1]
x = np.arange(2)
vals = [gap_real, gap_syn]
rec_vals = [rec_gap_real, rec_gap_syn]
bars = ax.bar(x, vals, 0.5, color=[C_REAL, C_SYN])
bars[1].set_hatch('//')
for j, (v, rv) in enumerate(zip(vals, rec_vals)):
    ax.annotate(f'{v:.2f}', (j, v), xytext=(0, 4),
                textcoords='offset points', ha='center',
                va='bottom', fontsize=10, fontweight='bold')
    ax.annotate(f'recorded run: {rv:.2f}', (j, v),
                xytext=(0, 19), textcoords='offset points',
                ha='center', va='bottom', fontsize=8,
                color='dimgray')
ax.axhline(noise_bar, ls='--', color='black', lw=1.2)
ax.text(0.5, noise_bar + 0.02,
        f'measured noise bar ({noise_bar:.2f})\n'
        '95th pct of real-vs-real null',
        fontsize=8, ha='center', va='bottom')
ax.set_xticks(x)
ax.set_xticklabels(['REAL data\n(held-out)',
                    'GENERATED data\n(live)'])
ax.set_ylabel('small-vs-large travel gap (aisles)')
ax.set_ylim(0, max(gap_syn, noise_bar) * 1.4)
ax.set_title('THE FINDING: the real travel gap sits BELOW the\n'
             'noise bar (no real effect) - the generated gap is '
             'fabricated')

plt.tight_layout()
plt.savefig('results/demo_run/demo.png', dpi=150,
            bbox_inches='tight')
print("\n" + "=" * 68)
print("saved results/demo_run/demo.png")
print("=" * 68)
print("""
THE STORY TO TELL WITH THIS DEMO
  1. The generator produces realistic orders LIVE - only
     legal values, exact conditioning, statistics matching
     real held-out data (panels a-c).
  2. It passes every standard synthetic-data check - quality
     92.0%, ML efficacy 89.2% +/- 0.7%, privacy DCR 1.08 -
     the numbers that would normally get it declared "safe".
  3. But panel (d) is the contribution: it invents a travel
     effect that does not exist in reality - proven by a
     fixture-verified, real-data-only test. Standard
     synthetic-data metrics do NOT catch this. Warehouse
     decisions need a demand-geometry check before trusting
     synthetic data. That is what this project gives Locus.
""")
