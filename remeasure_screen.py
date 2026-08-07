"""
remeasure_screen.py
===================
FIXES A BUG IN screen_thresholds.py

THE BUG
-------
screen_thresholds.py computed:

    REAL tail TVD      on rows where is_early_in_cart == 1
                       (the first-3-items size control)

    SYNTHETIC tail TVD on ALL generated rows
                       (no control at all)

Those are DIFFERENT POPULATIONS. So "recovery = synth /
real" was comparing apples to oranges, and the 120-128%
figures - and the ranking of candidates built on them -
are not trustworthy.

The conclusion that all three learned a substantial,
above-noise shift still stands. The RANKING does not.

WHAT THIS SCRIPT DOES
---------------------
Reloads the three already-trained models (no retraining)
and re-measures with a LIKE-FOR-LIKE comparison.

Because the populations must match, we report BOTH views:

  (A) ALL ROWS
      real: all training rows   vs  synth: all generated rows
      -> same population on both sides. Clean.
      -> but includes the arithmetic artefact (big baskets
         mechanically touch more aisles)

  (B) EARLY-CART ROWS
      real: is_early_in_cart==1  vs  synth: is_early_in_cart==1
      -> same population on both sides. Clean.
      -> removes the arithmetic artefact
      -> ONLY VALID IF the model generates is_early_in_cart
         at roughly the real rate. We check that first.

PRE-CHECK
---------
is_early_in_cart is one of the six trained columns, so
CTGAN generates it. If the synthetic rate is way off the
real ~0.28, then view (B) is measuring a subset that is
not "the first 3 items" of anything, and we must discard
it and rely on view (A).

I never checked this before. Checking now.

WHAT GOOD LOOKS LIKE
--------------------
  recovery near 100%  -> model reproduces the real shift
                         faithfully
  recovery >> 100%    -> model EXAGGERATES the difference.
                         That is a fidelity error, not a
                         bonus. Synthetic data overstating
                         a demand gap would mislead a
                         fleet planner.
  recovery << 100%    -> model under-learns the shift

We also compare the AISLE COMPOSITION, not just the
magnitude. A model can move the mix by the right amount
in the wrong direction. Magnitude alone would hide that.

No retraining. Minutes, not hours.
"""

import pandas as pd
import numpy as np
import random
import torch
import json
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_GEN = 30000
DOMINANT = [24, 83]
NOISE_FLOOR = 0.0357
COND_COL = 'order_size_grp'

CANDIDATES = [
    (6,  12, "sharp 33/67"),
    (9,  12, "55/70"),
    (10, 14, "60/75"),
]

print("=" * 68)
print("RE-MEASURE - fixing the population mismatch")
print("=" * 68)
print("""
screen_thresholds.py compared real TVD (early-cart
rows only) against synthetic TVD (all rows). Different
populations. The recovery numbers were invalid.

Re-measuring like-for-like. No retraining.
""")

# ══════════════════════════════════════════
# LOAD TRAINING DATA + VERIFY DISJOINTNESS
# ══════════════════════════════════════════
print("[1] Loading v3_train.csv ...")
train = pd.read_csv('data/v3_train.csv')

train_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
eval_ids = set(pd.read_csv(
    'data/v3_eval.csv')['order_id'])
assert len(train_ids & eval_ids) == 0, \
    "TRAIN/EVAL OVERLAP"
assert set(train['order_id']) <= train_ids, \
    "train contains non-training orders"
print(f"    rows: {len(train):,}   [OK] disjoint")

osize = train.groupby('order_id').size()
train['n_items'] = train['order_id'].map(osize)

real_early_rate = train['is_early_in_cart'].mean()
print(f"\n    REAL is_early_in_cart rate: "
      f"{real_early_rate:.3f}")


# ══════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════
def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


def tail_shift(d_small, d_large):
    """Aisle-mix TVD excluding the two dominant
    aisles, plus the top-5 lists so we can check
    COMPOSITION and not just magnitude."""
    if len(d_small) == 0 or len(d_large) == 0:
        return None
    sm = d_small['aisle_id'].value_counts(
        normalize=True)
    lg = d_large['aisle_id'].value_counts(
        normalize=True)
    smt = sm[~sm.index.isin(DOMINANT)]
    lgt = lg[~lg.index.isin(DOMINANT)]
    if smt.sum() == 0 or lgt.sum() == 0:
        return None
    smt = smt / smt.sum()
    lgt = lgt / lgt.sum()
    return {
        'tvd':    tvd(smt, lgt),
        'top_sm': list(sm.head(5).index.astype(int)),
        'top_lg': list(lg.head(5).index.astype(int)),
    }


def composition_match(real_r, syn_r):
    """How many of the real top-5 appear in the
    synthetic top-5? Magnitude can be right while
    the composition is wrong - this catches that."""
    sm_hit = len(set(real_r['top_sm'])
                 & set(syn_r['top_sm']))
    lg_hit = len(set(real_r['top_lg'])
                 & set(syn_r['top_lg']))
    return sm_hit, lg_hit


results = []

# ══════════════════════════════════════════
# RE-MEASURE EACH SAVED MODEL
# ══════════════════════════════════════════
for lo, hi, label in CANDIDATES:
    path = f'data/screen_{lo}_{hi}.pkl'
    print("\n" + "=" * 68)
    print(f"{label}   (small<={lo}, large>={hi})")
    print("=" * 68)

    try:
        model = CTGANSynthesizer.load(path)
    except Exception as e:
        print(f"  could not load {path}: {e}")
        continue

    # ── REAL side ──
    d = train.copy()
    d[COND_COL] = np.where(
        d['n_items'] <= lo, 'small',
        np.where(d['n_items'] >= hi,
                 'large', 'mid'))
    d = d[d[COND_COL] != 'mid']

    real_all = tail_shift(
        d[d[COND_COL] == 'small'],
        d[d[COND_COL] == 'large'])
    de = d[d['is_early_in_cart'] == 1]
    real_early = tail_shift(
        de[de[COND_COL] == 'small'],
        de[de[COND_COL] == 'large'])

    # ── SYNTHETIC side ──
    gen = {}
    for grp in ['small', 'large']:
        c = Condition(
            num_rows=N_GEN,
            column_values={COND_COL: grp})
        gen[grp] = model.sample_from_conditions(
            conditions=[c])

    syn_early_rate = pd.concat(
        gen.values())['is_early_in_cart'
                      ].astype(float).mean()

    syn_all = tail_shift(gen['small'],
                         gen['large'])
    gs = gen['small'][
        gen['small']['is_early_in_cart']
        .astype(int) == 1]
    gl = gen['large'][
        gen['large']['is_early_in_cart']
        .astype(int) == 1]
    syn_early = tail_shift(gs, gl)

    # ── PRE-CHECK: is early-cart generated
    #    faithfully? If not, view (B) is invalid.
    early_ok = abs(
        syn_early_rate - real_early_rate) < 0.08

    print(f"\n  is_early_in_cart rate")
    print(f"    real:      {real_early_rate:.3f}")
    print(f"    synthetic: {syn_early_rate:.3f}")
    print(f"    -> early-cart view "
          f"{'VALID' if early_ok else 'INVALID'}")
    if not early_ok:
        print("       (model does not reproduce the")
        print("        early-cart rate, so its")
        print("        'first 3 items' subset is not")
        print("        comparable. Use ALL-ROWS view.)")

    # ── VIEW A: ALL ROWS (like for like) ──
    rec_all = (syn_all['tvd'] / real_all['tvd']
               if real_all and real_all['tvd'] > 0
               else 0)
    sm_hit_a, lg_hit_a = composition_match(
        real_all, syn_all)

    print(f"\n  [A] ALL ROWS  (same population "
          f"both sides)")
    print(f"      real  TVD: {real_all['tvd']:.4f}")
    print(f"      synth TVD: {syn_all['tvd']:.4f}")
    print(f"      recovery:  {rec_all:.1%}")
    print(f"      real  small top5: "
          f"{real_all['top_sm']}")
    print(f"      synth small top5: "
          f"{syn_all['top_sm']}   "
          f"({sm_hit_a}/5 match)")
    print(f"      real  large top5: "
          f"{real_all['top_lg']}")
    print(f"      synth large top5: "
          f"{syn_all['top_lg']}   "
          f"({lg_hit_a}/5 match)")

    # ── VIEW B: EARLY-CART (like for like) ──
    rec_early = 0
    sm_hit_b = lg_hit_b = 0
    if real_early and syn_early \
            and real_early['tvd'] > 0:
        rec_early = (syn_early['tvd']
                     / real_early['tvd'])
        sm_hit_b, lg_hit_b = composition_match(
            real_early, syn_early)
        print(f"\n  [B] EARLY-CART ROWS  (same "
              f"population both sides)")
        print(f"      real  TVD: "
              f"{real_early['tvd']:.4f}")
        print(f"      synth TVD: "
              f"{syn_early['tvd']:.4f}")
        print(f"      recovery:  {rec_early:.1%}"
              f"   {'' if early_ok else '(INVALID)'}")
        print(f"      real  small top5: "
              f"{real_early['top_sm']}")
        print(f"      synth small top5: "
              f"{syn_early['top_sm']}   "
              f"({sm_hit_b}/5 match)")

    # what screen_thresholds WRONGLY reported
    old_rec = (syn_all['tvd'] / real_early['tvd']
               if real_early
               and real_early['tvd'] > 0 else 0)
    print(f"\n  [!] screen_thresholds reported "
          f"{old_rec:.1%}")
    print(f"      (synth ALL vs real EARLY - "
          f"mismatched populations)")

    results.append({
        'label':      label,
        'lo': lo, 'hi': hi,
        'real_all':   float(real_all['tvd']),
        'syn_all':    float(syn_all['tvd']),
        'rec_all':    float(rec_all),
        'comp_all':   f"{sm_hit_a}/5,{lg_hit_a}/5",
        'early_ok':   bool(early_ok),
        'rec_early':  float(rec_early),
        'syn_early_rate': float(syn_early_rate),
    })


# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("CORRECTED RESULTS  (view A - all rows,")
print("like-for-like)")
print("=" * 68)
print(f"\n{'Candidate':<14}{'realTVD':>9}"
      f"{'synTVD':>9}{'recov':>9}"
      f"{'top5 match':>13}")
print("-" * 68)
for r in results:
    print(f"{r['label']:<14}"
          f"{r['real_all']:>9.4f}"
          f"{r['syn_all']:>9.4f}"
          f"{r['rec_all']:>9.1%}"
          f"{r['comp_all']:>13}")

print(f"\n  noise floor: {NOISE_FLOOR}")
print("  recovery near 100% = faithful")
print("  recovery >> 100%   = model EXAGGERATES")
print("                       (fidelity error)")
print("  top5 match         = composition, not")
print("                       just magnitude")

# ══════════════════════════════════════════
# CHOOSE - by FIDELITY, not by biggest shift
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("DECISION")
print("=" * 68)

valid = [r for r in results
         if r['syn_all'] > NOISE_FLOOR]

if not valid:
    print("""
  NO CANDIDATE CLEARS THE NOISE FLOOR.
  CTGAN did not learn the shift. Report honestly
  and do not build the fleet story on conditioned
  aisle differences.
""")
else:
    # fidelity = how close recovery is to 1.0
    for r in valid:
        r['fidelity_err'] = abs(r['rec_all'] - 1.0)

    best = min(valid,
               key=lambda r: r['fidelity_err'])

    print("\n  Ranked by FIDELITY (closest to 100%),")
    print("  not by biggest shift. A model that")
    print("  overstates the demand gap would mislead")
    print("  a fleet planner.\n")
    for r in sorted(valid,
                    key=lambda x: x['fidelity_err']):
        print(f"    {r['label']:<14} "
              f"recovery {r['rec_all']:>6.1%}   "
              f"error {r['fidelity_err']:>5.1%}   "
              f"comp {r['comp_all']}")

    print(f"""
  ==> USE: {best['label']}
      small <= {best['lo']}, large >= {best['hi']}

      real TVD  {best['real_all']:.4f}
      synth TVD {best['syn_all']:.4f}
      recovery  {best['rec_all']:.1%}

  NOTE: these models are only 30 epochs. Recovery
  and composition should both IMPROVE with full
  training. That is a prediction - we verify it,
  we do not assume it.
""")

with open('data/remeasure_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved data/remeasure_results.json")
print("=" * 68)