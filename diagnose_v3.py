"""
diagnose_v3.py
==============
Three open questions from train_v3.py. All measured,
none guessed. No retraining.

QUESTION 1 - why did Column Pair Trends return nan?
---------------------------------------------------
The quality score reported 93.22%, but that was only
the Column Shapes half. Column Pair Trends came back
nan, so HALF the quality metric is missing.

Likely cause: days_since_prior_order is declared
numerical while the other five are categorical.
SDMetrics computes pair trends differently for
numerical-vs-categorical pairs, and a degenerate
column (zero variance in the sample, or NaNs) makes
the correlation undefined.

We check every pair and find which one produces nan.

QUESTION 2 - is the is_reorder gap uniform?
-------------------------------------------
  real 0.594  ->  synth 0.546   (-8% relative)

is_reorder is the TARGET of the ML efficacy test.
An 8% under-generation matters.

But is it uniform, or concentrated in one conditioning
class? If small baskets are fine and large are off (or
vice versa), that is a different problem with a
different fix. We break it down by class.

QUESTION 3 - did the conditional shift survive?
-----------------------------------------------
This is the whole reason for the rebuild. At 30 epochs
the screened model recovered 108% of the real aisle
shift with 5/5 composition match.

Does the 300-epoch model still do it? More training
should IMPROVE fidelity (recovery closer to 100%).
That is a prediction. We verify it - we do not assume
it. My last prediction (that training would fix
is_early_in_cart) was WRONG.

MY MEASUREMENT BUG - acknowledged
---------------------------------
train_v3.py flagged aisle_id as FAIL because I forgot
to give it a relative tolerance. Real 71.203 vs synth
70.885 is a 0.4% error on a 1-134 range. That is fine.
The FAIL was my bug, not the model's.

This script uses RELATIVE tolerance for every non-rate
column.
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

SMALL_MAX = 10
LARGE_MIN = 14
DOMINANT = [24, 83]
NOISE_FLOOR = 0.0357
COND_COL = 'order_size_grp'
N_GEN = 30000

BASE_COLS = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_reorder', 'is_early_in_cart',
    'days_since_prior_order',
]

# rate columns judged on ABSOLUTE difference,
# everything else on RELATIVE
RATE_COLS = ['is_reorder', 'is_early_in_cart']

print("=" * 68)
print("DIAGNOSE v3")
print("=" * 68)

print("\nLoading...")
train = pd.read_csv('data/v3_train.csv')
eval_df = pd.read_csv('data/v3_eval.csv')
synth = pd.read_csv('data/synthetic_v3.csv')

# disjointness re-check
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
assert len(set(eval_df['order_id']) & tr_ids) == 0
print("  [OK] disjointness re-verified")

osize = train.groupby('order_id').size()
train['n_items'] = train['order_id'].map(osize)
train[COND_COL] = np.where(
    train['n_items'] <= SMALL_MAX, 'small',
    np.where(train['n_items'] >= LARGE_MIN,
             'large', 'mid'))
td = train[train[COND_COL] != 'mid'].copy()

print(f"  train (2-class): {len(td):,}")
print(f"  eval:            {len(eval_df):,}")
print(f"  synthetic:       {len(synth):,}")


# ══════════════════════════════════════════
# CORRECTED MARGINAL CHECK
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[0] MARGINALS - with CORRECT tolerances")
print("=" * 68)
print("\n    (train_v3.py used an absolute 0.03")
print("     tolerance on aisle_id, a 1-134 column.")
print("     That was my bug. Relative here.)\n")

print(f"    {'Column':<26}{'Real':>9}{'Synth':>9}"
      f"{'Rel err':>9}  Status")
print("    " + "-" * 62)

marg = []
for c in BASE_COLS:
    r = pd.to_numeric(td[c], errors='coerce').mean()
    s = pd.to_numeric(synth[c],
                      errors='coerce').mean()
    if c in RATE_COLS:
        err = abs(r - s)
        ok = err <= 0.03
        shown = err
    else:
        err = abs(r - s) / max(abs(r), 1e-9)
        ok = err <= 0.05
        shown = err
    status = "OK" if ok else "*** FAIL ***"
    if c in RATE_COLS:
        print(f"    {c:<26}{r:>9.3f}{s:>9.3f}"
              f"{shown:>8.3f}   {status}")
    else:
        print(f"    {c:<26}{r:>9.3f}{s:>9.3f}"
              f"{shown:>8.1%}   {status}")
    marg.append({'col': c, 'real': float(r),
                 'synth': float(s),
                 'err': float(err), 'ok': bool(ok)})

fails = [m['col'] for m in marg if not m['ok']]
print(f"\n    Genuine failures: "
      f"{fails if fails else 'NONE'}")


# ══════════════════════════════════════════
# Q1 - WHY nan?
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[1] WHY DID COLUMN PAIR TRENDS RETURN nan?")
print("=" * 68)

print("\n    Column diagnostics (eval vs synthetic):")
print(f"\n    {'Column':<26}{'dtype':>10}"
      f"{'nuniq_e':>9}{'nuniq_s':>9}{'NaN_s':>8}")
print("    " + "-" * 62)
for c in BASE_COLS:
    e = eval_df[c]
    s = synth[c]
    print(f"    {c:<26}{str(s.dtype):>10}"
          f"{e.nunique():>9}{s.nunique():>9}"
          f"{int(s.isna().sum()):>8}")

print("\n    Checking each pair for undefined "
      "correlation:")
bad_pairs = []
for i in range(len(BASE_COLS)):
    for j in range(i + 1, len(BASE_COLS)):
        a, b = BASE_COLS[i], BASE_COLS[j]
        sa = pd.to_numeric(synth[a],
                           errors='coerce')
        sb = pd.to_numeric(synth[b],
                           errors='coerce')
        if sa.std() == 0 or sb.std() == 0:
            bad_pairs.append(
                (a, b, 'zero variance'))
            continue
        corr = sa.corr(sb)
        if pd.isna(corr):
            bad_pairs.append(
                (a, b, 'corr is nan'))

if bad_pairs:
    print("\n    PROBLEM PAIRS:")
    for a, b, why in bad_pairs:
        print(f"      {a} x {b}  -> {why}")
else:
    print("\n    No degenerate pairs found in the")
    print("    synthetic data itself.")
    print("\n    => The nan likely comes from SDMetrics")
    print("       handling the numerical column")
    print("       (days_since_prior_order) against")
    print("       categorical columns. Fix: declare")
    print("       ALL columns categorical for scoring,")
    print("       or drop the numerical from the")
    print("       quality metric and report it")
    print("       separately.")

# Does it compute if all are categorical?
print("\n    Testing: does pair-trends compute if we")
print("    treat every column as categorical?")
try:
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import (
        evaluate_quality)
    m2 = SingleTableMetadata()
    m2.detect_from_dataframe(eval_df[BASE_COLS])
    for c in BASE_COLS:
        m2.update_column(column_name=c,
                         sdtype='categorical')
    n = min(len(eval_df), len(synth), 9000)
    q2 = evaluate_quality(
        real_data=eval_df[BASE_COLS].sample(
            n=n, random_state=SEED),
        synthetic_data=synth[BASE_COLS].sample(
            n=n, random_state=SEED),
        metadata=m2, verbose=False)
    print(f"\n    ALL-CATEGORICAL quality: "
          f"{q2.get_score():.4f}")
    print("    (if this is not nan, the numerical")
    print("     declaration was the cause)")
except Exception as e:
    print(f"\n    failed: {e}")


# ══════════════════════════════════════════
# Q2 - IS THE is_reorder GAP UNIFORM?
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[2] IS THE is_reorder GAP UNIFORM?")
print("=" * 68)
print("\n    is_reorder is the TARGET of the ML")
print("    efficacy test. An 8% gap matters.")
print("    Uniform, or concentrated in one class?\n")

model = CTGANSynthesizer.load('data/ctgan_v3.pkl')

gen = {}
for grp in ['small', 'large']:
    c = Condition(num_rows=N_GEN,
                  column_values={COND_COL: grp})
    gen[grp] = model.sample_from_conditions(
        conditions=[c])

print(f"    {'Class':<10}{'Real':>10}{'Synth':>10}"
      f"{'Diff':>9}")
print("    " + "-" * 40)
for grp in ['small', 'large']:
    r = td[td[COND_COL] == grp][
        'is_reorder'].astype(float).mean()
    s = gen[grp]['is_reorder'].astype(float).mean()
    print(f"    {grp:<10}{r:>10.3f}{s:>10.3f}"
          f"{abs(r-s):>9.3f}")

r_all = td['is_reorder'].astype(float).mean()
s_all = synth['is_reorder'].astype(float).mean()
print(f"    {'overall':<10}{r_all:>10.3f}"
      f"{s_all:>10.3f}{abs(r_all-s_all):>9.3f}")

# same for early_in_cart
print(f"\n    is_early_in_cart by class:")
print(f"    {'Class':<10}{'Real':>10}{'Synth':>10}"
      f"{'Diff':>9}")
print("    " + "-" * 40)
for grp in ['small', 'large']:
    r = td[td[COND_COL] == grp][
        'is_early_in_cart'].astype(float).mean()
    s = gen[grp]['is_early_in_cart'].astype(
        float).mean()
    print(f"    {grp:<10}{r:>10.3f}{s:>10.3f}"
          f"{abs(r-s):>9.3f}")

print("\n    NOTE: is_early_in_cart went")
print("    0.335 (30ep) -> 0.317 (300ep).")
print("    10x the training moved it 0.018.")
print("    My undertraining hypothesis was WRONG.")
print("    It converges, but far too slowly to")
print("    close the gap. This is a real CTGAN")
print("    limitation on this column.")


# ══════════════════════════════════════════
# Q3 - DID THE CONDITIONAL SHIFT SURVIVE?
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[3] DID THE CONDITIONAL AISLE SHIFT SURVIVE?")
print("=" * 68)
print("\n    The whole point of the rebuild.")
print("    30-epoch screen: recovery 108%, 5/5 comp.")
print("    Does 300 epochs hold up?\n")


def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


def tail_shift(ds, dl):
    sm = ds['aisle_id'].value_counts(
        normalize=True)
    lg = dl['aisle_id'].value_counts(
        normalize=True)
    smt = sm[~sm.index.isin(DOMINANT)]
    lgt = lg[~lg.index.isin(DOMINANT)]
    smt = smt / smt.sum()
    lgt = lgt / lgt.sum()
    return (tvd(smt, lgt),
            list(sm.head(5).index.astype(int)),
            list(lg.head(5).index.astype(int)))


real_tvd, r_sm, r_lg = tail_shift(
    td[td[COND_COL] == 'small'],
    td[td[COND_COL] == 'large'])
syn_tvd, s_sm, s_lg = tail_shift(
    gen['small'], gen['large'])

recovery = syn_tvd / real_tvd if real_tvd else 0
sm_hit = len(set(r_sm) & set(s_sm))
lg_hit = len(set(r_lg) & set(s_lg))

print(f"    real  tail TVD: {real_tvd:.4f}")
print(f"    synth tail TVD: {syn_tvd:.4f}")
print(f"    recovery:       {recovery:.1%}")
print(f"    noise floor:    {NOISE_FLOOR}")
print(f"\n    real  small top5: {r_sm}")
print(f"    synth small top5: {s_sm}   "
      f"({sm_hit}/5 match)")
print(f"    real  large top5: {r_lg}")
print(f"    synth large top5: {s_lg}   "
      f"({lg_hit}/5 match)")

print(f"\n    30-epoch screen was: 108.0%, 5/5")
print(f"    300-epoch model is:  {recovery:.1%}, "
      f"{sm_hit}/5")

print()
if syn_tvd < NOISE_FLOOR:
    print("    *** THE SHIFT DID NOT SURVIVE. ***")
    print("    Synthetic TVD is below the noise")
    print("    floor. The 300-epoch model does not")
    print("    reproduce the conditioned aisle shift.")
    print("    Report this honestly.")
elif abs(recovery - 1.0) <= 0.20 and sm_hit >= 4:
    print("    THE SHIFT SURVIVED AND IS FAITHFUL.")
    print("    Recovery near 100% with correct aisle")
    print("    composition. The conditional generation")
    print("    genuinely works.")
elif recovery > 1.25:
    print("    SHIFT PRESENT BUT EXAGGERATED.")
    print("    The model overstates the demand gap.")
    print("    Usable, but report the exaggeration -")
    print("    synthetic data that overstates a gap")
    print("    would mislead a fleet planner.")
else:
    print("    SHIFT PRESENT BUT PARTIAL.")
    print("    Report the recovery figure honestly.")


# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("SUMMARY")
print("=" * 68)
print(f"""
  Genuine marginal failures: {fails if fails else 'NONE'}
  (aisle_id was my tolerance bug, not a real failure)

  Conditional aisle shift:
      real  {real_tvd:.4f}
      synth {syn_tvd:.4f}
      recovery {recovery:.1%}, composition {sm_hit}/5

  DECIDE FROM THIS - do not guess:
    - if the shift survived, the extension is real
    - the marginal gaps are honest limitations to
      report, NOT things to calibrate away
    - is_early_in_cart does not converge; it may
      need excluding from the ML efficacy features
""")

json.dump({
    'marginals': marg,
    'fails': fails,
    'real_tvd': float(real_tvd),
    'syn_tvd': float(syn_tvd),
    'recovery': float(recovery),
    'comp_small': sm_hit,
    'comp_large': lg_hit,
}, open('data/v3_diagnosis.json', 'w'), indent=2)
print("Saved data/v3_diagnosis.json")
print("=" * 68)