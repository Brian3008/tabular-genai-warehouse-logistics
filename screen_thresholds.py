"""
screen_thresholds.py
====================
THE QUESTION THIS ANSWERS
-------------------------
The threshold sweep measured the aisle shift in the
REAL data. It did NOT show that CTGAN can LEARN and
REPRODUCE that shift.

Those are different things. This project has already
been burned once by assuming a property of the data
transfers into a property of the model.

So: train three candidate thresholds, condition each
model on small vs large, and measure whether the
SYNTHETIC aisle mix shifts the way the real one does.

WHY LOW EPOCHS
--------------
30 epochs, not 300. Undertrained - but every
candidate is handicapped EQUALLY, so the RELATIVE
comparison is valid. We only need to know which
threshold CTGAN learns the shift from most readily.

The winner then gets a full 300-epoch run.

WHY THE PROVEN CONFIG
---------------------
batch=300, lr=0.00005, disc_steps=3 - identical to
the config that produced the best model.

A faster config would halve the runtime, but if all
three candidates then failed, we could not tell
whether the THRESHOLDS were wrong or the CONFIG was.
Two variables changing at once makes a negative
result uninterpretable.

Only the threshold varies. Everything else is held
fixed. So a negative result actually means
something.

THE HONEST OUTCOME WE MUST BE PREPARED FOR
------------------------------------------
CTGAN may reproduce NONE of them. If synthetic TVD
is near zero at every threshold, then conditional
aisle generation does not work on this data, and we
report that honestly instead of building on sand.

Finding that out in 75 minutes is a good outcome.

METRIC
------
  real tail TVD      = shift in the real data
  synthetic tail TVD = shift the model reproduces
  recovery = synthetic / real

  recovery > 0.60  -> model learned it well
  recovery > 0.30  -> partial, usable, report honestly
  recovery < 0.30  -> model did NOT learn it

Noise floor (weekend TVD in real data): 0.0357
Any synthetic TVD below that is indistinguishable
from nothing.
"""

import pandas as pd
import numpy as np
import random
import torch
import json
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 30          # screening only
N_GEN = 20000        # per class
DOMINANT = [24, 83]
NOISE_FLOOR = 0.0357

# small_max, large_min, label
CANDIDATES = [
    (6,  12, "sharp 33/67   (imbalanced)"),
    (9,  12, "55/70         (rule pick)"),
    (10, 14, "60/75         (recommended)"),
]

# Trained on. Nothing derived, nothing circular.
BASE_COLS = [
    'aisle_id',
    'order_dow',
    'order_hour_of_day',
    'is_reorder',
    'is_early_in_cart',
    'days_since_prior_order',
]
COND_COL = 'order_size_grp'

print("=" * 66)
print("SCREENING - can CTGAN LEARN the aisle shift?")
print("=" * 66)
print(f"\nEpochs: {EPOCHS} (screening - relative")
print("        comparison only, not final quality)")
print(f"Candidates: {len(CANDIDATES)}")
print(f"Noise floor: {NOISE_FLOOR}")


# ══════════════════════════════════════════
# LOAD - training rows ONLY
# ══════════════════════════════════════════
print("\n[1] Loading v3_train.csv ...")
train_full = pd.read_csv('data/v3_train.csv')
print(f"    rows:   {len(train_full):,}")
print(f"    orders: "
      f"{train_full['order_id'].nunique():,}")

# Verify we are not touching eval data
train_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
eval_ids = set(pd.read_csv(
    'data/v3_eval.csv')['order_id'])
assert len(train_ids & eval_ids) == 0, \
    "TRAIN/EVAL OVERLAP"
assert set(train_full['order_id']) <= train_ids, \
    "train_full contains non-training orders"
print("    [OK] disjointness verified")

# recompute items per order within the
# training rows
osize = train_full.groupby('order_id').size()
train_full['n_items'] = train_full[
    'order_id'].map(osize)


def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


def tail_tvd(df_small, df_large):
    """Aisle-mix shift, excluding the two
    dominant aisles."""
    sm = df_small['aisle_id'].value_counts(
        normalize=True)
    lg = df_large['aisle_id'].value_counts(
        normalize=True)
    smt = sm[~sm.index.isin(DOMINANT)]
    lgt = lg[~lg.index.isin(DOMINANT)]
    if smt.sum() == 0 or lgt.sum() == 0:
        return 0.0, [], []
    smt = smt / smt.sum()
    lgt = lgt / lgt.sum()
    return (tvd(smt, lgt),
            list(sm.head(5).index.astype(int)),
            list(lg.head(5).index.astype(int)))


results = []

# ══════════════════════════════════════════
# RUN EACH CANDIDATE
# ══════════════════════════════════════════
for lo, hi, label in CANDIDATES:
    print("\n" + "=" * 66)
    print(f"CANDIDATE: {label}")
    print(f"  small <= {lo} items, "
          f"large >= {hi} items")
    print("=" * 66)

    d = train_full.copy()
    d[COND_COL] = np.where(
        d['n_items'] <= lo, 'small',
        np.where(d['n_items'] >= hi,
                 'large', 'mid'))

    # drop mid - two-class conditioning
    d = d[d[COND_COL] != 'mid']

    share = d[COND_COL].value_counts(
        normalize=True)
    print(f"\n  rows: {len(d):,}")
    print(f"  small: {share.get('small', 0):.1%}"
          f"   large: {share.get('large', 0):.1%}")

    # ── REAL BENCHMARK ──
    # size-controlled: first 3 items only, so
    # every order contributes exactly 3 rows
    f3 = d[d['is_early_in_cart'] == 1]
    real_tvd, real_sm, real_lg = tail_tvd(
        f3[f3[COND_COL] == 'small'],
        f3[f3[COND_COL] == 'large'])
    print(f"\n  REAL tail TVD: {real_tvd:.4f}")
    print(f"    small top5: {real_sm}")
    print(f"    large top5: {real_lg}")

    # ── TRAIN ──
    train_cols = BASE_COLS + [COND_COL]
    td = d[train_cols].copy()

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(td)
    for c in train_cols:
        if c == 'days_since_prior_order':
            meta.update_column(
                column_name=c, sdtype='numerical')
        else:
            meta.update_column(
                column_name=c,
                sdtype='categorical')

    # PROVEN CONFIG - identical to the one that
    # produced your best model. Using the same
    # config for all three candidates means only
    # ONE variable changes (the threshold), so a
    # negative result is interpretable.
    print(f"\n  Training CTGAN "
          f"({EPOCHS} epochs, proven config)...")
    model = CTGANSynthesizer(
        meta,
        epochs=EPOCHS,
        batch_size=300,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=0.00005,
        discriminator_lr=0.00005,
        discriminator_steps=3,
        verbose=True,
        cuda=True,
    )
    model.fit(td)
    model.save(f'data/screen_{lo}_{hi}.pkl')

    # ── CONDITION AND GENERATE ──
    print("\n  Generating conditioned samples...")
    gen = {}
    for grp in ['small', 'large']:
        c = Condition(
            num_rows=N_GEN,
            column_values={COND_COL: grp})
        gen[grp] = model.sample_from_conditions(
            conditions=[c])

    synth_tvd, syn_sm, syn_lg = tail_tvd(
        gen['small'], gen['large'])

    recovery = (synth_tvd / real_tvd
                if real_tvd > 0 else 0)

    print(f"\n  SYNTHETIC tail TVD: "
          f"{synth_tvd:.4f}")
    print(f"    small top5: {syn_sm}")
    print(f"    large top5: {syn_lg}")
    print(f"\n  RECOVERY: {recovery:.1%}")
    print(f"    (synthetic {synth_tvd:.4f} / "
          f"real {real_tvd:.4f})")

    if synth_tvd < NOISE_FLOOR:
        print(f"    [!] BELOW NOISE FLOOR "
              f"({NOISE_FLOOR}) - the model")
        print(f"        did not learn the shift")

    results.append({
        'label':     label,
        'lo':        lo,
        'hi':        hi,
        'min_share': float(min(
            share.get('small', 0),
            share.get('large', 0))),
        'real_tvd':  float(real_tvd),
        'synth_tvd': float(synth_tvd),
        'recovery':  float(recovery),
        'syn_small': syn_sm,
        'syn_large': syn_lg,
    })

    with open('data/screen_results.json',
              'w') as f:
        json.dump(results, f, indent=2)


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 66)
print("SCREENING RESULTS")
print("=" * 66)
print(f"\n{'Candidate':<30}{'balance':>9}"
      f"{'realTVD':>9}{'synTVD':>9}"
      f"{'recov':>8}")
print("-" * 66)
for r in results:
    print(f"{r['label']:<30}"
          f"{r['min_share']:>8.1%}"
          f"{r['real_tvd']:>9.4f}"
          f"{r['synth_tvd']:>9.4f}"
          f"{r['recovery']:>8.1%}")

print(f"\n  noise floor: {NOISE_FLOOR}")

usable = [r for r in results
          if r['synth_tvd'] > NOISE_FLOOR
          and r['recovery'] > 0.30]

print("\n" + "=" * 66)
print("VERDICT")
print("=" * 66)

if not usable:
    print(f"""
  CTGAN DID NOT LEARN THE SHIFT AT ANY THRESHOLD.

  Every synthetic TVD is at or below the noise
  floor. The aisle shift exists in the real data
  but the model does not reproduce it when
  conditioned.

  This is an HONEST NEGATIVE RESULT, found in
  ~75 minutes rather than after a 4-hour run.

  What it means:
    - The generator still works for its main
      purpose (ML efficacy, privacy, quality).
    - Conditional AISLE generation does not work
      on this data. Report that plainly.
    - Do NOT build the fleet story on conditioned
      aisle differences.

  Next: consider whether more epochs would help
  (run ONE candidate at 300 to check), or accept
  and reframe.
""")
else:
    best = max(usable,
               key=lambda r: r['synth_tvd'])
    print(f"\n  USABLE: {len(usable)} of "
          f"{len(results)}\n")
    for r in sorted(usable,
                    key=lambda x: -x['synth_tvd']):
        print(f"    {r['label']}")
        print(f"      synth TVD {r['synth_tvd']:.4f}"
              f"   recovery {r['recovery']:.1%}"
              f"   balance {r['min_share']:.1%}")

    print(f"""
  ==> WINNER: {best['label']}
      small <= {best['lo']}, large >= {best['hi']}

      At only {EPOCHS} epochs CTGAN already
      reproduces {best['recovery']:.0%} of the real
      aisle shift ({best['synth_tvd']:.4f} vs real
      {best['real_tvd']:.4f}).

      A full 300-epoch run should do better.

  NEXT: train this threshold properly (300 epochs)
        and run the full honest evaluation.
""")

print("=" * 66)
print("Saved: data/screen_results.json")