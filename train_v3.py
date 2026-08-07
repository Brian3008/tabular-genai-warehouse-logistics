"""
train_v3.py
===========
The clean rebuild. Every defect found in the audit
is fixed at the source.

WHAT WAS BROKEN IN v2, AND HOW v3 FIXES IT
------------------------------------------
1. TRAIN/EVAL OVERLAP
   v2: train_best_model.py and create_eval_set.py both
       sampled freely from the same 32.4M rows. Nothing
       held them apart. The model was scored on rows it
       may have trained on.
   v3: trains ONLY on v3_train.csv, whose order_ids are
       provably disjoint from eval/compare. This script
       ASSERTS that before training and crashes if it
       is violated.

2. MODEL SELECTION ON TRAINING DATA
   v2: real_eval = sample[eval_cols].sample(n=5000)
       - each model was scored against a slice of its
       OWN training sample. The "winner" was chosen on
       a training-set score.
   v3: scored ONLY on held-out v3_eval.csv.

3. DELIBERATE OVERSAMPLING
   v2: added 15k extra night rows, 30k peak, 40k reorder
       on top of a 250k base. The model then learned an
       inflated night rate (0.111 vs the real 0.050) -
       CORRECTLY, because that is what it was shown.
   v3: trains on the true distribution. Night comes out
       near 0.050 because that is what the data says.

4. POST-PROCESSING THAT INJECTED REAL DATA
   v2: run_project.py overwrote synthetic
       order_hour_of_day with hours sampled from the REAL
       dataset, then recomputed is_night / is_peak_hour /
       time_of_day from them. Four of the seven scored
       metrics were partly real values. The 93.47% was
       inflated by construction.
   v3: NO post-processing. None. The score is the
       model's true score. Derived features are computed
       AFTER generation by the same deterministic
       function used on the real data, so they are
       consistent by construction and there is nothing
       to calibrate.

5. CIRCULAR FEATURES
   v2: trained on 13 columns, 7 of which were
       deterministic functions of the other 6.
       aisle_popularity was a straight lookup from
       aisle_id - conditioning on it constrained aisle
       directly, which is what faked the 98.7%.
   v3: trains on 6 INDEPENDENT columns only.
       aisle_popularity is never trained on and never
       conditioned on.

CONDITIONING AXIS
-----------------
order_size_grp: small (<=10 items) vs large (>=14).

Chosen on evidence, not instinct. Six axes were tested:
  calendar / weekend    TVD 0.036  (noise)
  peak hour             TVD 0.029  (noise)
  user frequency        lift +1.3% (nothing)
  cart depth            lift +0.0% (nothing)
  demand type           lift +2.8% (nearly nothing)
  department mix        CIRCULAR (it is aisle_id)
  ORDER SIZE            passed all three tests

And CTGAN demonstrably LEARNS it: at only 30 epochs the
screened model recovered 108% of the real aisle shift
with 5/5 aisle composition match.

THE OPEN QUESTION THIS SCRIPT ANSWERS
-------------------------------------
At 30 epochs, is_early_in_cart generated at 0.335 against
a real 0.281 - materially too high. I believe that is an
undertraining artefact. That is a HYPOTHESIS, not a
finding.

It matters because is_early_in_cart feeds the ML efficacy
classifier. If it stays broken, it contaminates the
headline result.

So this script VERIFIES ALL SIX MARGINALS after training
and reports pass/fail per column BEFORE anything else is
built on the model. The check runs whether I am right or
wrong.

OUTPUTS
-------
  data/ctgan_v3.pkl
  data/synthetic_v3.csv
  data/v3_training_report.json
"""

import sys
# Some prints use non-ASCII (e.g. the intersection symbol).
# A default Windows console is cp1252 and CRASHES on them;
# force UTF-8 so the script runs outside PyCharm too.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import random
import torch
import json
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── CONFIG ──
EPOCHS = 300
SMALL_MAX = 10      # from the threshold sweep
LARGE_MIN = 14
N_SYNTH = 50000

# The six INDEPENDENT columns. Nothing here is a
# function of anything else here.
BASE_COLS = [
    'aisle_id',
    'order_dow',
    'order_hour_of_day',
    'is_reorder',
    'is_early_in_cart',
    'days_since_prior_order',
]
COND_COL = 'order_size_grp'

print("=" * 68)
print("TRAIN v3 - THE CLEAN REBUILD")
print("=" * 68)
print(f"\n  epochs:      {EPOCHS}")
print(f"  conditioning: small <= {SMALL_MAX}, "
      f"large >= {LARGE_MIN} items")
print(f"  trained on:   {len(BASE_COLS)} independent "
      f"columns + {COND_COL}")
print(f"  post-process: NONE")

print("\n  CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(" ", torch.cuda.get_device_name(0))


# ══════════════════════════════════════════
# SHARED DERIVATION - single source of truth
# ══════════════════════════════════════════
# Used on real data AND on synthetic data. Same
# function both times, so derived features can
# never be inconsistent and never need calibrating.
def derive_features(d):
    d = d.copy()
    h = pd.to_numeric(d['order_hour_of_day'],
                      errors='coerce')
    dow = pd.to_numeric(d['order_dow'],
                        errors='coerce')
    dspo = pd.to_numeric(
        d['days_since_prior_order'],
        errors='coerce').fillna(0)

    d['is_weekend'] = dow.isin([0, 1]).astype(int)
    d['is_peak_hour'] = (
        (h >= 9) & (h <= 15)).astype(int)
    d['is_night'] = ((h < 6) | (h > 21)).astype(int)
    d['time_of_day'] = pd.cut(
        h, bins=[-1, 6, 12, 17, 21, 24],
        labels=['night', 'morning', 'afternoon',
                'evening', 'late']).astype(str)
    d['order_frequency'] = pd.cut(
        dspo, bins=[-1, 0, 7, 14, 30, 999],
        labels=['first', 'weekly', 'biweekly',
                'monthly', 'infrequent']).astype(str)
    return d


# ══════════════════════════════════════════
# [1] LOAD + PROVE DISJOINTNESS
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[1] LOADING - and proving disjointness")
print("=" * 68)

train = pd.read_csv('data/v3_train.csv')
eval_df = pd.read_csv('data/v3_eval.csv')
train_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
eval_ids = set(eval_df['order_id'])

# Hard assertions. Crash rather than quietly
# produce a flattering number.
assert len(train_ids & eval_ids) == 0, \
    "FATAL: train and eval share order_ids"
assert set(train['order_id']) <= train_ids, \
    "FATAL: train contains non-training orders"
assert len(set(eval_df['order_id'])
           & train_ids) == 0, \
    "FATAL: eval contains training orders"

print(f"    train:  {len(train):,} rows, "
      f"{train['order_id'].nunique():,} orders")
print(f"    eval:   {len(eval_df):,} rows, "
      f"{eval_df['order_id'].nunique():,} orders")
print(f"    train ∩ eval: "
      f"{len(train_ids & eval_ids)} orders")
print("\n    [ASSERTED] The model cannot see eval data.")


# ══════════════════════════════════════════
# [2] BUILD CONDITIONING COLUMN
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[2] CONDITIONING COLUMN")
print("=" * 68)

osize = train.groupby('order_id').size()
train['n_items'] = train['order_id'].map(osize)
train[COND_COL] = np.where(
    train['n_items'] <= SMALL_MAX, 'small',
    np.where(train['n_items'] >= LARGE_MIN,
             'large', 'mid'))

# two-class conditioning - mid dropped
td = train[train[COND_COL] != 'mid'].copy()

share = td[COND_COL].value_counts(normalize=True)
print(f"\n    rows after dropping 'mid': "
      f"{len(td):,}")
print(f"    small: {share.get('small', 0):.1%}")
print(f"    large: {share.get('large', 0):.1%}")

if min(share.get('small', 0),
       share.get('large', 0)) < 0.25:
    print("\n    [!] WARNING: class imbalance below")
    print("        25%. CTGAN may under-learn the")
    print("        minority class.")


# ══════════════════════════════════════════
# [3] METADATA
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[3] METADATA")
print("=" * 68)

train_cols = BASE_COLS + [COND_COL]
X = td[train_cols].copy()

meta = SingleTableMetadata()
meta.detect_from_dataframe(X)
for c in train_cols:
    if c == 'days_since_prior_order':
        meta.update_column(column_name=c,
                           sdtype='numerical')
    else:
        meta.update_column(column_name=c,
                           sdtype='categorical')

print(f"\n    trained columns ({len(train_cols)}):")
for c in train_cols:
    kind = ('numerical'
            if c == 'days_since_prior_order'
            else 'categorical')
    print(f"      {c:<26} {kind}")
print("\n    NOT trained on (derived after):")
for c in ['is_weekend', 'is_peak_hour', 'is_night',
          'time_of_day', 'order_frequency',
          'department_id', 'aisle_popularity']:
    print(f"      {c}")


# ══════════════════════════════════════════
# [4] TRAIN
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[4] TRAINING")
print("=" * 68)
print(f"\n    {EPOCHS} epochs, proven config")
print("    (batch 300, lr 5e-5, disc_steps 3)")
print("    ~4 hours. Leave it running.\n")

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
model.fit(X)
model.save('data/ctgan_v3.pkl')
print("\n    saved data/ctgan_v3.pkl")


# ══════════════════════════════════════════
# [5] GENERATE  (NO POST-PROCESSING)
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[5] GENERATE - no post-processing")
print("=" * 68)

synth = model.sample(num_rows=N_SYNTH)
print(f"\n    generated {len(synth):,} rows")

# Derive dependent features with the SAME function
# used on the real data. Consistent by construction.
synth = derive_features(synth)
synth.to_csv('data/synthetic_v3.csv', index=False)
print("    derived features added "
      "(same function as real data)")
print("    saved data/synthetic_v3.csv")
print("\n    NOTHING was injected. NOTHING was")
print("    calibrated. This is the model's true")
print("    output.")


# ══════════════════════════════════════════
# [6] MARGINAL VERIFICATION  <-- THE KEY CHECK
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[6] MARGINAL VERIFICATION")
print("=" * 68)
print("""
    At 30 epochs, is_early_in_cart generated at
    0.335 vs a real 0.281 - materially too high.
    I believe that was undertraining. This check
    tests that belief. It runs whether I am right
    or wrong.

    is_early_in_cart feeds the ML efficacy
    classifier. If it is still broken, that
    contaminates the headline result and we must
    say so.
""")

print(f"    {'Column':<26}{'Real':>9}"
      f"{'Synth':>9}{'Diff':>8}  Status")
print("    " + "-" * 60)

marginals = []
all_ok = True
for c in BASE_COLS:
    r = pd.to_numeric(td[c],
                      errors='coerce').mean()
    s = pd.to_numeric(synth[c],
                      errors='coerce').mean()
    diff = abs(r - s)
    # relative tolerance for the two that are
    # not 0-1 rates
    if c in ('order_hour_of_day', 'order_dow',
             'days_since_prior_order'):
        tol = 0.05 * max(abs(r), 1)
    else:
        tol = 0.03
    ok = diff <= tol
    if not ok:
        all_ok = False
    status = "OK" if ok else "*** FAIL ***"
    print(f"    {c:<26}{r:>9.3f}{s:>9.3f}"
          f"{diff:>8.3f}  {status}")
    marginals.append({
        'col': c, 'real': float(r),
        'synth': float(s), 'diff': float(diff),
        'ok': bool(ok)})

# derived features - should be right automatically
print(f"\n    Derived (computed, not generated):")
print(f"    {'Column':<26}{'Real':>9}"
      f"{'Synth':>9}{'Diff':>8}  Status")
print("    " + "-" * 60)
real_derived = derive_features(td)
for c in ['is_weekend', 'is_peak_hour', 'is_night']:
    r = real_derived[c].mean()
    s = synth[c].mean()
    diff = abs(r - s)
    ok = diff <= 0.03
    status = "OK" if ok else "*** FAIL ***"
    print(f"    {c:<26}{r:>9.3f}{s:>9.3f}"
          f"{diff:>8.3f}  {status}")

print()
if all_ok:
    print("    ALL SIX TRAINED MARGINALS PASS.")
    print("    The early-cart problem WAS")
    print("    undertraining. Safe to proceed.")
else:
    print("    *** ONE OR MORE MARGINALS FAIL ***")
    print()
    print("    Do NOT proceed as if this is clean.")
    print("    If is_early_in_cart failed, it must")
    print("    be excluded from the ML efficacy")
    print("    features, or reported as a known gap.")
    print("    This is a real limitation, not a")
    print("    rounding error.")


# ══════════════════════════════════════════
# [7] QUALITY - scored on HELD-OUT data only
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("[7] QUALITY  (held-out eval set only)")
print("=" * 68)

score_cols = BASE_COLS
qmeta = SingleTableMetadata()
qmeta.detect_from_dataframe(eval_df[score_cols])
for c in score_cols:
    if c == 'days_since_prior_order':
        qmeta.update_column(column_name=c,
                            sdtype='numerical')
    else:
        qmeta.update_column(column_name=c,
                            sdtype='categorical')

n = min(len(eval_df), len(synth), 9000)
q = evaluate_quality(
    real_data=eval_df[score_cols].sample(
        n=n, random_state=SEED),
    synthetic_data=synth[score_cols].sample(
        n=n, random_state=SEED),
    metadata=qmeta,
    verbose=True)
score = q.get_score()

print(f"\n    Quality: {score:.4f}")
print("\n    This is scored against orders the model")
print("    has NEVER seen, with no post-processing.")
print("    It is not comparable to the old 93.47%,")
print("    which was measured against possibly-seen")
print("    rows after real hours were injected.")


# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
report = {
    'epochs': EPOCHS,
    'small_max': SMALL_MAX,
    'large_min': LARGE_MIN,
    'train_rows': int(len(X)),
    'quality': float(score),
    'marginals_all_pass': bool(all_ok),
    'marginals': marginals,
}
with open('data/v3_training_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 68)
print("DONE")
print("=" * 68)
print(f"""
  quality (held-out):  {score:.4f}
  marginals all pass:  {all_ok}

  data/ctgan_v3.pkl
  data/synthetic_v3.csv
  data/v3_training_report.json

  NEXT: only if the marginals pass, run the full
  honest evaluation - quality, ML efficacy,
  correlation, a REAL privacy test, and the
  conditional aisle-shift reproduction test.

  If a marginal failed, fix that first.
""")