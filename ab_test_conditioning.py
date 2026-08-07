"""
ab_test_conditioning.py
=======================
THE QUESTION
------------
The 300-epoch model INVENTS structure that is not in
the data:

    is_reorder gap between small and large baskets
        real:      0.0061   (essentially zero -
                             basket size tells you
                             NOTHING about reorder rate)
        synthetic: 0.0654   (10x larger - fabricated)

And it over-separates the aisle mix by ~128%, growing
with training (111% at 30 epochs -> 128% at 300).

Crucially, it does this EVEN WHEN GENERATING FREELY
(unconditional: 127.2%). So it is not the conditioning
mechanism at sampling time.

TWO EXPLANATIONS, AND I CANNOT TELL THEM APART
----------------------------------------------
(i)  INHERENT. Conditional GANs sharpen class
     boundaries. Unfixable. Report honestly.

(ii) THE TRAINING COLUMN ITSELF. Including
     order_size_grp as a training column - regardless
     of whether we condition on it at sampling time -
     teaches the generator to treat small and large as
     distinct modes. It may be using order_size_grp as
     a cheap signal to satisfy the discriminator, and
     over-fitting a spurious association to it.

These lead to COMPLETELY DIFFERENT fixes. Guessing
would be the third wrong hypothesis in this project.
So: controlled experiment.

THE EXPERIMENT
--------------
Two models. Identical data, identical config, identical
seed. ONE variable changes.

    Model A:  6 base columns + order_size_grp
    Model B:  6 base columns ONLY (no order_size_grp)

For Model B we cannot condition at sampling time - the
column does not exist. Instead we generate freely, then
LABEL the rows post-hoc.

THE LABELLING PROBLEM - and how it is handled
---------------------------------------------
Model B's rows have no order_id and no basket size, so
we cannot bucket them by "items in the order".

But we do not need to. The question is whether the
model has learned a spurious ASSOCIATION between
basket-size-like structure and is_reorder.

is_early_in_cart is the strongest real correlate of
basket size (real gap 0.3505 - small baskets are ~50%
early-cart, large ~15%, because 3 early items out of 6
is 50% but 3 out of 30 is 10%).

So we use is_early_in_cart as a PROXY axis in BOTH
models and ask: does the reorder gap along that axis
get fabricated in Model B too?

    If YES  -> the over-separation is inherent to the
               data/architecture. order_size_grp is not
               the cause. Report honestly.

    If NO   -> including order_size_grp as a training
               column IS the cause. Redesign.

We measure the SAME proxy axis in both models, so the
comparison is like-for-like.

CONFIG
------
60 epochs. We know the effect is already present at 30
(111%), so 60 is enough to see it while keeping the run
to ~1 hour total for both models.

Reduced epochs affect BOTH models equally, so the
comparison is valid even though neither is fully
trained.
"""

import pandas as pd
import numpy as np
import random
import torch
import json
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 60
SMALL_MAX = 10
LARGE_MIN = 14
N_GEN = 60000
DOMINANT = [24, 83]
COND_COL = 'order_size_grp'

BASE_COLS = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_reorder', 'is_early_in_cart',
    'days_since_prior_order',
]

print("=" * 70)
print("A/B TEST - is order_size_grp the cause?")
print("=" * 70)
print(f"""
  Model A: 6 base columns + order_size_grp
  Model B: 6 base columns ONLY

  Identical data, config, seed. ONE variable changes.
  {EPOCHS} epochs each (~1 hour total).
""")

# ══════════════════════════════════════════
# DATA
# ══════════════════════════════════════════
print("[1] Loading + verifying disjointness ...")
train = pd.read_csv('data/v3_train.csv')
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
ev_ids = set(pd.read_csv(
    'data/v3_eval.csv')['order_id'])
assert len(tr_ids & ev_ids) == 0
assert set(train['order_id']) <= tr_ids
print(f"    [OK] {len(train):,} rows, disjoint")

osize = train.groupby('order_id').size()
train['n_items'] = train['order_id'].map(osize)
train[COND_COL] = np.where(
    train['n_items'] <= SMALL_MAX, 'small',
    np.where(train['n_items'] >= LARGE_MIN,
             'large', 'mid'))
td = train[train[COND_COL] != 'mid'].copy()
print(f"    2-class rows: {len(td):,}")


def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


def aisle_tvd(ds, dl):
    if len(ds) < 100 or len(dl) < 100:
        return 0.0
    sm = ds['aisle_id'].value_counts(
        normalize=True)
    lg = dl['aisle_id'].value_counts(
        normalize=True)
    smt = sm[~sm.index.isin(DOMINANT)]
    lgt = lg[~lg.index.isin(DOMINANT)]
    if smt.sum() == 0 or lgt.sum() == 0:
        return 0.0
    return tvd(smt / smt.sum(), lgt / lgt.sum())


def gap(ds, dl, col):
    a = pd.to_numeric(ds[col],
                      errors='coerce').mean()
    b = pd.to_numeric(dl[col],
                      errors='coerce').mean()
    return abs(a - b)


# ══════════════════════════════════════════
# THE PROXY AXIS - usable in BOTH models
# ══════════════════════════════════════════
# Model B has no order_size_grp, so we need an axis
# that exists in both. is_early_in_cart is the
# strongest real correlate of basket size:
#   small baskets ~50% early-cart (3 of 6)
#   large baskets ~15% early-cart (3 of 30)
# So it is a genuine, model-visible proxy for size.
PROXY = 'is_early_in_cart'

print("\n[2] Real benchmarks ...")

# --- along the TRUE axis (order_size_grp) ---
r_s = td[td[COND_COL] == 'small']
r_l = td[td[COND_COL] == 'large']
REAL_TRUE = {
    'aisle_tvd': aisle_tvd(r_s, r_l),
    'reorder':   gap(r_s, r_l, 'is_reorder'),
    'hour':      gap(r_s, r_l, 'order_hour_of_day'),
}

# --- along the PROXY axis (is_early_in_cart) ---
p_1 = td[td[PROXY] == 1]
p_0 = td[td[PROXY] == 0]
REAL_PROXY = {
    'aisle_tvd': aisle_tvd(p_1, p_0),
    'reorder':   gap(p_1, p_0, 'is_reorder'),
    'hour':      gap(p_1, p_0, 'order_hour_of_day'),
}

print(f"\n    TRUE axis (order_size_grp):")
for k, v in REAL_TRUE.items():
    print(f"      {k:<12} {v:.4f}")
print(f"\n    PROXY axis ({PROXY}):")
for k, v in REAL_PROXY.items():
    print(f"      {k:<12} {v:.4f}")
print("\n    (the proxy is what we can measure in")
print("     BOTH models, so it is the basis of the")
print("     like-for-like comparison)")


def build_meta(d, cols):
    m = SingleTableMetadata()
    m.detect_from_dataframe(d[cols])
    for c in cols:
        if c == 'days_since_prior_order':
            m.update_column(column_name=c,
                            sdtype='numerical')
        else:
            m.update_column(column_name=c,
                            sdtype='categorical')
    return m


def train_model(cols, tag):
    print("\n" + "=" * 70)
    print(f"TRAINING {tag}  ({len(cols)} columns)")
    print("=" * 70)
    print(f"  columns: {cols}\n")

    X = td[cols].copy()
    m = build_meta(td, cols)

    # reset seeds so both models start identically
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = CTGANSynthesizer(
        m,
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
    model.save(f'data/ab_{tag}.pkl')
    return model


# ══════════════════════════════════════════
# MODEL A - with order_size_grp
# ══════════════════════════════════════════
mA = train_model(BASE_COLS + [COND_COL], 'A')
genA = mA.sample(num_rows=N_GEN)

# ══════════════════════════════════════════
# MODEL B - WITHOUT order_size_grp
# ══════════════════════════════════════════
mB = train_model(BASE_COLS, 'B')
genB = mB.sample(num_rows=N_GEN)


# ══════════════════════════════════════════
# MEASURE - along the PROXY axis, both models
# ══════════════════════════════════════════
def measure_proxy(g, tag):
    a = g[pd.to_numeric(g[PROXY],
                        errors='coerce') == 1]
    b = g[pd.to_numeric(g[PROXY],
                        errors='coerce') == 0]
    out = {
        'aisle_tvd': aisle_tvd(a, b),
        'reorder':   gap(a, b, 'is_reorder'),
        'hour':      gap(a, b, 'order_hour_of_day'),
    }
    print(f"\n  {tag}  (n={len(g):,}, "
          f"{PROXY}=1: {len(a):,})")
    print(f"    {'metric':<12}{'real':>9}"
          f"{'synth':>9}{'ratio':>9}")
    print("    " + "-" * 40)
    for k in out:
        r = REAL_PROXY[k]
        s = out[k]
        ratio = s / r if r > 0 else 0
        flag = "  <-- FABRICATED" \
            if ratio > 2.0 else (
                "  <-- over" if ratio > 1.25 else "")
        print(f"    {k:<12}{r:>9.4f}{s:>9.4f}"
              f"{ratio:>8.1%}{flag}")
    return out


print("\n" + "=" * 70)
print(f"RESULTS - measured along the PROXY axis")
print(f"({PROXY}) in BOTH models")
print("=" * 70)

resA = measure_proxy(genA, "MODEL A (with order_size_grp)")
resB = measure_proxy(genB, "MODEL B (WITHOUT order_size_grp)")

# also measure A along its TRUE axis, for reference
print("\n  MODEL A along its TRUE axis "
      "(order_size_grp) - for reference")
a_s = genA[genA[COND_COL] == 'small']
a_l = genA[genA[COND_COL] == 'large']
print(f"    {'metric':<12}{'real':>9}"
      f"{'synth':>9}{'ratio':>9}")
print("    " + "-" * 40)
for k, fn in [
    ('aisle_tvd', lambda: aisle_tvd(a_s, a_l)),
    ('reorder',
     lambda: gap(a_s, a_l, 'is_reorder')),
    ('hour',
     lambda: gap(a_s, a_l, 'order_hour_of_day')),
]:
    r = REAL_TRUE[k]
    s = fn()
    ratio = s / r if r > 0 else 0
    print(f"    {k:<12}{r:>9.4f}{s:>9.4f}"
          f"{ratio:>8.1%}")


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

rA = (resA['reorder'] / REAL_PROXY['reorder']
      if REAL_PROXY['reorder'] > 0 else 0)
rB = (resB['reorder'] / REAL_PROXY['reorder']
      if REAL_PROXY['reorder'] > 0 else 0)
tA = (resA['aisle_tvd'] / REAL_PROXY['aisle_tvd']
      if REAL_PROXY['aisle_tvd'] > 0 else 0)
tB = (resB['aisle_tvd'] / REAL_PROXY['aisle_tvd']
      if REAL_PROXY['aisle_tvd'] > 0 else 0)

print(f"\n  reorder gap ratio (proxy axis):")
print(f"    Model A (with size col):    {rA:.1%}")
print(f"    Model B (without size col): {rB:.1%}")
print(f"\n  aisle TVD ratio (proxy axis):")
print(f"    Model A: {tA:.1%}")
print(f"    Model B: {tB:.1%}")

diff = abs(rA - rB)
print()
if diff > 0.30:
    if rB < rA:
        print("""
  ==> order_size_grp IS THE CAUSE.

  Removing it from training substantially reduces the
  fabricated structure. The generator was using the
  size column as a cheap signal for the discriminator
  and over-fitting a spurious association to it.

  FIX: redesign the conditioning so the size column
  is not a training input. Options to test:
    - condition via SDV constraints instead
    - train unconditional, filter/reweight at
      sampling time
    - use a continuous size feature rather than a
      hard 2-class label
""")
    else:
        print("""
  ==> Model B is WORSE. Removing the size column made
  the fabrication worse, not better. Unexpected -
  investigate before acting.
""")
else:
    print("""
  ==> order_size_grp IS NOT THE CAUSE.

  Both models fabricate the structure to a similar
  degree. The over-separation is INHERENT - it comes
  from CTGAN's architecture or from the data itself,
  not from including the size column.

  This means:
    - there is no design fix available
    - the limitation is REAL and must be REPORTED
    - the honest statement is: "CTGAN exaggerates
      differences between subgroups, most severely
      where the real difference is smallest. Synthetic
      data overstates the demand gap between small and
      large orders by ~28%."

  That is a legitimate, precise finding about
  conditional tabular GAN behaviour - and a genuinely
  useful contribution.
""")

json.dump({
    'real_true': REAL_TRUE,
    'real_proxy': REAL_PROXY,
    'model_A': resA,
    'model_B': resB,
    'reorder_ratio_A': rA,
    'reorder_ratio_B': rB,
}, open('data/ab_test_results.json', 'w'),
    indent=2, default=float)
print("Saved data/ab_test_results.json")
print("=" * 70)