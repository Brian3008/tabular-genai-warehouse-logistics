"""
diagnose_exaggeration.py
========================
The 300-epoch model reproduces the conditional aisle
shift, but OVERSTATES it:

    real  tail TVD  0.1296
    synth tail TVD  0.1660
    recovery        128.1%

Synthetic data that exaggerates a demand gap would
mislead a fleet planner. Before reporting this as an
inherent limitation, test WHY. Four candidates - I do
not know which, so all four get measured.

(A) SAMPLING ARTEFACT
    diagnose_v3 generated 30k small + 30k large - a
    forced 50/50. The real split is 40/60. If the
    exaggeration shrinks under NATURAL proportions,
    it is an artefact of how I sampled, not a property
    of the model.

(B) DROPPING 'mid' CREATED ARTIFICIAL SEPARATION
    Training used only small and large, deleting the
    transition zone. That may have taught the model
    the classes are more distinct than they are.
    Test: does unconditional sampling (letting the
    model choose the class freely) show the same
    exaggeration?

(C) GENUINE MODE SEPARATION
    Conditional GANs are known to over-commit to
    conditioned classes. If (A) and (B) are ruled out,
    this is the explanation and it is inherent.

(D) IT GROWS WITH TRAINING
    30 epochs -> 108%
    300 epochs -> 128%
    If exaggeration INCREASES with training, the model
    is sharpening the class separation as it converges.
    That is strong evidence for (C).
    We re-measure the saved 30-epoch model with the
    IDENTICAL method to make this comparison valid -
    the earlier numbers used different sampling.

ALSO CHECKED
------------
Is the exaggeration specific to aisle, or does the
model over-separate EVERY conditioned feature? If
is_reorder and is_early_in_cart are also over-separated,
it is one coherent phenomenon, not an aisle quirk.

No retraining. Minutes.
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
COND_COL = 'order_size_grp'
N_TOTAL = 60000

print("=" * 68)
print("WHY IS THE CONDITIONAL SHIFT EXAGGERATED?")
print("=" * 68)

# ══════════════════════════════════════════
# REAL BENCHMARK
# ══════════════════════════════════════════
print("\nLoading...")
train = pd.read_csv('data/v3_train.csv')
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
assert set(train['order_id']) <= tr_ids
print("  [OK] disjointness verified")

osize = train.groupby('order_id').size()
train['n_items'] = train['order_id'].map(osize)
train[COND_COL] = np.where(
    train['n_items'] <= SMALL_MAX, 'small',
    np.where(train['n_items'] >= LARGE_MIN,
             'large', 'mid'))
td = train[train[COND_COL] != 'mid'].copy()

real_share = td[COND_COL].value_counts(
    normalize=True)
p_small = real_share['small']
p_large = real_share['large']
print(f"\n  REAL class split: "
      f"small {p_small:.1%} / large {p_large:.1%}")


def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


def tail_tvd(ds, dl, early_only=False):
    """Aisle-mix shift, excluding aisles 24/83.

    early_only=True restricts to is_early_in_cart==1,
    so every order contributes at most 3 rows. This
    removes the arithmetic artefact that a 30-item
    basket mechanically touches more aisles than a
    6-item one.

    Both views are reported: recovery is largely
    unaffected (both sides inflate equally), but the
    ABSOLUTE TVD is only meaningful size-controlled -
    that is the basis on which the axis was validated
    (0.1296).
    """
    if early_only:
        ds = ds[pd.to_numeric(
            ds['is_early_in_cart'],
            errors='coerce') == 1]
        dl = dl[pd.to_numeric(
            dl['is_early_in_cart'],
            errors='coerce') == 1]
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
    return tvd(smt / smt.sum(),
               lgt / lgt.sum())


def gap(ds, dl, col):
    """Difference between classes on a feature.
    Tells us if the model over-separates
    EVERYTHING, not just aisle."""
    a = pd.to_numeric(ds[col],
                      errors='coerce').mean()
    b = pd.to_numeric(dl[col],
                      errors='coerce').mean()
    return abs(a - b)


# real benchmarks
r_small = td[td[COND_COL] == 'small']
r_large = td[td[COND_COL] == 'large']

REAL_TVD = tail_tvd(r_small, r_large)
REAL_TVD_SC = tail_tvd(r_small, r_large,
                       early_only=True)
REAL_GAPS = {
    c: gap(r_small, r_large, c)
    for c in ['is_reorder', 'is_early_in_cart',
              'order_hour_of_day']
}

print(f"\n  REAL aisle tail TVD")
print(f"    all rows:        {REAL_TVD:.4f}")
print(f"    size-controlled: {REAL_TVD_SC:.4f}"
      f"   <-- the validated basis")
print(f"\n  REAL class gaps:")
for c, v in REAL_GAPS.items():
    print(f"    {c:<22} {v:.4f}")


def measure(model, n_small, n_large, label):
    """Generate with a given class split and
    measure how much the model separates the
    classes on EVERY conditioned feature."""
    gs = model.sample_from_conditions(
        conditions=[Condition(
            num_rows=n_small,
            column_values={COND_COL: 'small'})])
    gl = model.sample_from_conditions(
        conditions=[Condition(
            num_rows=n_large,
            column_values={COND_COL: 'large'})])

    t = tail_tvd(gs, gl)
    t_sc = tail_tvd(gs, gl, early_only=True)
    g = {c: gap(gs, gl, c)
         for c in REAL_GAPS}

    rec = t / REAL_TVD if REAL_TVD else 0
    rec_sc = (t_sc / REAL_TVD_SC
              if REAL_TVD_SC else 0)

    print(f"\n  {label}")
    print(f"    n: {n_small:,} small / "
          f"{n_large:,} large")
    print(f"    aisle TVD all-rows:   {t:.4f}"
          f"   (real {REAL_TVD:.4f})"
          f"   recovery {rec:.1%}")
    print(f"    aisle TVD size-ctrl:  {t_sc:.4f}"
          f"   (real {REAL_TVD_SC:.4f})"
          f"   recovery {rec_sc:.1%}")
    for c in REAL_GAPS:
        rg = REAL_GAPS[c]
        sg = g[c]
        ratio = sg / rg if rg > 0 else 0
        flag = "  <-- over-separated" \
            if ratio > 1.20 else ""
        print(f"    {c:<22} gap {sg:.4f}  "
              f"(real {rg:.4f})  "
              f"{ratio:>6.1%}{flag}")
    return {'label': label,
            'tvd': float(t),
            'tvd_sc': float(t_sc),
            'recovery': float(rec),
            'recovery_sc': float(rec_sc),
            'gaps': {c: float(g[c])
                     for c in g}}


results = []

# ══════════════════════════════════════════
# 300-EPOCH MODEL
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("300-EPOCH MODEL (ctgan_v3.pkl)")
print("=" * 68)

m300 = CTGANSynthesizer.load('data/ctgan_v3.pkl')

# (A) forced 50/50 - what diagnose_v3 did
results.append(measure(
    m300, N_TOTAL // 2, N_TOTAL // 2,
    "(A) FORCED 50/50 - as diagnose_v3 did"))

# (A') natural proportions
ns = int(N_TOTAL * p_small)
nl = N_TOTAL - ns
results.append(measure(
    m300, ns, nl,
    "(A') NATURAL 40/60 proportions"))

# ══════════════════════════════════════════
# (B) UNCONDITIONAL - model picks class itself
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("(B) UNCONDITIONAL - model chooses the class")
print("=" * 68)
print("\n  If the model over-separates only when")
print("  FORCED, but not when free, the problem is")
print("  the conditioning mechanism, not the model.")

free = m300.sample(num_rows=N_TOTAL)
fs = free[free[COND_COL] == 'small']
fl = free[free[COND_COL] == 'large']

print(f"\n  model's own class split: "
      f"small {len(fs)/len(free):.1%} / "
      f"large {len(fl)/len(free):.1%}")
print(f"  (real: small {p_small:.1%} / "
      f"large {p_large:.1%})")

if len(fs) > 1000 and len(fl) > 1000:
    t_free = tail_tvd(fs, fl)
    print(f"\n  UNCONDITIONAL aisle tail TVD: "
          f"{t_free:.4f}")
    print(f"  recovery: {t_free/REAL_TVD:.1%}")
    for c in REAL_GAPS:
        sg = gap(fs, fl, c)
        rg = REAL_GAPS[c]
        print(f"    {c:<22} gap {sg:.4f}  "
              f"(real {rg:.4f})  "
              f"{sg/rg if rg>0 else 0:>6.1%}")
    results.append({
        'label': '(B) unconditional',
        'tvd': float(t_free),
        'recovery': float(t_free / REAL_TVD)})
else:
    print("\n  too few rows in one class")
    t_free = None

# ══════════════════════════════════════════
# (D) DOES IT GROW WITH TRAINING?
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("(D) DOES EXAGGERATION GROW WITH TRAINING?")
print("=" * 68)
print("\n  Re-measuring the saved 30-epoch model with")
print("  the IDENTICAL method, so the comparison is")
print("  valid. (The earlier 108% used different")
print("  sampling - not comparable.)")

try:
    m30 = CTGANSynthesizer.load(
        'data/screen_10_14.pkl')
    r30 = measure(
        m30, N_TOTAL // 2, N_TOTAL // 2,
        "30-epoch model, forced 50/50")
    results.append(r30)

    r300 = results[0]
    print(f"\n  LIKE-FOR-LIKE COMPARISON:")
    print(f"     30 epochs: recovery "
          f"{r30['recovery']:.1%}")
    print(f"    300 epochs: recovery "
          f"{r300['recovery']:.1%}")
    growth = r300['recovery'] - r30['recovery']
    print(f"    change:    {growth:+.1%}")
    if growth > 0.05:
        print("\n    EXAGGERATION GROWS WITH TRAINING.")
        print("    The model sharpens the class")
        print("    separation as it converges.")
        print("    -> strong evidence for (C):")
        print("       inherent mode separation.")
    elif growth < -0.05:
        print("\n    Exaggeration SHRINKS with training.")
        print("    More epochs would help.")
    else:
        print("\n    Roughly flat. Training is not")
        print("    the driver.")
except Exception as e:
    print(f"\n  could not load 30-epoch model: {e}")
    growth = None

# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 68)
print("VERDICT")
print("=" * 68)

forced = results[0]['recovery']
natural = results[1]['recovery']

print(f"\n  (A)  forced 50/50:   {forced:.1%}")
print(f"  (A') natural 40/60:  {natural:.1%}")
if t_free is not None:
    print(f"  (B)  unconditional:  "
          f"{t_free/REAL_TVD:.1%}")

print()
if abs(forced - natural) < 0.05:
    print("  (A) RULED OUT. Sampling proportions do")
    print("      not explain it. The exaggeration is")
    print("      the same either way.")
else:
    print("  (A) MATTERS. The forced 50/50 split")
    print("      inflated the measurement. Use the")
    print("      natural proportions figure.")

if t_free is not None:
    if abs(t_free / REAL_TVD - forced) < 0.10:
        print("\n  (B) RULED OUT. The model over-")
        print("      separates even when free to")
        print("      choose. Not a conditioning")
        print("      mechanism artefact.")
    else:
        print("\n  (B) MATTERS. The model separates")
        print("      classes far more when FORCED")
        print("      than when free. The conditioning")
        print("      mechanism is amplifying.")

print("""
  READ THE FEATURE GAPS ABOVE.

  If aisle, is_reorder AND is_early_in_cart are ALL
  over-separated, this is ONE coherent phenomenon -
  CTGAN over-commits to conditioned classes - and
  the two 'marginal failures' are the same bug as
  the aisle exaggeration, not three separate ones.

  That is a clean, honest, reportable finding.
""")

json.dump(results,
          open('data/exaggeration_diag.json', 'w'),
          indent=2)
print("Saved data/exaggeration_diag.json")
print("=" * 68)