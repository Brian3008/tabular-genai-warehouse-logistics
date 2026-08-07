"""
tabsyn_marginal_impact.py
=========================
Verbatim copy of test_marginal_impact.py for the TabSyn
bench (tabsyn_bench_contract.md). Measurement code is
character-identical to the original; deviations declared.

DECLARED DEVIATIONS (nothing else changed):
 1. I/O head: --synthetic-csv and --out-dir replace the
    hard-coded 'data/synthetic_v3.csv' and the hard-coded
    'data/marginal_impact.json'. The original JSON is
    never touched.
 2. The docstring/prints keep the original's CTGAN
    narrative (the two known CTGAN marginal errors) - it
    describes the audit story, not the file being scored.
    The printed "two known errors" section is measured
    live from whatever file is scored.

PROTOCOL NOTE (contract rule 4): for TabSyn this script is
run once per independent sample (5 samples), each run
identical to this known-answer procedure; the comparison
with CTGAN's 89.2% +/- 0.7% is mean-vs-mean under the
identical protocol. This file itself changes nothing about
the procedure.

READS:  data/v3_compare.csv, data/v3_eval.csv,
        data/v3_train_order_ids.csv, <--synthetic-csv>
WRITES: <--out-dir>/marginal_impact.json

Known-answer gate (contract rule 1): with
--synthetic-csv data/synthetic_v3.csv this must reproduce
data/marginal_impact.json: mean_efficacy 0.891553,
noise_range 0.019427, observed_range 0.021638,
stable true, eic_matters false (RF thread-ordering float
noise is the only allowed drift; Jul 17 re-run matched all
conclusions with <=4th-decimal drift).
"""

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument('--synthetic-csv', required=True)
_ap.add_argument('--out-dir', required=True)
_args = _ap.parse_args()
SYNTH_CSV = _args.synthetic_csv
OUT_DIR = _args.out_dir

import pandas as pd
import numpy as np
import random
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs(OUT_DIR, exist_ok=True)
assert os.path.normpath(OUT_DIR).replace('\\', '/').find(
    'results/tabsyn') != -1, \
    'FATAL: out-dir must live under results/tabsyn (protection rules)'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = 'is_reorder'
FEATURES = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_early_in_cart', 'days_since_prior_order',
]
SEEDS = [42, 7, 123, 2024, 999]

print("=" * 70)
print("DO THE MARGINAL ERRORS DISTORT THE EFFICACY FIGURE?")
print(f"scored file: {SYNTH_CSV}")
print("=" * 70)

# ══════════════════════════════════════════
# LOAD + PROVE DISJOINTNESS
# ══════════════════════════════════════════
comp = pd.read_csv('data/v3_compare.csv')
synth_full = pd.read_csv(SYNTH_CSV)
pseudo = pd.read_csv('data/v3_eval.csv')
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])

assert len(set(comp['order_id']) & tr_ids) == 0, \
    "FATAL: compare contains training orders"
assert len(set(pseudo['order_id']) & tr_ids) == 0, \
    "FATAL: eval contains training orders"
assert len(set(pseudo['order_id'])
           & set(comp['order_id'])) == 0, \
    "FATAL: eval overlaps compare"
print("\n  [ASSERTED] compare, eval and train are all")
print("             mutually disjoint")

# SIZE MATCHING - measured to matter (see docstring).
# The pseudo-synthetic set and the real synthetic set
# must train classifiers on the SAME number of rows, or
# the noise floor is not comparable.
N_MATCH = min(len(pseudo), len(synth_full))
synth = synth_full.sample(n=N_MATCH,
                          random_state=SEED)
pseudo = pseudo.sample(n=N_MATCH, random_state=SEED)

print(f"\n  synthetic (full):     "
      f"{len(synth_full):,} rows")
print(f"  pseudo-synthetic:     "
      f"{len(pseudo):,} rows (real, held out)")
print(f"  BOTH subsampled to:   {N_MATCH:,} rows")
print("  (a classifier on less data is noisier -")
print("   measured: 10k gives 2.49% spread vs 2.04%")
print("   at 50k. Sizes MUST match or the noise floor")
print("   is wrong.)")


def num(d, cols):
    return d[cols].apply(
        pd.to_numeric, errors='coerce').fillna(0)


print("\n  The two known errors:")
for c in ['is_reorder', 'is_early_in_cart']:
    r = float(pd.to_numeric(comp[c]).mean())
    s = float(pd.to_numeric(synth_full[c]).mean())
    print(f"    {c:<20} real {r:.3f}  "
          f"synth {s:.3f}  ({(s - r) / r:+.1%})")


def run(train_df, feats, seed):
    """Fit on train_df, score on a real test split."""
    tr, te = train_test_split(
        comp, test_size=0.3, random_state=seed,
        stratify=comp[TARGET])
    Xte = num(te, feats)

    real_m = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1)
    real_m.fit(num(tr, feats), tr[TARGET])
    ra = accuracy_score(te[TARGET],
                        real_m.predict(Xte))

    m = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(num(train_df, feats), train_df[TARGET])
    a = accuracy_score(te[TARGET], m.predict(Xte))

    maj = float(te[TARGET].value_counts(
        normalize=True).max())
    floor = maj / ra if ra else 0
    eff = a / ra if ra else 0
    head = 1.0 - floor
    above = (eff - floor) / head if head > 0 else 0
    return eff, floor, above


# ══════════════════════════════════════════
# [1] THE SPLIT-NOISE FLOOR  (the yardstick)
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[1] THE SPLIT-NOISE FLOOR - the one yardstick")
print("=" * 70)
print("""
    A PERFECT model still varies across test splits. We
    measure that floor by using a REAL held-out chunk AS
    IF it were synthetic - its true efficacy is ~1.0 by
    construction, so its spread IS pure split noise.

    Everything below is judged against this. No other
    thresholds.
""")

noise_effs = [run(pseudo, FEATURES, s)[0]
              for s in SEEDS]
noise_range = max(noise_effs) - min(noise_effs)

print(f"    baseline efficacies: "
      f"{[f'{e:.1%}' for e in noise_effs]}")
print(f"\n    SPLIT-NOISE FLOOR: {noise_range:.2%}")
print("    (any difference smaller than this is noise)")


# ══════════════════════════════════════════
# [2] DOES THE CLASSIFIER RELY ON
#     is_early_in_cart?
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[2] DOES THE CLASSIFIER EVEN USE IT?")
print("=" * 70)
print("""
    Permutation importance on REAL test data: shuffle a
    feature, measure the accuracy drop. That is TRUE
    dependence - unlike Gini importance, which is
    inflated for high-cardinality columns like aisle_id.

    Reported as a share of the LEARNABLE HEADROOM, not as
    a raw accuracy drop. A 0.005 drop sounds tiny, but if
    the classifier's total headroom above the do-nothing
    floor is only 0.09, that drop is 5.5% of everything
    there is to learn - not negligible at all.
""")

real_tr, real_te = train_test_split(
    comp, test_size=0.3, random_state=SEED,
    stratify=comp[TARGET])
clf = RandomForestClassifier(
    n_estimators=100, random_state=SEED, n_jobs=-1)
clf.fit(num(real_tr, FEATURES), real_tr[TARGET])

Xte = num(real_te, FEATURES)
base_acc = accuracy_score(
    real_te[TARGET], clf.predict(Xte))
maj = float(real_te[TARGET].value_counts(
    normalize=True).max())
headroom = base_acc - maj

print(f"    real classifier accuracy: {base_acc:.4f}")
print(f"    majority-class accuracy:  {maj:.4f}")
print(f"    LEARNABLE HEADROOM:       {headroom:.4f}")

rng = np.random.RandomState(SEED)
perm = {}
for f in FEATURES:
    drops = []
    for _ in range(5):
        Xp = Xte.copy()
        Xp[f] = rng.permutation(Xp[f].values)
        drops.append(base_acc - accuracy_score(
            real_te[TARGET], clf.predict(Xp)))
    perm[f] = float(np.mean(drops))

print(f"\n    {'feature':<26}{'acc drop':>10}"
      f"{'% of headroom':>15}")
print("    " + "-" * 52)
for f in sorted(perm, key=perm.get, reverse=True):
    share = perm[f] / headroom if headroom > 0 else 0
    print(f"    {f:<26}{perm[f]:>10.4f}{share:>14.1%}")

eic = perm['is_early_in_cart']
eic_share = eic / headroom if headroom > 0 else 0
print(f"\n    is_early_in_cart uses "
      f"{eic_share:.1%} of the headroom")


# ══════════════════════════════════════════
# [3] DOES DROPPING IT CHANGE THE RESULT?
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[3] EFFICACY WITH AND WITHOUT IT")
print("=" * 70)
print("""
    Judged against the split-noise floor: if dropping the
    feature moves the result by LESS than split noise
    alone would, then its marginal error cannot be
    driving the headline number.
""")

no_eic = [f for f in FEATURES
          if f != 'is_early_in_cart']

e_all, fl_all, ab_all = run(synth, FEATURES, SEED)
e_no, fl_no, ab_no = run(synth, no_eic, SEED)

print(f"    {'features':<26}{'MLeff':>9}{'floor':>9}"
      f"{'above floor':>13}")
print("    " + "-" * 58)
print(f"    {'all 5':<26}{e_all:>8.1%}{fl_all:>9.1%}"
      f"{ab_all:>12.1%}")
print(f"    {'without is_early_in_cart':<26}"
      f"{e_no:>8.1%}{fl_no:>9.1%}{ab_no:>12.1%}")

delta = abs(e_all - e_no)
print(f"\n    change in MLeff:   {delta:.2%}")
print(f"    split-noise floor: {noise_range:.2%}")

if delta <= noise_range:
    print("\n    => The change is WITHIN split noise.")
    print("       is_early_in_cart's marginal error is")
    print("       NOT driving the result.")
    eic_matters = False
else:
    print("\n    => The change EXCEEDS split noise.")
    print("       The feature matters, and its wrong")
    print("       distribution is a genuine concern that")
    print("       must be reported.")
    eic_matters = True


# ══════════════════════════════════════════
# [4] IS THE RESULT STABLE AT ALL?
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[4] IS THE RESULT STABLE ACROSS SPLITS?")
print("=" * 70)
print("""
    Never checked before the number was reported. An 88%
    that swings between 80% and 95% depending on the
    random split is not a result - it is noise.
""")

effs, abs_ = [], []
for s in SEEDS:
    e, f, a = run(synth, FEATURES, s)
    effs.append(e)
    abs_.append(a)
    print(f"    seed {s:<6} MLeff {e:.1%}   "
          f"above floor {a:.1%}")

e_mean = float(np.mean(effs))
e_std = float(np.std(effs))
a_mean = float(np.mean(abs_))
a_std = float(np.std(abs_))
e_range = max(effs) - min(effs)

print(f"\n    MLeff        {e_mean:.1%} +/- {e_std:.1%}"
      f"   (range {e_range:.2%})")
print(f"    above floor  {a_mean:.1%} +/- {a_std:.1%}")
print(f"\n    OBSERVED range:    {e_range:.2%}")
print(f"    SPLIT-NOISE floor: {noise_range:.2%}")

stable = e_range <= noise_range * 2.0
if stable:
    print("\n    => STABLE. The spread is within what")
    print("       split noise alone produces.")
else:
    print("\n    => UNSTABLE. The spread EXCEEDS split")
    print("       noise. Report the RANGE, never a")
    print("       single number.")


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"""
    THE YARDSTICK (measured, not guessed):
      split-noise floor  {noise_range:.2%}

    is_early_in_cart (a FEATURE):
      uses {eic_share:.1%} of the learnable headroom
      dropping it changes MLeff by {delta:.2%}
      split noise alone is {noise_range:.2%}
      => {'MATTERS - report it' if eic_matters
          else 'does NOT drive the result'}

    STABILITY:
      MLeff {e_mean:.1%} +/- {e_std:.1%}
      observed range {e_range:.2%} vs noise {noise_range:.2%}
      => {'STABLE' if stable else 'UNSTABLE - report the range'}

    HONEST HEADLINE:
      ML efficacy {e_mean:.1%}, which is {a_mean:.0%} of the
      learnable signal above the {fl_all:.0%} do-nothing floor.
      Never report the bare percentage without the floor.
""")

json.dump({
    'scored_file': SYNTH_CSV,
    'noise_range': float(noise_range),
    'perm_importance': perm,
    'headroom': float(headroom),
    'eic_share_of_headroom': float(eic_share),
    'efficacy_all': float(e_all),
    'efficacy_no_eic': float(e_no),
    'delta': float(delta),
    'eic_matters': bool(eic_matters),
    'seed_effs': [float(x) for x in effs],
    'seed_above_floor': [float(x) for x in abs_],
    'observed_range': float(e_range),
    'stable': bool(stable),
    'mean_efficacy': e_mean,
    'mean_above_floor': a_mean,
}, open(os.path.join(OUT_DIR, 'marginal_impact.json'),
        'w'), indent=2, default=float)
print(f"Saved {os.path.join(OUT_DIR, 'marginal_impact.json')}")
print("=" * 70)
