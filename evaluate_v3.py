"""
evaluate_v3.py
==============
The three remaining headline numbers, measured
honestly on the clean v3 model.

WHAT CHANGED, AND WHY EACH NUMBER IS NOW DIFFERENT
--------------------------------------------------

ML EFFICACY
  v2 trained on FINAL_synthetic_orders.csv - synthetic
  rows whose order_hour_of_day had been OVERWRITTEN
  with real hours by the post-processing step. So
  "trained on synthetic" was not true. Three of the
  eight features were partly real data.

  v3 trains on synthetic_v3.csv - no injection, no
  calibration. And it tests on v3_compare.csv, whose
  order_ids are PROVABLY disjoint from training.

  NOTE ON FABRICATION: association_audit.py showed the
  model invents weak associations. I initially thought
  this would contaminate the efficacy test, and built
  detection for it. I was WRONG, and I verified I was
  wrong on fixtures: a synthetic dataset with a
  DELIBERATELY planted false hour->target link still
  scored 99.3% efficacy - because on REAL test data the
  fabricated feature carries no signal, so the
  classifier simply ignores it. Fabrication corrupts
  the synthetic data as a DESCRIPTION OF RELATIONSHIPS,
  but it does not fake the efficacy measurement.
  So this test is run straight.

CORRELATION
  v2's 97.6% was inflated. Seven of its thirteen columns
  were deterministic functions of the other six
  (is_night = f(hour), aisle_popularity = f(aisle_id)).
  Deterministic relationships are trivial to preserve,
  so the score was flattered by them.

  v3 measures correlation on the six INDEPENDENT
  training columns only. The number will be lower and
  it will mean something.

PRIVACY  -- this one was genuinely broken
  v2's privacy_evaluation.py did:

      real_train, real_holdout = train_test_split(
          fixed_real_compare, test_size=0.5)

  and called real_train "what the model saw". THE MODEL
  NEVER SAW IT. CTGAN trained on a different sample
  entirely. So the DCR ratio was measuring "is synthetic
  closer to some random real rows than other random real
  rows are" - which is not a memorisation test at all.

  It ALSO loaded the post-processed synthetic file, which
  literally contains real hours copied in - so the test
  was blind to the one thing it should have caught.

  v3 measures distance from synthetic to the ACTUAL
  TRAINING ROWS (v3_train.csv), benchmarked against
  distance from genuinely held-out real rows to those
  same training rows. That is a real memorisation test.

  Verified on fixtures before writing:
      memorising model   -> ratio 0.000  (caught)
      generalising model -> ratio 1.000  (passed)
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
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = 'is_reorder'

# Features a real order actually arrives with.
# order_size_grp excluded - it is a label we impose.
FEATURES = [
    'aisle_id',
    'order_dow',
    'order_hour_of_day',
    'is_early_in_cart',
    'days_since_prior_order',
]

# The six independent trained columns
BASE_COLS = FEATURES + [TARGET]

print("=" * 70)
print("EVALUATE v3 - the honest numbers")
print("=" * 70)

# ══════════════════════════════════════════
# LOAD + PROVE DISJOINTNESS
# ══════════════════════════════════════════
print("\n[1] Loading and proving disjointness ...")

train = pd.read_csv('data/v3_train.csv')
comp = pd.read_csv('data/v3_compare.csv')
synth = pd.read_csv('data/synthetic_v3.csv')
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])

assert set(train['order_id']) <= tr_ids, \
    "FATAL: train file contains non-training orders"
assert len(set(comp['order_id']) & tr_ids) == 0, \
    "FATAL: compare set contains training orders"

print(f"    train:     {len(train):,} rows "
      f"(what the model actually saw)")
print(f"    compare:   {len(comp):,} rows "
      f"(model has NEVER seen these)")
print(f"    synthetic: {len(synth):,} rows")
print("    [ASSERTED] compare ∩ train = 0 orders")


def numeric(d, cols):
    return d[cols].apply(
        pd.to_numeric, errors='coerce').fillna(0)


# ══════════════════════════════════════════
# [2] ML EFFICACY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[2] ML EFFICACY")
print("=" * 70)
print("""
    Train a classifier on synthetic data only.
    Test it on real data the model has never seen.
    Compare against a classifier trained on real data.

    If synthetic data is a viable substitute for real
    customer data, the two should score the same.
""")

assert comp[TARGET].nunique() == 2, \
    "FATAL: target is single-class in compare set"

real_tr, real_te = train_test_split(
    comp, test_size=0.3, random_state=SEED,
    stratify=comp[TARGET])

assert real_te[TARGET].nunique() == 2, \
    "FATAL: test set is single-class - AUC undefined"

print(f"    real train: {len(real_tr):,}")
print(f"    real test:  {len(real_te):,}")
print(f"    synthetic:  {len(synth):,}")


def fit_score(train_df, label):
    clf = RandomForestClassifier(
        n_estimators=100, random_state=SEED,
        n_jobs=-1)
    clf.fit(numeric(train_df, FEATURES),
            train_df[TARGET])
    Xte = numeric(real_te, FEATURES)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]
    m = {
        'acc': accuracy_score(real_te[TARGET], pred),
        'f1':  f1_score(real_te[TARGET], pred),
        'auc': roc_auc_score(real_te[TARGET], prob),
    }
    print(f"\n    trained on {label}")
    print(f"      accuracy {m['acc']:.4f}   "
          f"F1 {m['f1']:.4f}   AUC {m['auc']:.4f}")
    return m


# KNOWN CAVEAT (do not silently "fix" - the saved JSON came
# from this code as written): training sizes are unmatched
# here - real_tr ~35k rows vs synth 50k - which slightly
# favours the synthetic side. test_marginal_impact.py repeats
# this measurement with BOTH sides matched (10k each) and a
# measured split-noise floor; its 89.2% +/- 0.7% is the
# headline figure. Cite that one, and this one only as the
# single-run detail.
m_real = fit_score(real_tr, "REAL data")
m_syn = fit_score(synth, "SYNTHETIC data")

print(f"\n    {'Metric':<10}{'Real':>9}"
      f"{'Synth':>9}{'Ratio':>9}")
print("    " + "-" * 38)
ratios = {}
for k in ['acc', 'f1', 'auc']:
    r = m_syn[k] / m_real[k] if m_real[k] else 0
    ratios[k] = r
    print(f"    {k:<10}{m_real[k]:>9.4f}"
          f"{m_syn[k]:>9.4f}{r:>8.1%}")

efficacy = float(np.mean(list(ratios.values())))
print(f"\n    ML EFFICACY: {efficacy:.1%}")
print(f"\n    (v2 reported 100.3% - but that was")
print(f"     measured on synthetic data with REAL")
print(f"     hours injected into it, tested against")
print(f"     rows the model may have trained on.)")

if efficacy >= 0.95:
    print("\n    STRONG - synthetic data is a viable")
    print("    substitute for real data in ML tasks.")
elif efficacy >= 0.90:
    print("\n    GOOD - most predictive structure")
    print("    preserved.")
else:
    print("\n    WEAKER than v2 claimed. Report honestly.")


# ══════════════════════════════════════════
# [3] CORRELATION - independent columns only
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[3] CORRELATION PRESERVATION")
print("=" * 70)
print("""
    v2 measured this across 13 columns, 7 of which were
    deterministic functions of the other 6. Preserving
    is_night = f(hour) is trivial, so the 97.6% was
    flattered.

    v3 measures ONLY the six independent columns. Lower
    number, real meaning.
""")

R = numeric(comp, BASE_COLS)
S = numeric(synth, BASE_COLS)

cr = R.corr()
cs = S.corr()
mask = ~np.eye(len(BASE_COLS), dtype=bool)
diff = (cr - cs).abs().values[mask]

# A NaN here means a column was degenerate (constant)
# in real or synthetic data. np.nanmean would SILENTLY
# SKIP those pairs - inflating the score by simply not
# counting the ones the model got wrong. That is the
# same class of bug that flattered the v2 numbers.
# So: detect them, report them, do not hide them.
n_nan = int(np.isnan(diff).sum())
if n_nan > 0:
    print(f"    [!] {n_nan} column pairs produced NaN")
    print("        correlation - a column is constant")
    print("        in real or synthetic data. These are")
    print("        counted as FULL disagreement (1.0),")
    print("        NOT silently skipped.")
    for i in range(len(BASE_COLS)):
        for j in range(i + 1, len(BASE_COLS)):
            if np.isnan(cr.iloc[i, j]) or \
               np.isnan(cs.iloc[i, j]):
                print(f"          {BASE_COLS[i]} ~ "
                      f"{BASE_COLS[j]}")
    diff = np.where(np.isnan(diff), 1.0, diff)

corr_score = 1 - diff.mean()

print(f"    correlation similarity: "
      f"{corr_score:.1%}")
print(f"    (v2 claimed 97.6% across 13 columns,")
print(f"     7 of them deterministic)")

print(f"\n    Strongest real relationships:")
print(f"    {'pair':<40}{'real':>8}{'synth':>8}"
      f"{'gap':>7}")
print("    " + "-" * 63)
pairs = []
for i in range(len(BASE_COLS)):
    for j in range(i + 1, len(BASE_COLS)):
        a, b = BASE_COLS[i], BASE_COLS[j]
        rv, sv = cr.iloc[i, j], cs.iloc[i, j]
        pairs.append((abs(rv), a, b, rv, sv))
pairs.sort(reverse=True)
for _, a, b, rv, sv in pairs[:6]:
    print(f"    {a+' ~ '+b:<40}{rv:>8.3f}"
          f"{sv:>8.3f}{abs(rv-sv):>7.3f}")


# ══════════════════════════════════════════
# [4] PRIVACY - a REAL memorisation test
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[4] PRIVACY - did the model memorise?")
print("=" * 70)
print("""
    v2's test was BROKEN. It split fixed_real_compare
    in half and called one half "what the model saw" -
    but the model never saw it. It was not measuring
    memorisation at all.

    v3 measures distance from synthetic rows to the
    ACTUAL TRAINING ROWS, benchmarked against distance
    from genuinely held-out real rows to those same
    training rows.

      ratio ~ 1.0  synthetic is no closer to the
                   training data than fresh real data
                   is -> the model generalised
      ratio << 1.0 synthetic sits unnaturally close to
                   training rows -> memorisation
""")

n = 5000
T = numeric(train.sample(n=n, random_state=SEED),
            BASE_COLS).values
H = numeric(comp.sample(n=n, random_state=SEED),
            BASE_COLS).values
Y = numeric(synth.sample(n=n, random_state=SEED),
            BASE_COLS).values

# normalise so no column dominates the distance
allv = np.vstack([T, H, Y])
lo, hi = allv.min(0), allv.max(0)
rng_ = np.where(hi - lo == 0, 1, hi - lo)
Tn, Hn, Yn = ((T - lo) / rng_,
              (H - lo) / rng_,
              (Y - lo) / rng_)

nn = NearestNeighbors(n_neighbors=1,
                      n_jobs=-1).fit(Tn)
d_syn = nn.kneighbors(Yn)[0].flatten()
d_hold = nn.kneighbors(Hn)[0].flatten()

med_syn = float(np.median(d_syn))
med_hold = float(np.median(d_hold))
dcr_ratio = (med_syn / med_hold
             if med_hold > 0 else 0)
exact = int((d_syn < 1e-9).sum())

print(f"    median distance, synthetic -> training: "
      f"{med_syn:.4f}")
print(f"    median distance, held-out  -> training: "
      f"{med_hold:.4f}")
print(f"\n    DCR RATIO: {dcr_ratio:.3f}")
print(f"    exact matches: {exact} / {n}")

print()
if dcr_ratio >= 0.9:
    print("    PRIVACY PRESERVED. Synthetic rows are no")
    print("    closer to the training data than fresh")
    print("    real data is. No memorisation.")
elif dcr_ratio >= 0.7:
    print("    MOSTLY PRESERVED. Slight closeness to")
    print("    training rows. Report the figure.")
else:
    print("    *** MEMORISATION RISK ***")
    print("    Synthetic rows sit unnaturally close to")
    print("    the training data. This must be reported.")

if exact > 0:
    print(f"\n    NOTE: {exact} exact matches. With only")
    print("    six low-cardinality columns, coincidental")
    print("    matches are expected - the RATIO is the")
    print("    real signal, not the raw count.")


# ══════════════════════════════════════════
# CHART
# ══════════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(
    'Synthetic Data Evaluation (v3 - clean pipeline)',
    fontsize=13, fontweight='bold')

ax[0].bar(['Real', 'Synthetic'],
          [m_real['acc'], m_syn['acc']],
          color=['steelblue', 'seagreen'])
ax[0].set_title(f'ML Efficacy: {efficacy:.1%}\n'
                '(classifier accuracy on real data)')
ax[0].set_ylim(0, 1)
for i, v in enumerate([m_real['acc'],
                       m_syn['acc']]):
    ax[0].text(i, v + 0.02, f'{v:.3f}',
               ha='center', fontweight='bold')

im = ax[1].imshow((cr - cs).abs(),
                  cmap='Reds', vmin=0, vmax=0.3)
ax[1].set_xticks(range(len(BASE_COLS)))
ax[1].set_yticks(range(len(BASE_COLS)))
lbl = [c[:9] for c in BASE_COLS]
ax[1].set_xticklabels(lbl, rotation=45,
                      ha='right', fontsize=7)
ax[1].set_yticklabels(lbl, fontsize=7)
ax[1].set_title(
    f'Correlation gap: {corr_score:.1%} similar\n'
    '(darker = bigger error)')
plt.colorbar(im, ax=ax[1], fraction=0.046)

ax[2].hist(d_hold, bins=40, alpha=0.6,
           color='steelblue', density=True,
           label='held-out real -> train')
ax[2].hist(d_syn, bins=40, alpha=0.6,
           color='seagreen', density=True,
           label='synthetic -> train')
ax[2].set_title(f'Privacy: DCR ratio '
                f'{dcr_ratio:.2f}\n'
                '(overlapping = no memorisation)')
ax[2].set_xlabel('distance to nearest training row')
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/evaluation_v3.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n    saved results/evaluation_v3.png")


# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("THE HONEST NUMBERS")
print("=" * 70)
print(f"""
    Quality (held-out):     0.9205
    ML efficacy:            {efficacy:.1%}
    Correlation:            {corr_score:.1%}
    Privacy (DCR ratio):    {dcr_ratio:.3f}

    Conditional aisle shift: reproduced, but the model
    OVERSTATES it by ~28% (real TVD 0.1296 -> synthetic
    0.1660).

    KNOWN LIMITATION - measured, not guessed:
    CTGAN fabricates weak associations. Where the real
    association between two columns is near zero, the
    model invents one (mean inflation +0.0585). Where it
    is strong, the model is faithful (+0.0304).

    Consequence: the synthetic data must NOT be used to
    study relationships that are weak in reality. Basket
    size and reorder rate are independent in the real
    data (0.0061) but linked in the synthetic (0.0574).

    Every one of these numbers is measured on orders the
    model has never seen, with no post-processing and no
    injected values.
""")

json.dump({
    # Quality is NOT computed in this script. 0.9205 is the
    # all-categorical held-out score with proper provenance in
    # model_comparison_v3.py (its GAP 2 note explains why the
    # train_v3.py figure was Shapes-only). If ctgan_v3.pkl or
    # synthetic_v3.csv is ever regenerated, re-run
    # model_comparison_v3.py and update this value by hand -
    # it does not refresh itself.
    'quality': 0.9205,
    'ml_efficacy': efficacy,
    'correlation': float(corr_score),
    'dcr_ratio': dcr_ratio,
    'exact_matches': exact,
    'efficacy_detail': {
        'real': m_real, 'synth': m_syn,
        'ratios': ratios},
}, open('data/evaluation_v3.json', 'w'),
    indent=2, default=float)
print("Saved data/evaluation_v3.json")
print("=" * 70)