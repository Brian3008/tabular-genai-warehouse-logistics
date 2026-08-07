"""
tabsyn_evaluate.py
==================
Verbatim copy of evaluate_v3.py for the TabSyn bench
(tabsyn_bench_contract.md). The measurement code is
character-identical to the original; every deviation is
declared here.

DECLARED DEVIATIONS (nothing else changed):
 1. I/O head: --synthetic-csv and --out-dir arguments
    replace the hard-coded 'data/synthetic_v3.csv' and the
    hard-coded output paths. ALL outputs go to --out-dir
    (protection rules: results\\tabsyn\\ only; the original
    data/evaluation_v3.json and results/evaluation_v3.png
    are never touched).
 2. QUALITY IS COMPUTED, NOT HARD-CODED. evaluate_v3.py
    hard-codes 'quality': 0.9205 with provenance in
    model_comparison_v3.py. Per the bench contract, the
    quality() function is ported VERBATIM from
    model_comparison_v3.py (all-categorical metadata,
    seeded n<=9000 samples vs v3_eval.csv) and called on
    the scored file. On data/synthetic_v3.csv it must
    reproduce 0.920462963007103 (model_comparison_v3.json)
    or this copy does not graduate. This adds v3_eval.csv
    to the files read.
 3. matplotlib forced to 'Agg' (headless console run; the
    original ran in PyCharm and calls plt.show()).
 4. Narrative prose in prints (v2 history, CTGAN findings)
    is kept verbatim from the original - it describes the
    CTGAN audit story, not the file being scored.

READS:  data/v3_train.csv, data/v3_compare.csv,
        data/v3_eval.csv, data/v3_train_order_ids.csv,
        <--synthetic-csv>
WRITES: <--out-dir>/evaluation.json,
        <--out-dir>/evaluation.png

Known-answer gate (contract rule 1): run with
  --synthetic-csv data/synthetic_v3.csv
and reproduce evaluation_v3.json (ml_efficacy 0.878125,
correlation 0.970223, dcr_ratio 1.124376, exact 154) and
quality 0.920463 before scoring TabSyn.
"""

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument('--synthetic-csv', required=True,
                 help='synthetic CSV to score')
_ap.add_argument('--out-dir', required=True,
                 help='directory for ALL outputs '
                      '(must be under results/tabsyn)')
_args = _ap.parse_args()
SYNTH_CSV = _args.synthetic_csv
OUT_DIR = _args.out_dir

import sys
# Some prints use non-ASCII (e.g. the intersection symbol).
# A default Windows console is cp1252 and CRASHES on them;
# force UTF-8 so the script runs outside PyCharm too.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')  # declared deviation 3

import pandas as pd
import numpy as np
import random
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import os
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
print("EVALUATE (tabsyn bench copy of evaluate_v3.py)")
print(f"scored file: {SYNTH_CSV}")
print("=" * 70)

# ══════════════════════════════════════════
# LOAD + PROVE DISJOINTNESS
# ══════════════════════════════════════════
print("\n[1] Loading and proving disjointness ...")

train = pd.read_csv('data/v3_train.csv')
comp = pd.read_csv('data/v3_compare.csv')
evald = pd.read_csv('data/v3_eval.csv')   # for quality (deviation 2)
synth = pd.read_csv(SYNTH_CSV)
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])

assert set(train['order_id']) <= tr_ids, \
    "FATAL: train file contains non-training orders"
assert len(set(comp['order_id']) & tr_ids) == 0, \
    "FATAL: compare set contains training orders"
assert len(set(evald['order_id']) & tr_ids) == 0, \
    "FATAL: eval set contains training orders"

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

if efficacy >= 0.95:
    print("\n    STRONG - synthetic data is a viable")
    print("    substitute for real data in ML tasks.")
elif efficacy >= 0.90:
    print("\n    GOOD - most predictive structure")
    print("    preserved.")
else:
    print("\n    WEAKER - report honestly.")


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
# [5] QUALITY - ported VERBATIM from
#     model_comparison_v3.py (deviation 2)
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[5] QUALITY (ported from model_comparison_v3.py)")
print("=" * 70)

# All columns categorical - declaring the continuous one
# as 'numerical' makes SDMetrics return nan for pair
# trends, silently dropping half the score.
qmeta = SingleTableMetadata()
qmeta.detect_from_dataframe(evald[BASE_COLS])
for c in BASE_COLS:
    qmeta.update_column(column_name=c,
                        sdtype='categorical')


def quality(synth):
    """Returns (overall, column_shapes).

    KNOWN CAVEAT - population mismatch that favours the
    shuffle baseline: CTGAN trained on small+large baskets
    only (mid dropped), so its output contains no mid-order
    rows, while evald includes all sizes. The shuffle is
    built from the full training set, so it covers the eval
    population better BY CONSTRUCTION. Part of its quality
    edge over CTGAN is this artifact, not GAN error. The
    bias understates CTGAN (honest direction) - state it
    when telling the quality-vs-efficacy story.

    SDMetrics' overall score is the AVERAGE of Column
    Shapes and Column Pair Trends.

    The shuffling baseline preserves every marginal
    EXACTLY (Shapes ~100%) but destroys every
    relationship (Pair Trends collapses). So its OVERALL
    score lands mid-range - which means testing the
    overall score tells you NOTHING about whether the
    shuffle worked.

    The sanity check needs the SHAPES component
    specifically. That is the one that must be ~100%
    for a column shuffle.
    """
    n = min(len(evald), len(synth), 9000)
    q = evaluate_quality(
        real_data=evald[BASE_COLS].sample(
            n=n, random_state=SEED),
        synthetic_data=synth[BASE_COLS].sample(
            n=n, random_state=SEED),
        metadata=qmeta, verbose=False)
    overall = float(q.get_score())

    # Read the Column Shapes component. The sanity check
    # DEPENDS on this, so it must not silently become
    # nan if the SDV API differs across versions.
    shapes = float('nan')
    try:
        props = q.get_properties()
        for pc in ['Property', 'property']:
            if pc in props.columns:
                row = props[props[pc].astype(str)
                            .str.contains('Shape',
                                          case=False)]
                if len(row):
                    sc = [c for c in props.columns
                          if c.lower() == 'score'][0]
                    shapes = float(row[sc].iloc[0])
                    break
    except Exception:
        pass

    # FALLBACK: compute Column Shapes ourselves if the
    # API path failed. Shapes = mean TV-complement across
    # columns - i.e. how well each marginal matches.
    if np.isnan(shapes):
        rs = evald[BASE_COLS].sample(
            n=n, random_state=SEED)
        ss = synth[BASE_COLS].sample(
            n=n, random_state=SEED)
        scores = []
        for c in BASE_COLS:
            a = rs[c].value_counts(normalize=True)
            b = ss[c].value_counts(normalize=True)
            idx = sorted(set(a.index) | set(b.index))
            a = a.reindex(idx, fill_value=0)
            b = b.reindex(idx, fill_value=0)
            tv = 0.5 * np.abs(a - b).sum()
            scores.append(1 - tv)
        shapes = float(np.mean(scores))
        print("      (Column Shapes computed directly -"
              " SDV API path unavailable)")

    return overall, shapes


q_overall, q_shapes = quality(synth)
print(f"    quality (overall): {q_overall:.4f}")
print(f"    column shapes:     {q_shapes:.4f}")
print("    (known-answer on data/synthetic_v3.csv:")
print("     0.920462963007103 per model_comparison_v3.json)")


# ══════════════════════════════════════════
# CHART
# ══════════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(
    'Synthetic Data Evaluation (tabsyn bench copy)',
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
plt.savefig(os.path.join(OUT_DIR, 'evaluation.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print(f"\n    saved {os.path.join(OUT_DIR, 'evaluation.png')}")


# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("THE HONEST NUMBERS")
print("=" * 70)
print(f"""
    Quality (held-out):     {q_overall:.4f}
    ML efficacy:            {efficacy:.1%}
    Correlation:            {corr_score:.1%}
    Privacy (DCR ratio):    {dcr_ratio:.3f}

    Every one of these numbers is measured on orders the
    model has never seen, with no post-processing and no
    injected values.
""")

json.dump({
    # Deviation 2: quality COMPUTED here via the function
    # ported verbatim from model_comparison_v3.py, not
    # hard-coded as in evaluate_v3.py.
    'scored_file': SYNTH_CSV,
    'quality': q_overall,
    'quality_shapes': q_shapes,
    'ml_efficacy': efficacy,
    'correlation': float(corr_score),
    'dcr_ratio': dcr_ratio,
    'exact_matches': exact,
    'efficacy_detail': {
        'real': m_real, 'synth': m_syn,
        'ratios': ratios},
}, open(os.path.join(OUT_DIR, 'evaluation.json'), 'w'),
    indent=2, default=float)
print(f"Saved {os.path.join(OUT_DIR, 'evaluation.json')}")
print("=" * 70)
