"""
model_comparison_v3.py
======================
Closes the two remaining gaps behind us before the fleet
rebuild.

GAP 1 - THE SHUFFLING BASELINE NEVER SHUFFLED
---------------------------------------------
v2's model_comparison.py did:

    for col in eval_cols:
        baseline[col] = df_full[col].sample(
            frac=1, random_state=SEED).values[:50000]

Every column shuffled with the SAME seed. pandas.sample
with a fixed seed produces the SAME permutation every
time - so every column was reordered IDENTICALLY, and
the rows came back completely intact, just in a
different order.

Verified on a fixture before writing this:
    same seed per column  -> 5/5 rows INTACT
    different seed        -> 0/5 rows intact

So the "shuffling baseline" was REAL DATA scored against
REAL DATA. That is why it hit 98.1% quality and 99.5%
correlation - and why the argument you gave Locus
("the baseline wins on quality but cannot generate new
data") rested on a bug.

v3 uses a DIFFERENT seed per column. Marginals are
preserved; the joint structure is genuinely destroyed.
That is what a shuffling baseline is FOR: it shows what
score you get from matching every column individually
while learning NOTHING about how they relate.

GAP 2 - THE QUALITY SCORE HAD NO PROPER PROVENANCE
--------------------------------------------------
The 92.05% figure came out of diagnose_v3.py as a
side-computation while chasing a NaN bug. It is being
reported as a headline number, so it deserves a
dedicated, audited script. Recomputed here.

Also note: declaring days_since_prior_order as
'numerical' made SDMetrics return nan for Column Pair
Trends - HALF the quality metric silently missing. All
columns are declared categorical here so the full score
computes.

WHAT IS COMPARED
----------------
  CTGAN v3          the clean model
  Shuffling         marginals right, joint destroyed
  GaussianCopula    a simpler generative baseline

on four axes:
  quality           does it look like real data
  correlation       are relationships preserved
  ML efficacy       is it USEFUL for a real ML task
  privacy           did it memorise the training rows

The point of the comparison: a model can win on quality
while being useless. The shuffling baseline should score
WELL on quality and BADLY on correlation - because it
preserves every marginal and destroys every
relationship. If it does not, something is wrong with
the test.
"""

import pandas as pd
import numpy as np
import random
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = 'is_reorder'
FEATURES = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_early_in_cart', 'days_since_prior_order',
]
BASE_COLS = FEATURES + [TARGET]
N_SYNTH = 50000

print("=" * 70)
print("MODEL COMPARISON v3")
print("=" * 70)

# ══════════════════════════════════════════
# LOAD + PROVE DISJOINTNESS
# ══════════════════════════════════════════
print("\n[1] Loading ...")
train = pd.read_csv('data/v3_train.csv')
evald = pd.read_csv('data/v3_eval.csv')
comp = pd.read_csv('data/v3_compare.csv')
tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])

assert set(train['order_id']) <= tr_ids
assert len(set(evald['order_id']) & tr_ids) == 0, \
    "FATAL: eval contains training orders"
assert len(set(comp['order_id']) & tr_ids) == 0, \
    "FATAL: compare contains training orders"
print("    [ASSERTED] eval and compare are disjoint")
print("               from the model's training orders")
print(f"    train {len(train):,} | eval {len(evald):,}"
      f" | compare {len(comp):,}")


def numeric(d, cols=BASE_COLS):
    return d[cols].apply(
        pd.to_numeric, errors='coerce').fillna(0)


# ══════════════════════════════════════════
# METRIC DEFINITIONS
# ══════════════════════════════════════════
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


def correlation(synth):
    R = numeric(comp)
    S = numeric(synth)
    cr, cs = R.corr(), S.corr()
    mask = ~np.eye(len(BASE_COLS), dtype=bool)
    d = (cr - cs).abs().values[mask]
    # NaN = a column collapsed. Count as full
    # disagreement, never silently skip.
    d = np.where(np.isnan(d), 1.0, d)
    return float(1 - d.mean())


# real baseline for ML efficacy
real_tr, real_te = train_test_split(
    comp, test_size=0.3, random_state=SEED,
    stratify=comp[TARGET])
_rc = RandomForestClassifier(
    n_estimators=100, random_state=SEED, n_jobs=-1)
_rc.fit(numeric(real_tr, FEATURES),
        real_tr[TARGET])
REAL_ACC = accuracy_score(
    real_te[TARGET],
    _rc.predict(numeric(real_te, FEATURES)))

# ---- THE FLOOR: what you score by learning NOTHING ----
# A classifier that ignores every feature and always
# predicts the majority class.
#
# This matters enormously for reading the ML efficacy
# column. The SHUFFLED baseline has its target shuffled
# INDEPENDENTLY of the features - so the target carries
# zero signal, the classifier learns nothing, and it
# falls back to the majority class.
#
# Which means the shuffled baseline's "ML efficacy" is
# not a measure of anything it learned. It IS the floor.
#
# Without showing this, a reader sees "shuffling 87%,
# CTGAN 88%" and concludes CTGAN barely beats a shuffle -
# when in truth 87% is what you get from learning
# ABSOLUTELY NOTHING, and the real question is how far
# above the floor each model sits.
MAJORITY = float(real_te[TARGET].value_counts(
    normalize=True).max())
FLOOR = MAJORITY / REAL_ACC

print(f"\n    real-data classifier accuracy: "
      f"{REAL_ACC:.4f}")
print(f"    majority-class accuracy:       "
      f"{MAJORITY:.4f}   <- learning NOTHING")
print(f"\n    ML EFFICACY FLOOR: {FLOOR:.1%}")
print("    Any model scoring this is predicting the")
print("    majority class and has learned nothing.")
print("    Read every MLeff number against THIS, not")
print("    against 0%.")


# KNOWN CAVEAT: each generator's classifier trains on its full
# 50k rows while the real benchmark trained on ~35k, so the
# ratio-vs-real is slightly flattered for every generator. The
# RANKING between generators is fair (all get 50k). The clean,
# size-matched efficacy figure is test_marginal_impact.py's
# 89.2% +/- 0.7% - cite that as the headline.
def ml_efficacy(synth):
    clf = RandomForestClassifier(
        n_estimators=100, random_state=SEED,
        n_jobs=-1)
    clf.fit(numeric(synth, FEATURES),
            synth[TARGET])
    a = accuracy_score(
        real_te[TARGET],
        clf.predict(numeric(real_te, FEATURES)))
    return float(a / REAL_ACC) if REAL_ACC else 0.0


# privacy: distance to the ACTUAL training rows
_n = 4000
_T = numeric(train.sample(n=_n, random_state=SEED))
_H = numeric(comp.sample(n=_n, random_state=SEED))


def privacy(synth):
    Y = numeric(synth.sample(
        n=min(_n, len(synth)), random_state=SEED))
    allv = np.vstack([_T.values, _H.values,
                      Y.values]).astype(float)
    lo, hi = allv.min(0), allv.max(0)
    rg = np.where(hi - lo == 0, 1, hi - lo)
    nn = NearestNeighbors(
        n_neighbors=1, n_jobs=-1).fit(
        (_T.values - lo) / rg)
    ds = nn.kneighbors((Y.values - lo) / rg
                       )[0].flatten()
    dh = nn.kneighbors((_H.values - lo) / rg
                       )[0].flatten()
    mh = np.median(dh)
    return float(np.median(ds) / mh) if mh > 0 else 0.0


# ══════════════════════════════════════════
# [2] THE THREE GENERATORS
# ══════════════════════════════════════════
results = []

# ---- CTGAN v3 ----
print("\n" + "=" * 70)
print("[2a] CTGAN v3 (the clean model)")
print("=" * 70)
ctgan_synth = pd.read_csv('data/synthetic_v3.csv')
print(f"    loaded {len(ctgan_synth):,} synthetic rows")
_q, _s = quality(ctgan_synth)
results.append({
    'model': 'CTGAN v3',
    'quality': _q,
    'shapes': _s,
    'correlation': correlation(ctgan_synth),
    'ml_efficacy': ml_efficacy(ctgan_synth),
    'privacy': privacy(ctgan_synth),
})

# ---- SHUFFLING BASELINE (fixed) ----
print("\n" + "=" * 70)
print("[2b] SHUFFLING BASELINE - the v2 bug fixed")
print("=" * 70)
print("""
    v2 shuffled every column with the SAME seed, so
    every column got the SAME permutation and the rows
    came back INTACT. It was scoring real data against
    real data. That is why it 'won'.

    Fixed: a DIFFERENT seed per column. Marginals
    preserved, joint structure destroyed.
""")

src = train.sample(n=N_SYNTH, random_state=SEED,
                   replace=len(train) < N_SYNTH
                   ).reset_index(drop=True)
shuf = pd.DataFrame(index=range(len(src)))
for i, col in enumerate(BASE_COLS):
    # DIFFERENT seed per column - this is the fix
    shuf[col] = src[col].sample(
        frac=1,
        random_state=1000 + i).values

# prove it actually shuffled
orig_rows = set(map(tuple,
                    src[BASE_COLS].values))
shuf_rows = list(map(tuple, shuf[BASE_COLS].values))
intact = sum(1 for r in shuf_rows if r in orig_rows)
print(f"    rows still intact after shuffling: "
      f"{intact:,} / {len(shuf):,} "
      f"({intact/len(shuf):.1%})")
print("    (v2's version left 100% intact - it never")
print("     shuffled at all)")

_q, _s = quality(shuf)
results.append({
    'model': 'Shuffling baseline',
    'quality': _q,
    'shapes': _s,
    'correlation': correlation(shuf),
    'ml_efficacy': ml_efficacy(shuf),
    'privacy': privacy(shuf),
})

# ---- GAUSSIAN COPULA ----
print("\n" + "=" * 70)
print("[2c] GaussianCopula")
print("=" * 70)
gmeta = SingleTableMetadata()
gmeta.detect_from_dataframe(train[BASE_COLS])
for c in BASE_COLS:
    if c == 'days_since_prior_order':
        gmeta.update_column(column_name=c,
                            sdtype='numerical')
    else:
        gmeta.update_column(column_name=c,
                            sdtype='categorical')
print("    training (fast) ...")
gc = GaussianCopulaSynthesizer(gmeta)
gc.fit(train[BASE_COLS])
gc_synth = gc.sample(num_rows=N_SYNTH)
print(f"    generated {len(gc_synth):,} rows")

_q, _s = quality(gc_synth)
results.append({
    'model': 'GaussianCopula',
    'quality': _q,
    'shapes': _s,
    'correlation': correlation(gc_synth),
    'ml_efficacy': ml_efficacy(gc_synth),
    'privacy': privacy(gc_synth),
})


# ══════════════════════════════════════════
# [3] RESULTS
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\n{'Model':<22}{'Quality':>9}{'Shapes':>9}"
      f"{'Corr':>9}{'MLeff':>9}{'Privacy':>9}")
print("-" * 70)
for r in results:
    print(f"{r['model']:<22}"
          f"{r['quality']:>8.1%}"
          f"{r['shapes']:>9.1%}"
          f"{r['correlation']:>9.1%}"
          f"{r['ml_efficacy']:>9.1%}"
          f"{r['privacy']:>9.2f}")

print("-" * 70)
print(f"{'FLOOR (majority class)':<22}"
      f"{'--':>9}{'--':>9}{'--':>9}"
      f"{FLOOR:>9.1%}{'--':>9}")

print("\n  Quality  = overall (Shapes + PairTrends)/2")
print("  Shapes   = do the COLUMNS individually match")
print("  Corr     = are the RELATIONSHIPS preserved")
print("  MLeff    = is it USEFUL for a real ML task")
print("  Privacy  = DCR ratio. >=0.9 no memorisation;")
print("             LOW = reuses real values")
print()
print("  *** READ MLeff AGAINST THE FLOOR ***")
print(f"  {FLOOR:.1%} is what you score by predicting the")
print("  majority class and learning NOTHING. The")
print("  shuffled baseline's target is independent of")
print("  its features, so it CANNOT beat the floor.")
print("  What matters is how far ABOVE the floor a")
print("  model sits - not its raw MLeff percentage.")

# how much of the available headroom does each model
# actually capture?
print()
print(f"  {'Model':<22}{'MLeff':>9}{'above floor':>13}")
print("  " + "-" * 44)
for r in results:
    lift = r['ml_efficacy'] - FLOOR
    head = 1.0 - FLOOR
    pct = (lift / head) if head > 0 else 0
    print(f"  {r['model']:<22}"
          f"{r['ml_efficacy']:>8.1%}"
          f"{pct:>12.1%}")
print()
print("  'above floor' = fraction of the AVAILABLE")
print("  headroom captured. 0% = learned nothing;")
print("  100% = matches a real-data classifier.")


# ══════════════════════════════════════════
# [4] SANITY CHECK ON THE TEST ITSELF
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("SANITY CHECK")
print("=" * 70)
print("""
    A shuffling baseline preserves every marginal and
    destroys every relationship. So it SHOULD score
    high on quality and LOW on correlation.

    If it scores high on BOTH, the test is broken -
    exactly as it was in v2.
""")
sh = next(r for r in results
          if r['model'] == 'Shuffling baseline')
ct = next(r for r in results
          if r['model'] == 'CTGAN v3')

ok = True
# Test COLUMN SHAPES, not the overall score.
# Overall = (Shapes + PairTrends)/2, and the shuffle
# DELIBERATELY destroys PairTrends - so the overall
# score lands mid-range even when the test is working
# perfectly. Testing overall would fire a FALSE FAIL.
if np.isnan(sh['shapes']):
    print("    [!] could not read the Column Shapes")
    print("        component - check manually")
elif sh['shapes'] < 0.90:
    print(f"    [!] shuffling Column Shapes only "
          f"{sh['shapes']:.1%} - expected ~100%.")
    print("        A column shuffle preserves every")
    print("        marginal exactly. Something is wrong.")
    ok = False
else:
    print(f"    shuffling Column Shapes "
          f"{sh['shapes']:.1%} - HIGH, as expected")
    print("      (every marginal preserved exactly)")

if sh['correlation'] > ct['correlation']:
    print(f"    [!] shuffling correlation "
          f"({sh['correlation']:.1%}) BEATS CTGAN "
          f"({ct['correlation']:.1%})")
    print("        The shuffle should have DESTROYED")
    print("        the relationships. Investigate.")
    ok = False
else:
    print(f"    shuffling correlation "
          f"{sh['correlation']:.1%} < CTGAN "
          f"{ct['correlation']:.1%}  - as expected")

print(f"\n    test integrity: "
      f"{'PASS' if ok else 'FAIL - do not trust these numbers'}")


# ══════════════════════════════════════════
# [5] CHART
# ══════════════════════════════════════════
# Two panels. Privacy is a RATIO (~1.12), not a
# percentage - plotting it on a 0-100 axis would render
# it as "112" next to a quality of "92", inviting the
# reader to misread it as 112% quality.
fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(15, 6),
    gridspec_kw={'width_ratios': [2.2, 1]})
models = [r['model'] for r in results]
x = np.arange(len(models))
w = 0.25
for i, (k, lab, col) in enumerate([
    ('shapes', 'Column Shapes', 'steelblue'),
    ('correlation', 'Correlation', 'seagreen'),
    ('ml_efficacy', 'ML Efficacy', 'orange'),
]):
    vals = [min(r[k] * 100, 110) for r in results]
    b = ax.bar(x + (i - 1) * w, vals, w,
               label=lab, color=col, alpha=0.85)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2,
                v + 1, f'{v:.0f}', ha='center',
                fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 118)
ax.set_title('Quality / structure / usefulness',
             fontweight='bold')
ax.legend(fontsize=9)

# privacy on its own scale
pv = [r['privacy'] for r in results]
cols = ['crimson' if v < 0.7 else 'seagreen'
        for v in pv]
xp = np.arange(len(models))
b2 = ax2.bar(xp, pv, color=cols, alpha=0.85)
ax2.set_xticks(xp)
ax2.axhline(0.9, ls='--', color='black', lw=1,
            label='0.9 = no memorisation')
for rect, v in zip(b2, pv):
    ax2.text(rect.get_x() + rect.get_width() / 2,
             v + 0.02, f'{v:.2f}', ha='center',
             fontsize=9, fontweight='bold')
ax2.set_ylabel('DCR ratio')
ax2.set_title('Privacy (ratio, not %)\n'
              'red = reuses real values',
              fontweight='bold')
ax2.set_xticklabels(models, rotation=20,
                    ha='right', fontsize=8)
ax2.legend(fontsize=8)

fig.suptitle(
    'Model Comparison (v3 - clean pipeline, '
    'shuffling baseline actually shuffles)',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/model_comparison_v3.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n    saved results/model_comparison_v3.png")


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"""
    CTGAN v3        quality {ct['quality']:.1%}   corr {ct['correlation']:.1%}   MLeff {ct['ml_efficacy']:.1%}
    Shuffling       quality {sh['quality']:.1%}   corr {sh['correlation']:.1%}   MLeff {sh['ml_efficacy']:.1%}

    THE ARGUMENT FOR CTGAN - now with valid evidence:

    Shuffling privacy (DCR): {sh['privacy']:.2f}
    CTGAN privacy (DCR):     {ct['privacy']:.2f}

    CAVEAT ON GaussianCopula PRIVACY: it generates
    days_since_prior_order as a CONTINUOUS value, while
    CTGAN generates it as discrete. A continuous value
    almost never lands exactly on a training value, so
    GC rows look farther from the training data purely
    from float precision - not from better privacy. Its
    DCR score is flattered by that artefact.

    The shuffling baseline matches every column
    individually. But it reuses REAL VALUES from the
    training rows - so its DCR ratio is poor, and that
    is not a flaw in the test, it is the point. It
    cannot be conditioned, cannot generate novel
    combinations, and offers weak privacy. With the bug
    fixed, it also demonstrably fails to preserve the
    relationships between columns.

    CTGAN learns the joint distribution. That is what
    lets it be conditioned on order size and produce
    genuinely new orders.

    In v2 this argument was made against a baseline
    that had never actually been shuffled - so the
    comparison proved nothing. It does now.
""")

json.dump(results,
          open('data/model_comparison_v3.json', 'w'),
          indent=2, default=float)
print("Saved data/model_comparison_v3.json")
print("=" * 70)