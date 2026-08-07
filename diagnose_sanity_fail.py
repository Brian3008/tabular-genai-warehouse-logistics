"""
diagnose_sanity_fail.py
=======================
The sanity check in model_comparison_v3.py FAILED:

    shuffling correlation (97.8%) BEATS CTGAN (97.0%)

My check assumed a shuffle must SCORE WORSE on
correlation, because a shuffle destroys relationships.
It did not. So either:

  (A) the shuffle is not working, or
  (B) my CHECK is wrong.

MY HYPOTHESIS (which must be TESTED, not asserted)
--------------------------------------------------
The six columns are almost entirely UNCORRELATED in the
real data. From evaluate_v3.py, the strongest real pair
was is_early_in_cart ~ is_reorder at just 0.105.
Everything else sat below 0.05.

If that is right:

  - a SHUFFLE destroys correlations that were ALREADY
    near zero. It moves ~0.03 to ~0.00. Error: ~0.03.

  - CTGAN FABRICATES correlations (association_audit.py
    measured this). It pushed is_early_in_cart ~
    is_reorder from 0.105 to 0.219. Error: 0.114.

So the shuffle would "win" on correlation NOT because
the test is broken, but because CTGAN's fabrication
costs it MORE than the shuffle's destruction costs the
shuffle.

That would mean my sanity check is built on a false
premise - it assumed the real data HAS strong
correlations to destroy. It does not.

WHAT THIS SCRIPT DOES
---------------------
1. Prints every real correlation, so we can see whether
   they are genuinely near-zero.

2. Confirms the shuffle actually destroyed what little
   correlation there was (shuffled correlations should
   be ~0.00 across the board).

3. Compares per-pair errors so we can see EXACTLY where
   CTGAN loses ground.

4. Checks the 18.9% intact rows - is that coincidence,
   or did the shuffle fail?

If the hypothesis holds, the FIX is to the CHECK, not
the test - and the correlation metric should be
reported differently, because "97% correlation
similarity" is close to meaningless when the real
correlations are all ~0.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

TARGET = 'is_reorder'
FEATURES = [
    'aisle_id', 'order_dow', 'order_hour_of_day',
    'is_early_in_cart', 'days_since_prior_order',
]
BASE_COLS = FEATURES + [TARGET]
N_SYNTH = 50000

print("=" * 70)
print("WHY DID THE SANITY CHECK FAIL?")
print("=" * 70)

train = pd.read_csv('data/v3_train.csv')
comp = pd.read_csv('data/v3_compare.csv')
ctgan = pd.read_csv('data/synthetic_v3.csv')

tr_ids = set(pd.read_csv(
    'data/v3_train_order_ids.csv')['order_id'])
assert set(train['order_id']) <= tr_ids
assert len(set(comp['order_id']) & tr_ids) == 0, \
    "FATAL: compare contains training orders"
print("\n  [ASSERTED] compare is disjoint from train")


def numeric(d):
    return d[BASE_COLS].apply(
        pd.to_numeric, errors='coerce').fillna(0)


# rebuild the shuffle exactly as the comparison did
src = train.sample(n=N_SYNTH, random_state=SEED
                   ).reset_index(drop=True)
shuf = pd.DataFrame(index=range(len(src)))
for i, col in enumerate(BASE_COLS):
    shuf[col] = src[col].sample(
        frac=1, random_state=1000 + i).values

R = numeric(comp)
C = numeric(ctgan)
S = numeric(shuf)

cr, cc, cs = R.corr(), C.corr(), S.corr()


# ══════════════════════════════════════════
# [1] ARE THE REAL CORRELATIONS NEAR ZERO?
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[1] HOW STRONG ARE THE REAL CORRELATIONS?")
print("=" * 70)
print("""
    If they are all near zero, then a shuffle has almost
    NOTHING to destroy - and my sanity check was built
    on a false premise.
""")

print(f"    {'pair':<44}{'real':>8}")
print("    " + "-" * 52)
pairs = []
for i in range(len(BASE_COLS)):
    for j in range(i + 1, len(BASE_COLS)):
        a, b = BASE_COLS[i], BASE_COLS[j]
        rv = cr.iloc[i, j]
        pairs.append((abs(rv), i, j, a, b, rv))
pairs.sort(reverse=True)
for _, i, j, a, b, rv in pairs:
    print(f"    {a + ' ~ ' + b:<44}{rv:>8.3f}")

strongest = pairs[0][0]
mean_abs = np.mean([p[0] for p in pairs])
print(f"\n    strongest |correlation|: {strongest:.3f}")
print(f"    mean      |correlation|: {mean_abs:.3f}")

if strongest < 0.15:
    print("\n    => THE REAL DATA IS ESSENTIALLY")
    print("       UNCORRELATED. A shuffle has almost")
    print("       nothing to destroy. My sanity check")
    print("       assumed otherwise. The CHECK is wrong.")
else:
    print("\n    => Real correlations are substantial.")
    print("       A shuffle SHOULD have destroyed them.")
    print("       If it did not, the shuffle is broken.")


# ══════════════════════════════════════════
# [2] DID THE SHUFFLE ACTUALLY DESTROY THEM?
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[2] DID THE SHUFFLE DESTROY THE CORRELATIONS?")
print("=" * 70)
print("""
    A correctly shuffled dataset should have ALL
    correlations at ~0.00, because the columns are now
    independent by construction.
""")

print(f"    {'pair':<40}{'real':>7}{'shuf':>7}"
      f"{'ctgan':>7}")
print("    " + "-" * 61)
for _, i, j, a, b, rv in pairs:
    sv = cs.iloc[i, j]
    cv = cc.iloc[i, j]
    print(f"    {a + ' ~ ' + b:<40}{rv:>7.3f}"
          f"{sv:>7.3f}{cv:>7.3f}")

shuf_max = max(abs(cs.iloc[i, j])
               for _, i, j, _, _, _ in pairs)

# Do NOT hard-code a threshold. A shuffle of n rows
# leaves RANDOM residual correlation of order 1/sqrt(n).
# Guessing "0.03" risks a FALSE ALARM - declaring the
# shuffle broken when it is merely noisy. So MEASURE the
# noise floor: shuffle independent random data the same
# way, and see what residual correlation that produces.
rng = np.random.RandomState(SEED)
noise = []
for rep in range(20):
    fake = pd.DataFrame(
        {c: rng.permutation(S[c].values)
         for c in BASE_COLS})
    fc = fake.corr()
    noise.append(max(
        abs(fc.iloc[i, j])
        for _, i, j, _, _, _ in pairs))
noise_p95 = float(np.percentile(noise, 95))

print(f"\n    largest |correlation| in the shuffle: "
      f"{shuf_max:.4f}")
print(f"    empirical noise floor (95th pct of 20")
print(f"    independent shuffles):               "
      f"{noise_p95:.4f}")
print("    (a shuffle of n rows leaves residual")
print("     correlation of order 1/sqrt(n) - this is")
print("     measured, not guessed)")

if shuf_max <= noise_p95 * 1.5:
    print("\n    => THE SHUFFLE WORKED. Residual")
    print("       correlation is within the noise")
    print("       expected from random permutation.")
    shuffle_ok = True
else:
    print("\n    => Residual correlation EXCEEDS the")
    print("       noise floor. Some structure survived.")
    print("       Investigate.")
    shuffle_ok = False


# ══════════════════════════════════════════
# [3] WHERE DOES CTGAN LOSE?
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[3] PER-PAIR ERROR: who is closer to real?")
print("=" * 70)

print(f"\n    {'pair':<36}{'shufErr':>9}"
      f"{'ctganErr':>10}{'winner':>10}")
print("    " + "-" * 65)
shuf_errs, ctgan_errs = [], []
for _, i, j, a, b, rv in pairs:
    se = abs(rv - cs.iloc[i, j])
    ce = abs(rv - cc.iloc[i, j])
    shuf_errs.append(se)
    ctgan_errs.append(ce)
    w = "shuffle" if se < ce else "CTGAN"
    print(f"    {a + ' ~ ' + b:<36}{se:>9.3f}"
          f"{ce:>10.3f}{w:>10}")

print(f"\n    mean error  shuffle: "
      f"{np.mean(shuf_errs):.4f}")
print(f"    mean error  CTGAN:   "
      f"{np.mean(ctgan_errs):.4f}")

worst = max(zip(ctgan_errs, pairs),
            key=lambda t: t[0])
print(f"\n    CTGAN's worst pair: "
      f"{worst[1][3]} ~ {worst[1][4]}")
print(f"      real {worst[1][5]:.3f}  ->  "
      f"ctgan {cc.iloc[worst[1][1], worst[1][2]]:.3f}"
      f"   (error {worst[0]:.3f})")
print("      This is the FABRICATION we measured in")
print("      association_audit.py showing up here.")


# ══════════════════════════════════════════
# [4] THE 18.9% INTACT ROWS
# ══════════════════════════════════════════
print("\n" + "=" * 70)
print("[4] WHY WERE 18.9% OF ROWS STILL INTACT?")
print("=" * 70)

orig = set(map(tuple, src[BASE_COLS].values))
intact = sum(1 for r in map(tuple, shuf[BASE_COLS].values)
             if r in orig)
print(f"\n    intact: {intact:,} / {len(shuf):,} "
      f"({intact/len(shuf):.1%})")

# how many DISTINCT rows exist in the real data?
n_distinct = len(orig)
print(f"    distinct real rows in the sample: "
      f"{n_distinct:,} / {len(src):,}")
print(f"    duplication rate: "
      f"{1 - n_distinct/len(src):.1%}")

print("""
    A shuffled row counts as "intact" if the SAME
    combination happens to exist somewhere in the real
    data - not if it is the SAME ROW.

    With only six low-cardinality columns and 50,000
    rows, many combinations repeat. So a shuffle
    reassembles real-looking combinations by pure
    coincidence.

    This is NOT the v2 bug. In v2, EVERY row was intact
    because every column got the identical permutation.
    Here, columns are independently permuted and the
    correlations are destroyed - the coincidental
    matches are a consequence of low cardinality, not of
    a failed shuffle.
""")


# ══════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════
print("=" * 70)
print("VERDICT")
print("=" * 70)

check_wrong = (strongest < 0.15 and shuffle_ok)

if check_wrong:
    print(f"""
    MY SANITY CHECK WAS WRONG. The test is fine.

    The real data is essentially UNCORRELATED - the
    strongest pair is only {strongest:.3f}. So:

      - the shuffle destroys correlations that were
        already ~0. Its error is tiny by default.
      - CTGAN FABRICATES correlations (measured
        independently in association_audit.py). Its
        error is LARGER.

    So the shuffle "wins" on correlation for a real
    reason, not a broken test. My check assumed the data
    had strong correlations to destroy. It does not.

    *** AND THIS EXPOSES A DEEPER PROBLEM ***

    "97% correlation similarity" is close to MEANINGLESS
    on this data. When every real correlation is ~0.03,
    a model that outputs ZERO correlation everywhere
    scores ~97%. The metric rewards doing nothing.

    That applies to your v2 number too (97.6%) and to
    the v3 number (97.0%). Neither means what it sounds
    like.

    HONEST REPORTING:
      - do NOT lead with correlation similarity
      - report the STRONGEST real correlation
        ({strongest:.3f}) and how well it is reproduced
      - the ML EFFICACY floor comparison is the metric
        that actually discriminates: CTGAN +12.9% above
        floor, shuffle -59%, GaussianCopula -53%
""")
else:
    print("""
    THE CHECK MAY BE RIGHT - the shuffle did not behave
    as a shuffle should. Investigate the shuffle itself
    before trusting any of these numbers.
""")
print("=" * 70)