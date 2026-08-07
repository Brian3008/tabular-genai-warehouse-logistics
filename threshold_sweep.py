"""
threshold_sweep.py
==================
Pick the order-size threshold definition with
EVIDENCE, not instinct.

THE PROBLEM
-----------
The current split (33rd/67th percentile of
items-per-order) produces badly imbalanced ROWS:

    large  61.0%
    mid    25.0%
    small  14.0%

Because a 20-item order contributes 20 rows and a
3-item order contributes 3. So large orders swamp
the row count.

CTGAN struggles with rare classes - that is the
lesson from the night-ratio problem. If 'small' is
only 14% of training rows, conditioning on it will
generate from a thinly-learned class.

THE TRADE-OFF (this is why we test rather than guess)
-----------------------------------------------------
Balancing the rows means moving the 'small'
threshold UP. But if 'small' becomes "<= 12 items",
it may stop being meaningfully small, and the aisle
contrast could weaken or vanish.

It is genuinely possible the signal IS the
imbalance - that small baskets differ precisely
BECAUSE they are unusual. Balancing might destroy
the thing we are trying to capture.

That cannot be reasoned out. It must be measured.

WHAT THIS SCRIPT DOES
---------------------
Sweeps several threshold definitions. For each:
  - row balance (what CTGAN would actually see)
  - size-controlled tail TVD (the real signal)
  - top-5 aisle lists (does the mix genuinely move)

DECISION RULE (set BEFORE seeing results)
-----------------------------------------
  Prefer the definition where BOTH:
     min class share >= 0.30   (learnable)
     tail TVD        >= 0.10   (real signal)
  Among those, take the highest TVD.

  If NO definition satisfies both, keep the
  sharpest contrast and report the imbalance
  honestly as a limitation.

Noise floor for reference: weekend TVD = 0.0357
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DOMINANT = [24, 83]

print("=" * 66)
print("THRESHOLD SWEEP - balance vs signal")
print("=" * 66)

print("\nLoading...")
df = pd.read_csv('data/clean_orders_v3.csv',
                 nrows=8000000)
print(f"  rows:   {len(df):,}")
print(f"  orders: {df['order_id'].nunique():,}")

osize = df.groupby('order_id').size()
df['n_items'] = df['order_id'].map(osize)

print(f"\nItems per order:")
for q in [0.10, 0.25, 0.33, 0.50,
          0.67, 0.75, 0.90]:
    print(f"  {int(q*100):>3}th pct: "
          f"{osize.quantile(q):>5.0f} items")


def tvd(a, b):
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx, fill_value=0)
    b = b.reindex(idx, fill_value=0)
    return 0.5 * np.abs(a - b).sum()


def evaluate(lo, hi, name):
    """lo = max items for 'small'
       hi = min items for 'large'
       Two groups only (option C) - mid dropped."""
    d = df.copy()
    d['grp'] = np.where(
        d['n_items'] <= lo, 'small',
        np.where(d['n_items'] >= hi,
                 'large', 'mid'))

    two = d[d['grp'] != 'mid']
    if len(two) == 0:
        return None

    # ROW balance among the two conditioning
    # classes - this is what CTGAN sees
    share = two['grp'].value_counts(
        normalize=True)
    s_small = share.get('small', 0)
    s_large = share.get('large', 0)
    min_share = min(s_small, s_large)

    # what fraction of ALL rows we keep
    coverage = len(two) / len(d)

    # --- SIZE-CONTROLLED TVD ---
    # first 3 items only, so every order
    # contributes exactly 3 rows.
    # This removes the arithmetic artefact.
    f3 = two[two['is_early_in_cart'] == 1]
    sm = f3[f3['grp'] == 'small'][
        'aisle_id'].value_counts(normalize=True)
    lg = f3[f3['grp'] == 'large'][
        'aisle_id'].value_counts(normalize=True)
    if len(sm) == 0 or len(lg) == 0:
        return None

    full_tvd = tvd(sm, lg)

    smt = sm[~sm.index.isin(DOMINANT)]
    lgt = lg[~lg.index.isin(DOMINANT)]
    if smt.sum() == 0 or lgt.sum() == 0:
        return None
    smt = smt / smt.sum()
    lgt = lgt / lgt.sum()
    tail_tvd = tvd(smt, lgt)

    return {
        'name':      name,
        'lo':        lo,
        'hi':        hi,
        's_small':   s_small,
        's_large':   s_large,
        'min_share': min_share,
        'coverage':  coverage,
        'full_tvd':  full_tvd,
        'tail_tvd':  tail_tvd,
        'top_sm':    list(
            sm.head(5).index.astype(int)),
        'top_lg':    list(
            lg.head(5).index.astype(int)),
    }


# ══════════════════════════════════════════
# CANDIDATE DEFINITIONS
# ══════════════════════════════════════════
p = osize.quantile

CANDIDATES = [
    (int(p(0.33)), int(p(0.67)),
     "current (33/67 pct)"),
    (int(p(0.40)), int(p(0.60)),
     "40/60 pct"),
    (int(p(0.50)), int(p(0.50)) + 1,
     "median split (50/50)"),
    (int(p(0.55)), int(p(0.70)),
     "55/70 pct"),
    (int(p(0.60)), int(p(0.75)),
     "60/75 pct"),
    (int(p(0.65)), int(p(0.80)),
     "65/80 pct"),
    (int(p(0.25)), int(p(0.75)),
     "sharp (25/75 pct)"),
    (int(p(0.20)), int(p(0.80)),
     "sharpest (20/80 pct)"),
]

print("\n" + "=" * 66)
print("SWEEP RESULTS")
print("=" * 66)
print(f"\n{'Definition':<22}{'small<=':>8}"
      f"{'large>=':>8}{'minShr':>8}"
      f"{'cover':>7}{'tailTVD':>9}")
print("-" * 66)

results = []
for lo, hi, name in CANDIDATES:
    r = evaluate(lo, hi, name)
    if r is None:
        continue
    results.append(r)
    print(f"{r['name']:<22}{r['lo']:>8}"
          f"{r['hi']:>8}"
          f"{r['min_share']:>7.1%}"
          f"{r['coverage']:>7.1%}"
          f"{r['tail_tvd']:>9.4f}")

print("\n  minShr  = smallest class share of the")
print("            two conditioning classes.")
print("            This is what CTGAN sees.")
print("  cover   = fraction of all rows kept")
print("            (mid is dropped)")
print("  tailTVD = size-controlled, excl. 24/83")
print("\n  Noise floor (weekend): 0.0357")

# ══════════════════════════════════════════
# TOP AISLE LISTS
# ══════════════════════════════════════════
print("\n" + "=" * 66)
print("DOES THE AISLE MIX ACTUALLY MOVE?")
print("=" * 66)
for r in results:
    same = "SAME" if r['top_sm'] == r['top_lg'] \
        else "MOVES"
    print(f"\n  {r['name']}  [{same}]")
    print(f"    small: {r['top_sm']}")
    print(f"    large: {r['top_lg']}")

# ══════════════════════════════════════════
# DECISION - rule fixed in advance
# ══════════════════════════════════════════
print("\n" + "=" * 66)
print("DECISION")
print("=" * 66)
print("\n  Rule (set before seeing results):")
print("    min class share >= 30%  (learnable)")
print("    tail TVD        >= 0.10 (real signal)")
print("    -> among those, take highest TVD")

viable = [r for r in results
          if r['min_share'] >= 0.30
          and r['tail_tvd'] >= 0.10]

if viable:
    best = max(viable,
               key=lambda r: r['tail_tvd'])
    print(f"\n  VIABLE DEFINITIONS: {len(viable)}")
    for r in sorted(viable,
                    key=lambda x: -x['tail_tvd']):
        print(f"    {r['name']:<22} "
              f"share {r['min_share']:.1%}  "
              f"TVD {r['tail_tvd']:.4f}")

    print(f"""
  ==> USE: {best['name']}

      small  = <= {best['lo']} items
      large  = >= {best['hi']} items
      mid    = dropped from conditioning

      class balance: small {best['s_small']:.1%} /
                     large {best['s_large']:.1%}
      tail TVD:      {best['tail_tvd']:.4f}
                     ({best['tail_tvd']/0.0357:.1f}x
                      the noise floor)

  Both classes are learnable AND the aisle mix
  genuinely shifts. This is A+C: row-balanced,
  two groups.
""")
else:
    if results:
        sharp = max(results,
                    key=lambda r: r['tail_tvd'])
        print(f"""
  NO DEFINITION SATISFIES BOTH.

  Balancing the rows destroys the signal, or the
  signal only exists where the classes are
  imbalanced.

  Strongest contrast available:
      {sharp['name']}
      small <= {sharp['lo']}, large >= {sharp['hi']}
      min class share: {sharp['min_share']:.1%}
      tail TVD:        {sharp['tail_tvd']:.4f}

  ==> Keep the sharp thresholds. Accept the
      imbalance. Report honestly that 'small' is
      a minority class and conditioning on it is
      correspondingly weaker.

      That is an honest limitation, not a bug.
""")

print("=" * 66)