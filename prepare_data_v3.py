"""
prepare_data_v3.py
==================
Rebuilds the training data with a clean, honest
feature set.

WHAT CHANGED FROM v2 AND WHY
----------------------------
v2 trained CTGAN on 13 columns, but 7 of them were
DETERMINISTIC functions of the other 6:

    is_weekend       = f(order_dow)
    is_peak_hour     = f(order_hour_of_day)
    is_night         = f(order_hour_of_day)
    time_of_day      = f(order_hour_of_day)
    order_frequency  = f(days_since_prior_order)
    aisle_popularity = f(aisle_id)        <-- worst
    department_id    = f(aisle_id)

Problems this caused:
  1. Wasted model capacity learning if-statements.
  2. Inflated the correlation score - deterministic
     relationships are trivial to preserve.
  3. aisle_popularity is aisle_id in disguise, so
     conditioning on it silently constrained aisle.
     That is what made the "98.7% on free features"
     result invalid.
  4. Generated rows could be internally inconsistent
     (is_night=1 next to hour=14), which is what the
     old post-processing was patching over.

v3 trains on ONLY the independent columns and
derives the rest AFTER generation, from a single
shared function. Derived features are then always
consistent by construction, and there is nothing
to calibrate.

NEW: demand_type
----------------
The structure test showed:
    aisle ~ is_weekend    TVD 0.037  (noise)
    aisle ~ is_peak_hour  TVD 0.029  (noise)
    aisle ~ is_reorder    TVD 0.172  (REAL)

Calendar flags do not move the aisle mix.
Reorder behaviour does. So scenarios are built on
reorder intensity, which is a genuinely learnable
signal, rather than on dates, which are not.

demand_type is computed per ORDER from the share of
items that are reorders. It is upstream of aisle,
not derived from it.

OUTPUT: data/clean_orders_v3.csv
Nothing from v2 is overwritten.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 62)
print("PREPARE DATA v3 - CLEAN FEATURE SET")
print("=" * 62)

# ══════════════════════════════════════════
# SHARED DERIVATION FUNCTION
# ══════════════════════════════════════════
# This is the SINGLE source of truth for every
# derived feature. Used here on real data, and
# used again after generation on synthetic data.
# Same function both times => guaranteed
# consistency, no calibration ever needed.

def derive_features(df):
    """Add all deterministic features from the
    six independent base columns. Safe to call on
    real OR synthetic data."""
    d = df.copy()

    h = pd.to_numeric(d['order_hour_of_day'],
                      errors='coerce')
    dow = pd.to_numeric(d['order_dow'],
                        errors='coerce')
    dspo = pd.to_numeric(
        d['days_since_prior_order'],
        errors='coerce').fillna(0)

    # --- from order_dow ---
    d['is_weekend'] = (
        dow.isin([0, 1])).astype(int)

    # --- from order_hour_of_day ---
    d['is_peak_hour'] = (
        (h >= 9) & (h <= 15)).astype(int)
    d['is_night'] = (
        (h < 6) | (h > 21)).astype(int)
    d['time_of_day'] = pd.cut(
        h,
        bins=[-1, 6, 12, 17, 21, 24],
        labels=['night', 'morning', 'afternoon',
                'evening', 'late']
    ).astype(str)

    # --- from days_since_prior_order ---
    d['order_frequency'] = pd.cut(
        dspo,
        bins=[-1, 0, 7, 14, 30, 999],
        labels=['first', 'weekly', 'biweekly',
                'monthly', 'infrequent']
    ).astype(str)

    return d


if __name__ == '__main__':

    # ══════════════════════════════════════
    # STEP 1 - LOAD RAW
    # ══════════════════════════════════════
    print("\n[1] Loading raw Instacart files...")
    orders = pd.read_csv('data/orders.csv')
    products = pd.read_csv('data/products.csv')
    order_products = pd.read_csv(
        'data/order_products__prior.csv')
    print(f"    orders:         {len(orders):,}")
    print(f"    order_products: "
          f"{len(order_products):,}")

    # ══════════════════════════════════════
    # STEP 2 - MERGE
    # ══════════════════════════════════════
    print("\n[2] Merging...")
    df = order_products.merge(
        products[['product_id', 'aisle_id',
                  'department_id']],
        on='product_id', how='left')

    df = df.merge(
        orders[['order_id', 'user_id',
                'order_dow', 'order_hour_of_day',
                'days_since_prior_order',
                'order_number']],
        on='order_id', how='left')

    print(f"    merged: {df.shape}")

    # ══════════════════════════════════════
    # STEP 3 - BASE (INDEPENDENT) COLUMNS
    # ══════════════════════════════════════
    print("\n[3] Building independent base columns...")

    df['is_reorder'] = df['reordered'].astype(int)
    df['is_early_in_cart'] = (
        df['add_to_cart_order'] <= 3).astype(int)
    df['days_since_prior_order'] = df[
        'days_since_prior_order'].fillna(0)

    # These SIX are what CTGAN will train on.
    # Nothing here is a function of anything
    # else here.
    BASE_COLS = [
        'aisle_id',
        'order_dow',
        'order_hour_of_day',
        'is_reorder',
        'is_early_in_cart',
        'days_since_prior_order',
    ]
    print(f"    base columns: {BASE_COLS}")

    # department_id is kept for reference /
    # simulator use, but is NOT trained on
    # (it is a lookup from aisle_id).
    print("    department_id kept for reference"
          " only (derived from aisle_id)")

    # ══════════════════════════════════════
    # STEP 4 - DEMAND TYPE (the real signal)
    # ══════════════════════════════════════
    print("\n[4] Computing demand_type per order...")
    print("    (based on reorder share - the only")
    print("     axis that actually shifts aisle mix)")

    order_reorder_share = df.groupby(
        'order_id')['is_reorder'].mean()

    # Thirds: novelty / mixed / replenishment
    q33 = order_reorder_share.quantile(0.33)
    q67 = order_reorder_share.quantile(0.67)

    def label(share):
        if share <= q33:
            return 'novelty'        # mostly new items
        if share >= q67:
            return 'replenishment'  # mostly rebuys
        return 'mixed'

    demand_map = order_reorder_share.apply(label)
    df['demand_type'] = df['order_id'].map(
        demand_map)

    print(f"    novelty threshold:       "
          f"<= {q33:.3f} reorder share")
    print(f"    replenishment threshold: "
          f">= {q67:.3f} reorder share")
    print("\n    distribution:")
    for k, v in df['demand_type'].value_counts(
            normalize=True).items():
        print(f"      {k:<15} {v:6.1%}")

    # ══════════════════════════════════════
    # STEP 5 - DERIVE THE REST
    # ══════════════════════════════════════
    print("\n[5] Deriving dependent features...")
    df = derive_features(df)
    print("    is_weekend, is_peak_hour, is_night,")
    print("    time_of_day, order_frequency")
    print("    (NOT trained on - derived after)")

    # ══════════════════════════════════════
    # STEP 6 - AISLE POPULARITY (reference only)
    # ══════════════════════════════════════
    # Kept ONLY for analysis/reporting.
    # NEVER trained on. NEVER conditioned on.
    # It is aisle_id in disguise.
    counts = df['aisle_id'].value_counts()
    total = counts.sum()
    df['aisle_popularity'] = df['aisle_id'].map(
        counts / total).apply(
        lambda p: 'high' if p > 0.02
        else 'medium' if p > 0.005
        else 'low')
    print("\n[6] aisle_popularity computed for")
    print("    REFERENCE ONLY - never trained on,")
    print("    never conditioned on (it is a")
    print("    lookup from aisle_id).")

    # ══════════════════════════════════════
    # STEP 7 - SAVE
    # ══════════════════════════════════════
    FINAL_COLS = (
        ['order_id']
        + BASE_COLS
        + ['demand_type', 'department_id',
           'is_weekend', 'is_peak_hour',
           'is_night', 'time_of_day',
           'order_frequency',
           'aisle_popularity']
    )

    clean = df[FINAL_COLS].dropna()
    clean.to_csv('data/clean_orders_v3.csv',
                 index=False)

    print("\n" + "=" * 62)
    print("SAVED data/clean_orders_v3.csv")
    print("=" * 62)
    print(f"  rows:            {len(clean):,}")
    print(f"  unique orders:   "
          f"{clean['order_id'].nunique():,}")
    print(f"  unique aisles:   "
          f"{clean['aisle_id'].nunique()}")

    print(f"\n  TRAINED ON (6 independent):")
    for c in BASE_COLS:
        print(f"    - {c}")
    print(f"\n  CANDIDATE CONDITIONING AXIS "
          f"(kept in the data):")
    print(f"    - demand_type "
          f"(novelty/mixed/replenishment)")
    print(f"    NOTE: the axis v3 actually trains on is")
    print(f"    order_size_grp (small<=10 / large>=14),")
    print(f"    chosen later by signal_search.py and the")
    print(f"    threshold screen - demand_type lost")
    print(f"    (lift +2.8%, nearly nothing).")
    print(f"\n  DERIVED AFTER GENERATION "
          f"(never trained):")
    for c in ['is_weekend', 'is_peak_hour',
              'is_night', 'time_of_day',
              'order_frequency']:
        print(f"    - {c}")
    print(f"\n  REFERENCE ONLY (never used in model):")
    print(f"    - department_id")
    print(f"    - aisle_popularity")

    # ══════════════════════════════════════
    # STEP 8 - SANITY CHECK THE SIGNAL
    # ══════════════════════════════════════
    print("\n" + "=" * 62)
    print("SANITY CHECK: does demand_type actually")
    print("shift the aisle mix?")
    print("=" * 62)

    def tvd(a, b):
        idx = sorted(set(a.index) | set(b.index))
        a = a.reindex(idx, fill_value=0)
        b = b.reindex(idx, fill_value=0)
        return 0.5 * np.abs(a - b).sum()

    nov = clean[clean['demand_type'] == 'novelty'][
        'aisle_id'].value_counts(normalize=True)
    rep = clean[
        clean['demand_type'] == 'replenishment'][
        'aisle_id'].value_counts(normalize=True)

    d = tvd(nov, rep)
    print(f"\n  TVD(novelty vs replenishment) "
          f"= {d:.4f}")
    print(f"  (compare: weekend vs weekday was "
          f"0.0365 = noise)")

    print(f"\n  Top 5 aisles - novelty:      "
          f"{list(nov.head(5).index.astype(int))}")
    print(f"  Top 5 aisles - replenishment:"
          f" {list(rep.head(5).index.astype(int))}")

    if d > 0.10:
        print("\n  GOOD - demand_type genuinely")
        print("  shifts which shelves get hit.")
        print("  This is a learnable signal.")
    elif d > 0.05:
        print("\n  MODERATE - some shift present.")
    else:
        print("\n  WEAK - reconsider the split.")

    print("\nDone.")