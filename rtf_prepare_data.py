"""
rtf_prepare_data.py
===================
Builds the PARENT (orders) / CHILD (items) tables for the REaLTabFormer
bench, from the RAW Instacart files, restricted to exactly the orders
CTGAN v3 and TabSyn trained on.

APPROVED SCHEMA (Brian, Jul 26 2026)
------------------------------------
PARENT  one row per order (27,664 after mid-drop)
    order_id                 join key - DROPPED by RTF before encoding
    order_dow                categorical  0-6
    order_hour_of_day        categorical  0-23
    days_since_prior_order   NUMERICAL    0-30 (NaN -> 0)
    order_size_grp           categorical  small | large

CHILD   one row per item (272,820 after mid-drop)
    order_id                 join key
    aisle_id                 categorical  134 distinct
    reordered                categorical  0 | 1

order_size is NOT in the parent: it is a deterministic function of the
child table's own length, and including it would recreate the v2
deterministic-redundancy defect (7 of 13 columns were functions of the
other 6). The basket-length distribution is compared against the real
one instead - stronger evidence, no redundant column.

add_to_cart_order is NOT a child column: it was verified to be exactly
1..n in file order on all 31,617 training orders (gate [5] below), so
the child row's POSITION carries it. Generating it as a column would
also push the longest basket to 547 decoder tokens, over RTF's default
output_max_length=512, and RTF SILENTLY DELETES over-long training rows
(data_utils.get_relational_input_ids sets labels=None, then
make_relational_dataset filters them). Those would be our LARGEST
baskets - the group the travel finding depends on.

DTYPES ARE LOAD-BEARING
-----------------------
data_utils.process_data picks numeric columns with
    df.select_dtypes(include=np.number)
so ANY int column is tokenised as a NUMBER (multiple digit tokens, and
treated as an ordered magnitude). CTGAN v3 and TabSyn treat aisle_id /
order_dow / order_hour_of_day / is_reorder as CATEGORICAL and only
days_since_prior_order as numerical. To match, every categorical column
is written and re-read as a STRING. RTF_DTYPES below is the single
source of truth; rtf_train.py re-applies it and asserts.

TOKEN BUDGET (from data_utils.get_relational_input_ids, verbatim)
-----------------------------------------------------------------
decoder label = [BOS] + SUM_items([BMEM] + 1 token per child column
                + [EMEM]) + [EOS]
              = 2 + n_items * (n_child_cols + 2)
With 2 child columns and a 109-item longest basket: 2 + 109*4 = 438,
inside the 512 default. Gate [6] asserts this from the DATA, not from
this comment.

READS  (read-only, never modified):
    data/orders.csv
    data/order_products__prior.csv
    data/products.csv
    data/v3_train.csv
    data/v3_train_order_ids.csv
    data/v3_eval.csv
    data/v3_compare.csv
WRITES (new files only):
    data/rtf/parent.csv
    data/rtf/child.csv
    data/rtf/prep_manifest.json
  --smoke additionally writes:
    data/rtf/parent_smoke.csv
    data/rtf/child_smoke.csv

Usage:
    python rtf_prepare_data.py            # full tables
    python rtf_prepare_data.py --smoke    # + 2,000-order smoke subset
"""

import sys

# Some prints use non-ASCII. A default Windows console is cp1252 and
# CRASHES on them; force UTF-8 so this runs outside PyCharm too.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'rtf'

# identical to train_v3.py / tabsyn_prepare_data.py
SMALL_MAX = 10
LARGE_MIN = 14

SEED = 20260726
SMOKE_ORDERS = 2000

PARENT_COLS = ['order_id', 'order_dow', 'order_hour_of_day',
               'days_since_prior_order', 'order_size_grp']
CHILD_COLS = ['order_id', 'aisle_id', 'reordered']

# THE dtype contract. Categorical -> str so process_data does not
# treat them as numbers. days_since_prior_order stays float.
RTF_DTYPES = {
    'order_dow': 'str',
    'order_hour_of_day': 'str',
    'order_size_grp': 'str',
    'aisle_id': 'str',
    'reordered': 'str',
    'days_since_prior_order': 'float64',
}
# Number of NON-join child columns -> tokens per item = this + 2
N_CHILD_VALUE_COLS = len(CHILD_COLS) - 1
RTF_OUTPUT_MAX_LENGTH = 512   # RTF default; gate [6] proves it is safe


def banner(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help=f'also write a {SMOKE_ORDERS}-order smoke subset')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('RTF PREPARE DATA - parent/child tables')
    print('=' * 70)
    print(f'READS : {DATA} (read-only)')
    print(f'WRITES: {OUT}\\*  (new files only)')

    # ══════════════════════════════════════════
    # [1] THE TRAINING ORDERS - and disjointness
    # ══════════════════════════════════════════
    banner('[1] Training orders + disjointness')

    v3_train = pd.read_csv(DATA / 'v3_train.csv')
    tr_ids_file = set(pd.read_csv(
        DATA / 'v3_train_order_ids.csv')['order_id'])
    ev_ids = set(pd.read_csv(DATA / 'v3_eval.csv',
                             usecols=['order_id'])['order_id'])
    cp_ids = set(pd.read_csv(DATA / 'v3_compare.csv',
                             usecols=['order_id'])['order_id'])
    tr_ids = set(v3_train['order_id'])

    assert tr_ids == tr_ids_file, \
        'FATAL: v3_train.csv order_ids differ from the saved manifest'
    assert len(tr_ids & ev_ids) == 0, \
        'FATAL: training orders overlap the eval set'
    assert len(tr_ids & cp_ids) == 0, \
        'FATAL: training orders overlap the compare set'

    print(f'    v3_train rows          : {len(v3_train):,}')
    print(f'    v3_train orders        : {len(tr_ids):,}')
    print(f'    train n eval           : {len(tr_ids & ev_ids)}  [OK]')
    print(f'    train n compare        : {len(tr_ids & cp_ids)}  [OK]')
    print('    [ASSERTED] RTF can only ever see training orders.')

    # ══════════════════════════════════════════
    # [2] RAW CHILD ROWS
    # ══════════════════════════════════════════
    banner('[2] Raw child rows (items)')

    prods = pd.read_csv(DATA / 'products.csv',
                        usecols=['product_id', 'aisle_id'])
    op = pd.read_csv(DATA / 'order_products__prior.csv')
    print(f'    order_products__prior  : {len(op):,}')

    child = op[op['order_id'].isin(tr_ids)].copy()
    child = child.merge(prods, on='product_id', how='left')

    assert child['aisle_id'].notna().all(), \
        'FATAL: a product has no aisle_id'
    assert child['add_to_cart_order'].notna().all(), \
        'FATAL: null add_to_cart_order'
    assert child['reordered'].notna().all(), \
        'FATAL: null reordered'
    assert set(child['reordered'].unique()) <= {0, 1}, \
        'FATAL: reordered is not binary'
    assert len(child) == len(v3_train), \
        (f'FATAL: rebuilt {len(child):,} child rows but v3_train has '
         f'{len(v3_train):,}')

    print(f'    child rows (all sizes) : {len(child):,}')
    print(f'    == v3_train rows       : True  [OK]')

    # ══════════════════════════════════════════
    # [3] RAW PARENT ROWS
    # ══════════════════════════════════════════
    banner('[3] Raw parent rows (orders)')

    orders = pd.read_csv(DATA / 'orders.csv')
    parent = orders[orders['order_id'].isin(tr_ids)][
        ['order_id', 'order_dow', 'order_hour_of_day',
         'days_since_prior_order']].copy()

    n_nan_dspo = int(parent['days_since_prior_order'].isna().sum())
    parent['days_since_prior_order'] = parent[
        'days_since_prior_order'].fillna(0).astype('float64')

    assert parent['order_id'].is_unique, \
        'FATAL: duplicate order_id in the parent table'
    assert len(parent) == len(tr_ids), \
        'FATAL: parent row count != training order count'
    assert set(child['order_id']).issubset(set(parent['order_id'])), \
        'FATAL: a child order_id is missing from the parent table'

    print(f'    parent rows            : {len(parent):,}')
    print(f'    days_since_prior NaN   : {n_nan_dspo:,} -> filled 0 '
          f'(first orders; same as prepare_data_v3.py)')
    print(f'    order_dow range        : {parent["order_dow"].min()}'
          f'-{parent["order_dow"].max()}')
    print(f'    order_hour range       : '
          f'{parent["order_hour_of_day"].min()}'
          f'-{parent["order_hour_of_day"].max()}')

    # ══════════════════════════════════════════
    # [4] SIZE BUCKETS - recomputed, never the stored column
    # ══════════════════════════════════════════
    banner('[4] order_size_grp - RECOMPUTED with 10/14')
    print('    (the stored 33/67 column in v3_*.csv is a trap -')
    print('     project-notes caveat 4. It is never read here.)')

    osize = child.groupby('order_id').size()
    grp = pd.Series(
        np.where(osize <= SMALL_MAX, 'small',
                 np.where(osize >= LARGE_MIN, 'large', 'mid')),
        index=osize.index, name='order_size_grp')

    print(f'\n    {"group":<8}{"orders":>10}{"share":>9}'
          f'{"item-rows":>12}{"share":>9}')
    print('    ' + '-' * 48)
    for g in ['small', 'mid', 'large']:
        keep = grp[grp == g].index
        n_o = len(keep)
        n_r = int(osize.loc[keep].sum())
        print(f'    {g:<8}{n_o:>10,}{n_o/len(grp):>9.1%}'
              f'{n_r:>12,}{n_r/len(child):>9.1%}')

    keep_ids = set(grp[grp != 'mid'].index)
    parent = parent[parent['order_id'].isin(keep_ids)].copy()
    child = child[child['order_id'].isin(keep_ids)].copy()
    parent['order_size_grp'] = parent['order_id'].map(grp)

    assert parent['order_size_grp'].isin(['small', 'large']).all(), \
        'FATAL: a mid order survived the drop'

    print(f'\n    after mid-drop: {len(parent):,} orders / '
          f'{len(child):,} item-rows')

    # ══════════════════════════════════════════
    # [5] GATE - add_to_cart_order == row position
    # ══════════════════════════════════════════
    banner('[5] GATE: add_to_cart_order == within-basket position')
    print('    is_early_in_cart is derived from the generated child')
    print('    ROW POSITION. That is only legitimate if position and')
    print('    add_to_cart_order are the same thing in the real data.')

    child = child.sort_values(['order_id', 'add_to_cart_order'],
                              kind='mergesort').reset_index(drop=True)

    pos = child.groupby('order_id').cumcount() + 1
    contiguous = bool((pos.values ==
                       child['add_to_cart_order'].values).all())
    assert contiguous, \
        ('FATAL: add_to_cart_order is not 1..n by position - '
         'is_early_in_cart CANNOT be derived from position')
    assert not child.duplicated(
        ['order_id', 'add_to_cart_order']).any(), \
        'FATAL: duplicate add_to_cart_order within an order'

    print(f'\n    1..n contiguous on all {parent.shape[0]:,} baskets: '
          f'{contiguous}  [OK]')
    print('    => is_early_in_cart = (position <= 3) is EXACTLY')
    print('       prepare_data_v3.py:158  (add_to_cart_order <= 3)')

    # ══════════════════════════════════════════
    # [6] GATE - decoder token budget, measured
    # ══════════════════════════════════════════
    banner('[6] GATE: decoder token budget (measured, not guessed)')

    sz = child.groupby('order_id').size()
    tokens_per_item = N_CHILD_VALUE_COLS + 2      # + [BMEM] + [EMEM]
    seq_len = 2 + sz * tokens_per_item            # + [BOS] + [EOS]
    longest = int(seq_len.max())
    n_over = int((seq_len > RTF_OUTPUT_MAX_LENGTH).sum())

    print(f'    formula (data_utils.get_relational_input_ids):')
    print(f'      2 + n_items * ({N_CHILD_VALUE_COLS} cols + 2 markers)'
          f' = 2 + n_items * {tokens_per_item}')
    print(f'    longest basket         : {int(sz.max())} items')
    print(f'    longest decoder label  : {longest} tokens')
    print(f'    output_max_length      : {RTF_OUTPUT_MAX_LENGTH}')
    print(f'    baskets that would be SILENTLY DROPPED: {n_over}')

    assert n_over == 0, \
        (f'FATAL: {n_over} baskets exceed output_max_length='
         f'{RTF_OUTPUT_MAX_LENGTH} and RTF would silently delete them. '
         f'Longest is {longest} tokens. STOP and get approval before '
         f'raising the limit.')
    print('    [ASSERTED] zero training baskets are dropped.')

    # ══════════════════════════════════════════
    # [7] GATE - flatten mapping reproduces v3_train
    # ══════════════════════════════════════════
    banner('[7] GATE: flatten mapping reproduces v3_train exactly')
    print('    The SAME mapping used on generated baskets is applied')
    print('    to the REAL parent/child tables here. If it does not')
    print('    rebuild v3_train bit-for-bit, the mapping is wrong and')
    print('    the whole flat comparison is silently broken.')

    flat = flatten(parent, child)

    FLAT7 = ['order_id', 'aisle_id', 'order_dow', 'order_hour_of_day',
             'is_reorder', 'is_early_in_cart', 'days_since_prior_order']

    # v3_train side: the same orders, mid dropped
    ref = v3_train[v3_train['order_id'].isin(keep_ids)][FLAT7].copy()

    A = ref.astype({c: 'float64' for c in FLAT7}).sort_values(
        FLAT7, kind='mergesort').reset_index(drop=True)
    B = flat[FLAT7].astype({c: 'float64' for c in FLAT7}).sort_values(
        FLAT7, kind='mergesort').reset_index(drop=True)

    print(f'\n    v3_train (mid-dropped) : {A.shape}')
    print(f'    rebuilt from parent/child: {B.shape}')
    assert A.shape == B.shape, 'FATAL: row count differs'
    exact = bool((A.values == B.values).all())
    assert exact, 'FATAL: flatten mapping does NOT reproduce v3_train'
    print(f'    EXACT MATCH on all 7 columns: {exact}  [OK]')

    # ══════════════════════════════════════════
    # [8] APPLY THE DTYPE CONTRACT + SAVE
    # ══════════════════════════════════════════
    banner('[8] dtype contract + save')
    print('    process_data uses select_dtypes(include=np.number), so')
    print('    an int column is tokenised as a NUMBER. Every')
    print('    categorical column is written as a STRING to match how')
    print('    CTGAN v3 and TabSyn treat it.')

    parent = parent[PARENT_COLS].copy()
    child_out = child[CHILD_COLS].copy()
    for c, t in RTF_DTYPES.items():
        if c in parent.columns:
            parent[c] = parent[c].astype(t)
        if c in child_out.columns:
            child_out[c] = child_out[c].astype(t)

    print(f'\n    {"column":<26}{"table":<9}{"dtype":<10}')
    print('    ' + '-' * 45)
    for c in PARENT_COLS:
        print(f'    {c:<26}{"parent":<9}{str(parent[c].dtype):<10}')
    for c in CHILD_COLS:
        print(f'    {c:<26}{"child":<9}{str(child_out[c].dtype):<10}')

    parent.to_csv(OUT / 'parent.csv', index=False)
    child_out.to_csv(OUT / 'child.csv', index=False)
    print(f'\n    saved {OUT / "parent.csv"}')
    print(f'    saved {OUT / "child.csv"}')

    # ══════════════════════════════════════════
    # [9] SMOKE SUBSET
    # ══════════════════════════════════════════
    smoke_info = None
    if args.smoke:
        banner(f'[9] Smoke subset ({SMOKE_ORDERS} orders)')
        rng = np.random.RandomState(SEED)
        ids = parent['order_id'].values.copy()
        rng.shuffle(ids)
        s_ids = set(ids[:SMOKE_ORDERS])
        sp = parent[parent['order_id'].isin(s_ids)]
        sc = child_out[child_out['order_id'].isin(s_ids)]
        # preserve within-basket order (child_out is already sorted)
        sp.to_csv(OUT / 'parent_smoke.csv', index=False)
        sc.to_csv(OUT / 'child_smoke.csv', index=False)
        ssz = sc.groupby('order_id').size()
        smoke_info = {
            'orders': int(len(sp)), 'items': int(len(sc)),
            'mean_basket': float(ssz.mean()),
            'max_basket': int(ssz.max()),
            'grp_share': {k: float(v) for k, v in
                          sp['order_size_grp'].value_counts(
                              normalize=True).items()},
        }
        print(f'    parent_smoke.csv: {len(sp):,} orders')
        print(f'    child_smoke.csv : {len(sc):,} items')
        print(f'    mean basket {ssz.mean():.2f}, max {ssz.max()}')

    # ══════════════════════════════════════════
    # [10] MANIFEST + PROPORTIONS (rule 3)
    # ══════════════════════════════════════════
    banner('[10] Proportions - the rule-3 baselines')
    print('    These are what generated output gets compared against.')
    print('    ORDER-share and ROW-share are DIFFERENT numbers and')
    print('    mixing them would be a real error.')

    small_sz = sz[grp.reindex(sz.index) == 'small']
    large_sz = sz[grp.reindex(sz.index) == 'large']
    order_share = {
        'small': float(len(small_sz) / len(sz)),
        'large': float(len(large_sz) / len(sz))}
    row_share = {
        'small': float(small_sz.sum() / sz.sum()),
        'large': float(large_sz.sum() / sz.sum())}

    print(f'\n    ORDER-level  small {order_share["small"]:.4f}   '
          f'large {order_share["large"]:.4f}')
    print(f'    ROW-level    small {row_share["small"]:.4f}   '
          f'large {row_share["large"]:.4f}')
    print(f'\n    mean basket size  overall {sz.mean():.3f}   '
          f'small {small_sz.mean():.3f}   large {large_sz.mean():.3f}')

    manifest = {
        'built': 'rtf_prepare_data.py',
        'seed': SEED,
        'small_max': SMALL_MAX, 'large_min': LARGE_MIN,
        'parent_rows': int(len(parent)),
        'child_rows': int(len(child_out)),
        'parent_cols': PARENT_COLS,
        'child_cols': CHILD_COLS,
        'rtf_dtypes': RTF_DTYPES,
        'disjoint_from_eval': True,
        'disjoint_from_compare': True,
        'gates': {
            'raw_child_rows_equal_v3_train': True,
            'add_to_cart_order_is_position': True,
            'flatten_reproduces_v3_train_exactly': True,
            'baskets_over_output_max_length': n_over,
        },
        'token_budget': {
            'tokens_per_item': int(tokens_per_item),
            'longest_basket_items': int(sz.max()),
            'longest_decoder_label_tokens': longest,
            'output_max_length': RTF_OUTPUT_MAX_LENGTH,
        },
        'basket_size': {
            'mean': float(sz.mean()),
            'median': float(sz.median()),
            'max': int(sz.max()),
            'mean_small': float(small_sz.mean()),
            'mean_large': float(large_sz.mean()),
        },
        'order_size_grp_order_share': order_share,
        'order_size_grp_row_share': row_share,
        'smoke': smoke_info,
    }
    (OUT / 'prep_manifest.json').write_text(
        json.dumps(manifest, indent=2))
    print(f'\n    saved {OUT / "prep_manifest.json"}')

    banner('DONE - all gates passed')
    print(f"""
  PARENT  {len(parent):,} rows   {PARENT_COLS}
  CHILD   {len(child_out):,} rows   {CHILD_COLS}

  Gates:
    raw child rows == v3_train rows            PASS
    train disjoint from eval / compare         PASS
    add_to_cart_order == basket position       PASS
    zero baskets over output_max_length        PASS
    flatten reproduces v3_train bit-for-bit    PASS
""")


def flatten(parent: pd.DataFrame, child: pd.DataFrame) -> pd.DataFrame:
    """THE mapping. Used on real tables (gate 7) and on generated
    baskets (rtf_sample.py) - identical code both times, so the
    comparison cannot silently diverge.

    child MUST already be ordered within each order_id: the row's
    position in its basket IS its add_to_cart_order.

        is_reorder       = int(reordered)          prepare_data_v3:157
        is_early_in_cart = int(position <= 3)      prepare_data_v3:158
    """
    d = child.copy()
    d['position'] = d.groupby('order_id').cumcount() + 1
    d['is_reorder'] = pd.to_numeric(d['reordered']).astype(int)
    d['is_early_in_cart'] = (d['position'] <= 3).astype(int)

    p = parent.set_index('order_id')
    for c in ['order_dow', 'order_hour_of_day',
              'days_since_prior_order', 'order_size_grp']:
        if c in p.columns:
            d[c] = d['order_id'].map(p[c])

    d['aisle_id'] = pd.to_numeric(d['aisle_id'])
    d['order_dow'] = pd.to_numeric(d['order_dow'])
    d['order_hour_of_day'] = pd.to_numeric(d['order_hour_of_day'])
    d['days_since_prior_order'] = pd.to_numeric(
        d['days_since_prior_order']).fillna(0)
    return d


if __name__ == '__main__':
    sys.exit(main())
