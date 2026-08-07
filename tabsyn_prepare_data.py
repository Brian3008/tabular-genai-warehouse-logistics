"""tabsyn_prepare_data.py - build TabSyn-format training data from v3_train.csv.

Bench contract (tabsyn_bench_contract.md):
- READS (read-only): data/v3_train.csv
- WRITES (new files only): tabsyn_repo/data/<dataname>/{X_num_train,X_cat_train,
  y_train,X_num_test,X_cat_test,y_test}.npy, info.json, train.csv,
  prep_manifest.json
- Never touches v3_eval.csv / v3_compare.csv. TabSyn's internal "test" split
  (used only by its LR scheduler / best-checkpoint selection during VAE
  training) is a random SUBSET COPY of the training rows - the model still
  trains on ALL rows, same as CTGAN v3, and the real held-out eval side is
  never seen by anything in this pipeline.

Column contract mirrors train_v3.py exactly:
  6 BASE_COLS + order_size_grp RECOMPUTED with screened buckets small<=10 /
  large>=14 (never the stored 33/67 column - project-notes caveat 4), mid dropped.
  days_since_prior_order numerical; everything else categorical;
  is_reorder is the binclass "target" (TabSyn folds it into the categorical
  block internally - same information, different file layout).

Usage:
  python tabsyn_prepare_data.py --smoke   -> dataname warehouse_smoke (10k rows)
  python tabsyn_prepare_data.py           -> dataname warehouse (all rows)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO = PROJECT_ROOT / 'tabsyn_repo'

# identical to train_v3.py
SMALL_MAX = 10
LARGE_MIN = 14
BASE_COLS = [
    'aisle_id',
    'order_dow',
    'order_hour_of_day',
    'is_reorder',
    'is_early_in_cart',
    'days_since_prior_order',
]
COND_COL = 'order_size_grp'

# column layout handed to TabSyn (original-order indices)
ALL_COLS = BASE_COLS[:3] + ['is_reorder', 'is_early_in_cart',
                            'days_since_prior_order', COND_COL]
NUM_COL_IDX = [5]                 # days_since_prior_order
CAT_COL_IDX = [0, 1, 2, 4, 6]     # aisle, dow, hour, eic, size_grp
TARGET_COL_IDX = [3]              # is_reorder (binclass)

VAL_FRACTION = 0.10               # subset COPY of train, scheduler-only
SEED = 20260719


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='10k-row smoke dataset (warehouse_smoke)')
    args = ap.parse_args()

    dataname = 'warehouse_smoke' if args.smoke else 'warehouse'
    out_dir = REPO / 'data' / dataname
    out_dir.mkdir(parents=True, exist_ok=True)

    src_csv = PROJECT_ROOT / 'data' / 'v3_train.csv'
    print(f'READS : {src_csv} (read-only)')
    print(f'WRITES: {out_dir}\\*  (new files only)')

    train = pd.read_csv(src_csv)

    # recompute size groups with the SCREENED buckets - never the stored column
    osize = train.groupby('order_id').size()
    train['n_items'] = train['order_id'].map(osize)
    grp = np.where(train['n_items'] <= SMALL_MAX, 'small',
                   np.where(train['n_items'] >= LARGE_MIN, 'large', 'mid'))
    train = train.assign(**{COND_COL: grp})
    td = train[train[COND_COL] != 'mid'].copy()

    X = td[ALL_COLS].copy()
    assert not X.isna().any().any(), 'FATAL: NaNs in training columns'
    assert list(X.columns) == ALL_COLS

    if args.smoke:
        X = X.sample(n=10_000, random_state=SEED).reset_index(drop=True)

    n = len(X)
    rng = np.random.RandomState(SEED)
    val_idx = rng.choice(n, size=int(n * VAL_FRACTION), replace=False)
    val = X.iloc[val_idx]           # subset COPY; X itself stays complete

    def dump(frame, split):
        num = frame.iloc[:, NUM_COL_IDX].to_numpy().astype(np.float32)
        cat = frame.iloc[:, CAT_COL_IDX].astype(str).to_numpy()
        y = frame.iloc[:, TARGET_COL_IDX].astype(str).to_numpy()
        np.save(out_dir / f'X_num_{split}.npy', num)
        np.save(out_dir / f'X_cat_{split}.npy', cat)
        np.save(out_dir / f'y_{split}.npy', y)
        return num.shape, cat.shape, y.shape

    tr_shapes = dump(X, 'train')
    va_shapes = dump(val, 'test')
    X.to_csv(out_dir / 'train.csv', index=False)

    column_names = list(X.columns)
    # num cols first, then cat, then target - their canonical ordering
    idx_mapping, curr = {}, 0
    for i in NUM_COL_IDX:
        idx_mapping[i] = curr; curr += 1
    for i in CAT_COL_IDX:
        idx_mapping[i] = curr; curr += 1
    for i in TARGET_COL_IDX:
        idx_mapping[i] = curr; curr += 1
    inverse_idx_mapping = {v: k for k, v in idx_mapping.items()}
    idx_name_mapping = {i: c for i, c in enumerate(column_names)}

    info = {
        'name': dataname,
        'task_type': 'binclass',
        'n_classes': 2,
        'header': 'infer',
        'column_names': column_names,
        'num_col_idx': NUM_COL_IDX,
        'cat_col_idx': CAT_COL_IDX,
        'target_col_idx': TARGET_COL_IDX,
        'file_type': 'csv',
        'data_path': f'data/{dataname}/train.csv',
        'test_path': None,
        'train_num': n,
        'test_num': len(val),
        'idx_mapping': idx_mapping,
        'inverse_idx_mapping': inverse_idx_mapping,
        'idx_name_mapping': idx_name_mapping,
    }
    (out_dir / 'info.json').write_text(json.dumps(info, indent=4))

    shares = X[COND_COL].value_counts(normalize=True)
    manifest = {
        'dataname': dataname,
        'seed': SEED,
        'source': str(src_csv),
        'rows_after_mid_drop': int(len(td)),
        'rows_used': n,
        'val_subset_rows': int(len(val)),
        'val_is_subset_of_train': True,
        'order_size_grp_share': {k: float(v) for k, v in shares.items()},
        'is_reorder_share': float((X['is_reorder'].astype(int) == 1).mean()),
        'shapes': {'train': [list(s) for s in tr_shapes],
                   'val': [list(s) for s in va_shapes]},
    }
    (out_dir / 'prep_manifest.json').write_text(json.dumps(manifest, indent=4))

    print(f'\nrows after mid-drop: {len(td):,} (full) -> used: {n:,}')
    print(f'val subset (scheduler-only, copy of train rows): {len(val):,}')
    print(f'order_size_grp shares: '
          + ', '.join(f'{k}={v:.1%}' for k, v in shares.items()))
    print('Done - TabSyn-format data written.')


if __name__ == '__main__':
    sys.exit(main())
