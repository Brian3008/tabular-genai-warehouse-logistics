"""tabsyn_sample.py - generate synthetic rows from a trained TabSyn model.

Replicates tabsyn_repo/tabsyn/sample.py's main() body (their functions, called
unchanged) with one difference: an explicit --num-samples argument instead of
the hard-coded train-size row count, so the bench can match CTGAN's scoring
file size exactly (contract rule 2: matched sample sizes everywhere).

Bench contract (tabsyn_bench_contract.md):
- READS: tabsyn_repo/data/<dataname>/* and the trained checkpoints under
  tabsyn_repo/tabsyn/(vae/)ckpt/<dataname>/
- WRITES: the --out CSV only (must live under data/tabsyn/)

Usage:
  python tabsyn_sample.py --dataname warehouse_smoke --num-samples 10000 ^
      --out data/tabsyn/smoke_sample.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO = PROJECT_ROOT / 'tabsyn_repo'


# Copied VERBATIM from train_v3.py ("SHARED DERIVATION - single source of
# truth") so synthetic_tabsyn.csv carries the same 12 columns, derived the
# same way, as synthetic_v3.csv. Do not modify.
def derive_features(d):
    d = d.copy()
    h = pd.to_numeric(d['order_hour_of_day'],
                      errors='coerce')
    dow = pd.to_numeric(d['order_dow'],
                        errors='coerce')
    dspo = pd.to_numeric(
        d['days_since_prior_order'],
        errors='coerce').fillna(0)

    d['is_weekend'] = dow.isin([0, 1]).astype(int)
    d['is_peak_hour'] = (
        (h >= 9) & (h <= 15)).astype(int)
    d['is_night'] = ((h < 6) | (h > 21)).astype(int)
    d['time_of_day'] = pd.cut(
        h, bins=[-1, 6, 12, 17, 21, 24],
        labels=['night', 'morning', 'afternoon',
                'evening', 'late']).astype(str)
    d['order_frequency'] = pd.cut(
        dspo, bins=[-1, 0, 7, 14, 30, 999],
        labels=['first', 'weekly', 'biweekly',
                'monthly', 'infrequent']).astype(str)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True)
    ap.add_argument('--num-samples', type=int, required=True)
    ap.add_argument('--out', required=True,
                    help='output CSV path relative to project root')
    ap.add_argument('--steps', type=int, default=50, help='NFEs')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    out_path = (PROJECT_ROOT / args.out).resolve()
    assert PROJECT_ROOT / 'data' / 'tabsyn' in out_path.parents, \
        'FATAL: output must live under data\\tabsyn\\ (protection rules)'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    import torch
    from tabsyn.model import MLPDiffusion, Model
    from tabsyn.latent_utils import (get_input_generate, recover_data,
                                     split_num_cat_target)
    from tabsyn.diffusion_utils import sample

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    run_args = SimpleNamespace(dataname=args.dataname, gpu=args.gpu,
                               device=device, steps=args.steps)

    # body below mirrors tabsyn/sample.py main(), num_samples parameterized
    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = \
        get_input_generate(run_args)
    in_dim = train_z.shape[1]
    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    start = time.time()
    x_next = sample(model.denoise_fn_D, args.num_samples, in_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, cat_inverse, device)
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = {int(k): v
                        for k, v in info['idx_name_mapping'].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    syn_df = derive_features(syn_df)
    syn_df.to_csv(out_path, index=False)

    print(f'rows: {len(syn_df):,}  cols: {list(syn_df.columns)}')
    print(f'time: {time.time() - start:.1f}s')
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
