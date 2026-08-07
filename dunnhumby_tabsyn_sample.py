"""dunnhumby_tabsyn_sample.py - generate synthetic rows from the
trained dunnhumby_season TabSyn model.

WHY THIS FILE EXISTS RATHER THAN REUSING tabsyn_sample.py
---------------------------------------------------------
tabsyn_sample.py hard-asserts that its output lives under
data\\tabsyn\\ (its own protection rule). Dunnhumby artifacts must stay
in data\\dunnhumby\\, and tabsyn_sample.py is read-only, so its logic
is COPIED here per the protection rule ("copy it into the new file,
never edit the original"). The repo's own functions are called
unchanged, exactly as tabsyn_sample.py calls them.

Two deliberate differences:
  1. output must live under data\\dunnhumby\\ (asserted below)
  2. NO derive_features(). That helper is Instacart-specific - it
     derives is_weekend/is_peak_hour/time_of_day from
     order_hour_of_day and order_dow, columns this 4-column schema
     does not have. Applying it here would invent columns.

READS:  tabsyn_repo/data/dunnhumby_season/*, and the trained
        checkpoints under tabsyn_repo/tabsyn/(vae/)ckpt/dunnhumby_season/
WRITES: the --out CSV only (must live under data/dunnhumby/)

Usage:
  python dunnhumby_tabsyn_sample.py --num-samples 600000 ^
      --out data/dunnhumby/synthetic_season.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
REPO = PROJECT_ROOT / 'tabsyn_repo'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', default='dunnhumby_season')
    ap.add_argument('--num-samples', type=int, required=True)
    ap.add_argument('--out', required=True,
                    help='output CSV path relative to project root')
    ap.add_argument('--steps', type=int, default=50, help='NFEs')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    out_path = (PROJECT_ROOT / args.out).resolve()
    assert PROJECT_ROOT / 'data' / 'dunnhumby' in out_path.parents, \
        'FATAL: output must live under data\\dunnhumby\\ (separation)'
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
    syn_df.to_csv(out_path, index=False)

    print(f'rows: {len(syn_df):,}  cols: {list(syn_df.columns)}')
    print(f'time: {time.time() - start:.1f}s')
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
