"""tabsyn_train.py - runner for TabSyn's UNMODIFIED training code.

Bench contract (tabsyn_bench_contract.md):
- READS: tabsyn_repo/data/<dataname>/*  (built by tabsyn_prepare_data.py)
- WRITES (new files only):
    stage vae       -> tabsyn_repo/tabsyn/vae/ckpt/<dataname>/{model,encoder,
                       decoder}.pt, train_z.npy
    stage diffusion -> tabsyn_repo/tabsyn/ckpt/<dataname>/model.pt (+ periodic)
- No repo file is edited. Two runner-level adjustments, both declared:
  1. DataLoader num_workers forced 4 -> 0 (Windows respawns workers every
     epoch; pure per-epoch overhead, no effect on the training algorithm).
  2. CWD is set to tabsyn_repo/ because their code uses relative paths.

Usage:
  python tabsyn_train.py --stage vae --dataname warehouse_smoke
  python tabsyn_train.py --stage diffusion --dataname warehouse_smoke
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent / 'tabsyn_repo'


def patch_dataloader_workers():
    import torch.utils.data as tud
    orig = tud.DataLoader

    def patched(*args, **kwargs):
        if kwargs.get('num_workers'):
            kwargs['num_workers'] = 0
        return orig(*args, **kwargs)

    tud.DataLoader = patched


def patch_scheduler_verbose():
    # torch >= 2.7 removed the deprecated 'verbose' kwarg that TabSyn's
    # (torch 2.0-era) code still passes; drop it, change nothing else
    import torch.optim.lr_scheduler as lrs
    orig = lrs.ReduceLROnPlateau

    class Patched(orig):
        def __init__(self, *args, **kwargs):
            kwargs.pop('verbose', None)
            super().__init__(*args, **kwargs)

    lrs.ReduceLROnPlateau = Patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['vae', 'diffusion'], required=True)
    ap.add_argument('--dataname', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    patch_dataloader_workers()   # must precede tabsyn module imports
    patch_scheduler_verbose()

    import torch
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'stage={args.stage} dataname={args.dataname} device={device}')

    run_args = SimpleNamespace(
        dataname=args.dataname,
        gpu=args.gpu,
        device=device,
        # VAE defaults from their argparse
        max_beta=1e-2,
        min_beta=1e-5,
        lambd=0.7,
    )

    if args.stage == 'vae':
        from tabsyn.vae.main import main as stage_main
    else:
        from tabsyn.main import main as stage_main
    stage_main(run_args)


if __name__ == '__main__':
    main()
