"""
rtf_train.py
============
Production training run for the REaLTabFormer bench: parent (orders)
then child (items), on the full 27,664 / 272,820 tables.

--parent-epochs and --child-epochs are REQUIRED and have NO defaults.
That is deliberate. The epoch counts are approved by Brian against the
measured loss curves from rtf_probe.py before this script is ever run;
a default here would let a multi-hour run start on a number nobody
signed off (standing guard).

WHAT THIS RUN DOES DIFFERENTLY FROM THE PROBE
---------------------------------------------
- Q_delta sensitivity training (n_critic=5) is ON for the parent, and
  the FULL trace is saved, not just printed. This is the built-in
  overfitting detection and it is a deliverable in its own right.
- The child gets mask_rate > 0 (target masking). RTF's paper proposes
  it to prevent the decoder copying training values; data_utils.py:740
  confirms it is wired for the relational path. NOTE it is applied to
  the CHILD tokens.
- Models are saved to data/rtf/models/ and are the artifacts of record.

THE OVERFITTING GAP THIS SCRIPT DOCUMENTS
-----------------------------------------
realtabformer.py:461-501 routes model_type='tabular' through
_train_with_sensitivity (Q_delta, bootstrap, n_critic early stop) but
model_type='relational' through _fit_relational + a plain
trainer.train(). So RTF's built-in overfitting detection covers the
PARENT ONLY; the CHILD - the model that could memorise whole real
baskets - gets none. That gap is why rtf_memorisation.py adds our own
flat DCR, exact-match count, and a basket-level verbatim-copy rate
benchmarked against held-out real baskets.

READS  (read-only): data/rtf/parent.csv, data/rtf/child.csv,
                    data/rtf/prep_manifest.json
WRITES (new only) : data/rtf/models/parent/**, data/rtf/models/child/**
                    data/rtf/train_ckpt/**   (checkpoints, throwaway)
                    results/rtf/train/train_report.json
                    results/rtf/train/train_loss.png
                    results/rtf/train/train_stdout.log

Usage (after approval):
    python rtf_train.py --parent-epochs N --child-epochs M
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rtf_probe import (Tee, capped_bootstrap_workers, parse_critic,
                       parse_losses, plateau)
from rtf_smoke import rtf_save

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'rtf'
MODELS = DATA / 'models'
OUT = ROOT / 'results' / 'rtf' / 'train'
CKPT = DATA / 'train_ckpt'

SEED = 20260726
BATCH_SIZE = 32
N_CRITIC = 5
CHILD_MASK_RATE = 0.10

# Retain a child checkpoint every N epochs so the plateau knee can be
# SELECTED after the fact instead of retraining. Approved epoch pair is
# parent 30 / child 300, so this yields 12 checkpoints at epochs
# 25,50,...,300. ~825 MB each (measured on the probe) -> ~10 GB.
# This buys insurance against overtraining past the knee, which matters
# because the relational path has NO built-in overfitting detection.
CHILD_SAVE_EVERY_EPOCHS = 25

REPORT = {}


def banner(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


def load_tables():
    manifest = json.loads((DATA / 'prep_manifest.json').read_text())
    dtypes = manifest['rtf_dtypes']

    def load(path, cols):
        d = pd.read_csv(path, dtype={c: t for c, t in dtypes.items()
                                     if c in cols and t == 'str'})
        for c in cols:
            if dtypes.get(c) == 'float64':
                d[c] = d[c].astype('float64')
        return d[cols]

    parent = load(DATA / 'parent.csv', manifest['parent_cols'])
    child = load(DATA / 'child.csv', manifest['child_cols'])

    # The prep script already asserted these, but this run is long and
    # unattended - re-assert rather than trust a file on disk.
    assert len(parent) == manifest['parent_rows'], \
        'FATAL: parent.csv row count changed since prep'
    assert len(child) == manifest['child_rows'], \
        'FATAL: child.csv row count changed since prep'
    assert parent['order_id'].is_unique, \
        'FATAL: duplicate parent order_id'
    assert set(child['order_id']).issubset(set(parent['order_id'])), \
        'FATAL: orphan child order_id'
    assert parent['order_size_grp'].dtype == object, \
        'FATAL: order_size_grp is not categorical - dtype contract broken'
    assert child['aisle_id'].dtype == object, \
        'FATAL: aisle_id is not categorical - dtype contract broken'
    return parent, child, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent-epochs', type=int, required=True)
    ap.add_argument('--child-epochs', type=int, required=True)
    ap.add_argument('--mask-rate', type=float,
                    default=CHILD_MASK_RATE)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if CKPT.exists():
        shutil.rmtree(CKPT, ignore_errors=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    tee = Tee(OUT / 'train_stdout.log')
    sys.stdout = tee
    try:
        run(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()

    finalize((OUT / 'train_stdout.log').read_text(encoding='utf-8'),
             args)
    return 0


def run(args):
    import torch
    from realtabformer import REaLTabFormer

    print('=' * 70)
    print('RTF PRODUCTION TRAINING')
    print('=' * 70)
    print(f'  parent epochs : {args.parent_epochs}  (n_critic='
          f'{N_CRITIC}, Q_delta on)')
    print(f'  child epochs  : {args.child_epochs}  (mask_rate='
          f'{args.mask_rate})')

    parent, child, manifest = load_tables()
    print(f'\n  parent {parent.shape}   child {child.shape}')
    print('  [ASSERTED] row counts, join integrity, dtype contract')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    REPORT['config'] = {
        'parent_epochs': args.parent_epochs,
        'child_epochs': args.child_epochs,
        'n_critic': N_CRITIC,
        'child_mask_rate': args.mask_rate,
        'batch_size': BATCH_SIZE,
        'seed': SEED,
        'parent_rows': int(len(parent)),
        'child_rows': int(len(child)),
    }

    # ── PARENT ──
    banner(f'PARENT - {args.parent_epochs} epochs, Q_delta on')

    p_model = REaLTabFormer(
        model_type='tabular',
        batch_size=BATCH_SIZE,
        epochs=args.parent_epochs,
        random_state=SEED,
        checkpoints_dir=str(CKPT / 'parent'),
        samples_save_dir=str(CKPT / 'parent_samples'),
        logging_steps=25,
        report_to=[],
    )

    t0 = time.time()
    with capped_bootstrap_workers(target_jobs=3):
        p_trainer = p_model.fit(
            parent.drop(columns=['order_id']), device='cuda',
            n_critic=N_CRITIC, save_full_every_epoch=0, gen_kwargs={})
    p_secs = time.time() - t0

    pdir = MODELS / 'parent'
    if pdir.exists():
        shutil.rmtree(pdir, ignore_errors=True)
    rtf_save(p_model, pdir)
    ppath = sorted(pdir.glob('id*'))[-1]

    REPORT['parent'] = {
        'wall_seconds': p_secs,
        'final_global_step': int(p_trainer.state.global_step),
        'final_epoch_reached': float(p_trainer.state.epoch or 0),
        'model_path': str(ppath),
    }
    print(f'\n  PARENT done in {p_secs/60:.1f} min -> {ppath}')

    # ── CHILD ──
    banner(f'CHILD - {args.child_epochs} epochs, '
           f'mask_rate={args.mask_rate}')
    print('  NOTE: RTF runs NO overfitting detection on the relational')
    print('  path (realtabformer.py:494-501). Memorisation for the')
    print('  child is measured separately by rtf_memorisation.py.')

    # gradient_accumulation_steps=4 is an RTF default, so an optimizer
    # step consumes batch*4 rows and the relational dataset has one
    # example per PARENT.
    ga = int(REaLTabFormer(
        model_type='tabular').training_args_kwargs[
            'gradient_accumulation_steps'])
    spe = math.ceil(len(parent) / (BATCH_SIZE * ga))
    save_steps = spe * CHILD_SAVE_EVERY_EPOCHS
    keep = args.child_epochs // CHILD_SAVE_EVERY_EPOCHS + 2
    print(f'  steps/epoch {spe}  -> checkpoint every '
          f'{CHILD_SAVE_EVERY_EPOCHS} epochs = {save_steps} steps, '
          f'keeping {keep}')
    REPORT['child_checkpointing'] = {
        'steps_per_epoch': spe, 'save_steps': save_steps,
        'save_total_limit': keep,
        'every_n_epochs': CHILD_SAVE_EVERY_EPOCHS}

    c_model = REaLTabFormer(
        model_type='relational',
        parent_realtabformer_path=str(ppath),
        batch_size=BATCH_SIZE,
        epochs=args.child_epochs,
        random_state=SEED,
        mask_rate=args.mask_rate,
        output_max_length=manifest['token_budget'][
            'output_max_length'],
        checkpoints_dir=str(CKPT / 'child'),
        samples_save_dir=str(CKPT / 'child_samples'),
        logging_steps=25,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=keep,
        report_to=[],
    )

    t0 = time.time()
    c_trainer = c_model.fit(df=child, in_df=parent,
                            join_on='order_id', device='cuda')
    c_secs = time.time() - t0

    cdir = MODELS / 'child'
    if cdir.exists():
        shutil.rmtree(cdir, ignore_errors=True)
    rtf_save(c_model, cdir)
    cpath = sorted(cdir.glob('id*'))[-1]

    REPORT['child'] = {
        'wall_seconds': c_secs,
        'final_global_step': int(c_trainer.state.global_step),
        'final_epoch_reached': float(c_trainer.state.epoch or 0),
        'relational_max_length': int(c_model.relational_max_length),
        'mask_rate': args.mask_rate,
        'model_path': str(cpath),
    }
    print(f'\n  CHILD done in {c_secs/60:.1f} min -> {cpath}')

    ga = int(p_model.training_args_kwargs.get(
        'gradient_accumulation_steps', 1))
    REPORT['steps_per_epoch'] = math.ceil(
        len(parent) / (BATCH_SIZE * ga))


def finalize(log_text, args):
    banner('LOSS CURVES + Q_delta TRACE')

    losses = parse_losses(log_text)
    critic = parse_critic(log_text)

    p_curve, c_curve, cur, prev = [], [], 'p', -1.0
    for ep, ls in losses:
        if cur == 'p' and ep < prev - 1e-9:
            cur = 'c'
        (p_curve if cur == 'p' else c_curve).append((ep, ls))
        prev = ep

    for tag, curve in [('parent', p_curve), ('child', c_curve)]:
        if not curve:
            continue
        pl = plateau(curve)
        REPORT.setdefault(tag, {})['loss_curve'] = curve
        REPORT[tag]['plateau'] = pl
        print(f'\n  {tag}: loss {curve[0][1]:.4f} -> '
              f'{curve[-1][1]:.4f} over {curve[-1][0]:.1f} epochs')
        if pl:
            verdict = ('PLATEAUED' if pl['relative_improvement'] < 0.01
                       else 'STILL DESCENDING')
            print(f'    final-window improvement '
                  f'{pl["relative_improvement"]:+.2%} -> {verdict}')
            REPORT[tag]['verdict'] = verdict
            if verdict == 'STILL DESCENDING':
                print('    [!] the approved epoch count stopped this')
                print('        model mid-descent. Report before using.')

    if critic:
        REPORT['parent']['critic_rounds'] = critic
        breaches = [c for c in critic
                    if c['val_sensitivity'] >= c['threshold']]
        print(f'\n  Q_delta trace ({len(critic)} rounds, '
              f'{len(breaches)} breach(es)):')
        print(f'    {"round":>7}{"threshold":>13}{"val_sens":>12}'
              f'{"status":>12}')
        for c in critic:
            st = ('within' if c['val_sensitivity'] < c['threshold']
                  else 'BREACHED')
            print(f'    {c["round"]:>7}{c["threshold"]:>13.5f}'
                  f'{c["val_sensitivity"]:>12.5f}{st:>12}')
        REPORT['parent']['critic_breaches'] = len(breaches)

    try:
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
        for a, (tag, curve) in zip(ax, [('parent', p_curve),
                                        ('child', c_curve)]):
            if curve:
                a.plot([x for x, _ in curve], [y for _, y in curve],
                       lw=1.2, color='steelblue')
                a.set_title(f'{tag} training loss '
                            f'({REPORT.get(tag, {}).get("verdict","")})')
                a.set_xlabel('epoch')
                a.set_ylabel('loss')
                a.grid(alpha=0.3)
        if critic:
            r = [c['round'] for c in critic]
            v = [c['val_sensitivity'] for c in critic]
            th = critic[0]['threshold']
            ax[2].plot(r, v, 'o-', color='seagreen',
                       label='val_sensitivity')
            ax[2].axhline(th, ls='--', color='crimson',
                          label=f'threshold {th:.4f}')
            ax[2].set_title('Q_delta sensitivity (parent)\n'
                            'above the line = RTF calls overfitting')
            ax[2].set_xlabel('epoch')
            ax[2].legend(fontsize=8)
            ax[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / 'train_loss.png', dpi=150,
                    bbox_inches='tight')
        print(f'\n  saved {OUT / "train_loss.png"}')
    except Exception as e:
        print(f'  chart failed: {e!r}')

    (OUT / 'train_report.json').write_text(
        json.dumps(REPORT, indent=2, default=float))
    print(f'  saved {OUT / "train_report.json"}')

    banner('TRAINING DONE')
    print('  NEXT: rtf_sample.py (generate ~50,000 flattened rows),')
    print('  then the rtf_* known-answer gates before any scoring.')


if __name__ == '__main__':
    sys.exit(main())
