"""
rtf_probe.py
============
Convergence + cost probe on the FULL parent/child tables. This is a
MEASUREMENT run, not the production run. It answers the two things the
Day-1 smoke test could not:

  1. What does RTF's Q_delta sensitivity path (n_critic) actually COST?
     The smoke measured 106.3 ms/step on the n_critic=0 path. The
     production parent run needs n_critic=5, and _train_with_sensitivity
     generates gen_rounds * frac * n_rows = 3 * 0.165 * 27,664 ~ 13,700
     synthetic rows EVERY critic round. That overhead is unmeasured and
     could dominate the step time.

  2. Is the loss actually plateauing, and by which epoch?
     Three smoke epochs say nothing about convergence. An epoch count
     that fits the time budget but stops mid-descent gives a crippled
     model - that is a STOP signal, not a ship signal (Brian's guard 2).

Both models' loss curves are recorded per epoch and the plateau is
assessed numerically, not by eyeballing a chart.

WHY THE LOSS IS PARSED FROM STDOUT
----------------------------------
_train_with_sensitivity rebuilds the Trainer every n_critic epochs
(realtabformer.py:707-722), so the returned trainer.state.log_history
only covers the FINAL segment. The HuggingFace per-log-step dicts
printed to stdout cover every segment, so those are the complete
record. trainer.state is still captured as a cross-check.

READS  (read-only): data/rtf/parent.csv, data/rtf/child.csv,
                    data/rtf/prep_manifest.json
WRITES (new only) : results/rtf/probe/probe_report.json
                    results/rtf/probe/probe_loss.png
                    results/rtf/probe/probe_stdout.log
                    data/rtf/probe_ckpt/**   (throwaway)

Usage:
    python rtf_probe.py                       # 30 parent / 40 child
    python rtf_probe.py --parent-epochs 30 --child-epochs 40
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import ast
import json
import math
import re
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Imported HERE, before stdout is replaced by Tee: rtf_smoke runs a
# `sys.stdout.encoding` guard at import time, which a bare file-like
# object would fail.
from rtf_smoke import rtf_save

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'rtf'
OUT = ROOT / 'results' / 'rtf' / 'probe'
CKPT = DATA / 'probe_ckpt'

SEED = 20260726
BATCH_SIZE = 32
N_CRITIC = 5

REPORT = {}


class Tee:
    """Mirror stdout to a log file so RTF's own prints (the
    'Critic round: ... val_sensitivity: ...' lines and the HF loss
    dicts) are captured for parsing. RTF exposes neither through a
    return value."""

    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    # Anything that inspects sys.stdout (encoding guards, tqdm's
    # isatty check) must still work while stdout is replaced.
    @property
    def encoding(self):
        return getattr(self.stdout, 'encoding', 'utf-8')

    def isatty(self):
        return False

    def fileno(self):
        return self.stdout.fileno()

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def banner(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


class capped_bootstrap_workers:
    """Cap the sensitivity bootstrap's worker count WITHOUT changing
    any statistic.

    rtf_analyze.py:706-719 computes
        n_jobs = min(max(2, os.cpu_count() // 4), 16)
    and passes it straight to joblib.Parallel. On a 24-core box that
    is 6 workers, each allocating a float64 manhattan-distance matrix
    of (4*frac*N, frac*N) = (18256, 4564) = 636 MiB, plus cdist's own
    intermediates -> ~10 GB peak. With ~15 GB free this fails
    intermittently:
        numpy._core._exceptions._ArrayMemoryError: Unable to allocate
        636. MiB for an array with shape (18256, 4564)
    It killed one probe run at bootstrap round 12/500 while an earlier
    identical run happened to survive - i.e. it is memory-pressure
    flaky, which is unacceptable for a long unattended run.

    n_jobs is derived from os.cpu_count(), so temporarily reporting
    fewer CPUs caps the workers. num_bootstrap, frac, qt_max and every
    other parameter stay at RTF's defaults, so the computed
    sensitivity threshold is unchanged - only how many are computed at
    once. The library is NOT modified.
    """

    def __init__(self, target_jobs=3):
        # invert n_jobs = cpu_count // 4
        self.fake_cpus = max(4, target_jobs * 4)
        self.target_jobs = target_jobs
        self._real = None

    def __enter__(self):
        import os
        self._real = os.cpu_count
        os.cpu_count = lambda: self.fake_cpus
        print(f'    [bootstrap workers capped to '
              f'~{self.target_jobs} to bound peak RAM]')
        return self

    def __exit__(self, *exc):
        import os
        os.cpu_count = self._real
        return False


def parse_losses(log_text):
    """Pull every HuggingFace {'loss': .., 'epoch': ..} dict out of the
    captured stdout, in order."""
    out = []
    for m in re.finditer(r"\{'loss':.*?\}", log_text):
        try:
            d = ast.literal_eval(m.group(0))
            if 'loss' in d and 'epoch' in d:
                out.append((float(d['epoch']), float(d['loss'])))
        except Exception:
            continue
    return out


def parse_critic(log_text):
    """Pull RTF's sensitivity trace. Format (realtabformer.py:821):
    'Critic round: N, sensitivity_threshold: T, val_sensitivity: V, ...'
    """
    rounds = []
    for m in re.finditer(
            r'Critic round:\s*(\d+),\s*sensitivity_threshold:\s*'
            r'([-\d.eE+]+),\s*val_sensitivity:\s*([-\d.eE+]+)',
            log_text.replace('\\\n', '').replace('  ', ' ')):
        rounds.append({'round': int(m.group(1)),
                       'threshold': float(m.group(2)),
                       'val_sensitivity': float(m.group(3))})
    return rounds


def plateau(curve, tail_frac=0.30):
    """Numeric plateau test. Compare the mean loss over the last
    `tail_frac` of training against the segment before it. Returns the
    relative improvement; small => flat."""
    if len(curve) < 6:
        return None
    losses = [l for _, l in curve]
    n = len(losses)
    k = max(2, int(n * tail_frac))
    tail = float(np.mean(losses[-k:]))
    prev = float(np.mean(losses[-2 * k:-k])) if n >= 2 * k else \
        float(np.mean(losses[:-k]))
    rel = (prev - tail) / abs(prev) if prev else 0.0
    return {'prev_window_mean': prev, 'tail_window_mean': tail,
            'relative_improvement': rel,
            'window_epochs': k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent-epochs', type=int, default=30)
    ap.add_argument('--child-epochs', type=int, default=40)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if CKPT.exists():
        shutil.rmtree(CKPT, ignore_errors=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    tee = Tee(OUT / 'probe_stdout.log')
    sys.stdout = tee
    try:
        run(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()

    # parse the captured log AFTER restoring stdout
    log_text = (OUT / 'probe_stdout.log').read_text(encoding='utf-8')
    finalize(log_text, args)
    return 0


def run(args):
    import torch
    from realtabformer import REaLTabFormer

    print('=' * 70)
    print('RTF CONVERGENCE + COST PROBE (full data)')
    print('=' * 70)

    manifest = json.loads((DATA / 'prep_manifest.json').read_text())
    dtypes = manifest['rtf_dtypes']
    P_COLS, C_COLS = manifest['parent_cols'], manifest['child_cols']

    def load(path, cols):
        d = pd.read_csv(path, dtype={c: t for c, t in dtypes.items()
                                     if c in cols and t == 'str'})
        for c in cols:
            if dtypes.get(c) == 'float64':
                d[c] = d[c].astype('float64')
        return d[cols]

    parent = load(DATA / 'parent.csv', P_COLS)
    child = load(DATA / 'child.csv', C_COLS)
    print(f'\n    parent {parent.shape}   child {child.shape}')
    assert len(parent) == manifest['parent_rows']
    assert len(child) == manifest['child_rows']

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ══════════════════════════════════════════
    banner(f'PARENT - n_critic={N_CRITIC}, '
           f'{args.parent_epochs} epochs (Q_delta path)')
    print('    This is the path the production run uses. It generates')
    print(f'    ~{int(len(parent)*0.165)*3:,} rows every {N_CRITIC} '
          f'epochs to score sensitivity.')

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
    # save_full_every_epoch=0 is REQUIRED: any non-zero value makes
    # _train_with_sensitivity call self.save(), which hits the 0.2.4
    # full_save_dir WindowsPath JSON bug.
    with capped_bootstrap_workers(target_jobs=3):
        p_trainer = p_model.fit(
            parent.drop(columns=['order_id']), device='cuda',
            n_critic=N_CRITIC, save_full_every_epoch=0, gen_kwargs={})
    p_secs = time.time() - t0

    REPORT['parent'] = {
        'epochs_requested': args.parent_epochs,
        'n_critic': N_CRITIC,
        'wall_seconds': p_secs,
        'final_global_step': int(p_trainer.state.global_step),
        'final_epoch_reached': float(p_trainer.state.epoch or 0),
    }
    print(f'\n    PARENT wall clock: {p_secs/60:.1f} min')

    # ══════════════════════════════════════════
    banner(f'CHILD - relational, {args.child_epochs} epochs')

    pdir = CKPT / 'parent_model'
    rtf_save(p_model, pdir)
    saved = sorted(pdir.glob('id*'))
    ppath = saved[-1] if saved else pdir
    print(f'    parent model -> {ppath}')

    c_model = REaLTabFormer(
        model_type='relational',
        parent_realtabformer_path=str(ppath),
        batch_size=BATCH_SIZE,
        epochs=args.child_epochs,
        random_state=SEED,
        output_max_length=manifest['token_budget']['output_max_length'],
        checkpoints_dir=str(CKPT / 'child'),
        samples_save_dir=str(CKPT / 'child_samples'),
        logging_steps=25,
        report_to=[],
    )

    t0 = time.time()
    c_trainer = c_model.fit(df=child, in_df=parent,
                            join_on='order_id', device='cuda')
    c_secs = time.time() - t0

    REPORT['child'] = {
        'epochs_requested': args.child_epochs,
        'wall_seconds': c_secs,
        'final_global_step': int(c_trainer.state.global_step),
        'final_epoch_reached': float(c_trainer.state.epoch or 0),
        'relational_max_length': int(c_model.relational_max_length),
    }
    print(f'\n    CHILD wall clock: {c_secs/60:.1f} min')
    print(f'    relational_max_length: '
          f'{c_model.relational_max_length} tokens')

    ga = int(p_model.training_args_kwargs.get(
        'gradient_accumulation_steps', 1))
    REPORT['steps_per_epoch'] = math.ceil(
        len(parent) / (BATCH_SIZE * ga))
    REPORT['gradient_accumulation_steps'] = ga
    REPORT['batch_size'] = BATCH_SIZE
    REPORT['parent_rows'] = int(len(parent))
    REPORT['child_rows'] = int(len(child))


def finalize(log_text, args):
    banner('LOSS CURVES + PLATEAU ASSESSMENT')

    losses = parse_losses(log_text)
    critic = parse_critic(log_text)

    # split the loss stream into parent vs child by epoch resets:
    # the child run restarts at a low epoch after the parent finishes.
    p_curve, c_curve, cur = [], [], 'p'
    prev_ep = -1.0
    for ep, ls in losses:
        if cur == 'p' and ep < prev_ep - 1e-9:
            cur = 'c'
        (p_curve if cur == 'p' else c_curve).append((ep, ls))
        prev_ep = ep

    for tag, curve in [('parent', p_curve), ('child', c_curve)]:
        if not curve:
            print(f'\n    {tag}: no loss records parsed')
            continue
        pl = plateau(curve)
        REPORT.setdefault(tag, {})['loss_curve'] = curve
        REPORT[tag]['plateau'] = pl
        print(f'\n    {tag}: {len(curve)} log points, '
              f'epoch {curve[0][0]:.1f} -> {curve[-1][0]:.1f}')
        print(f'      loss {curve[0][1]:.4f} -> {curve[-1][1]:.4f}')
        if pl:
            print(f'      last {pl["window_epochs"]} points mean '
                  f'{pl["tail_window_mean"]:.4f} vs previous '
                  f'{pl["prev_window_mean"]:.4f}')
            print(f'      relative improvement in final window: '
                  f'{pl["relative_improvement"]:+.2%}')
            verdict = ('PLATEAUED' if pl['relative_improvement'] < 0.01
                       else 'STILL DESCENDING')
            print(f'      -> {verdict}')
            REPORT[tag]['verdict'] = verdict

    if critic:
        REPORT['parent']['critic_rounds'] = critic
        print(f'\n    Q_delta sensitivity trace '
              f'({len(critic)} critic rounds):')
        print(f'      {"round":>7}{"threshold":>13}'
              f'{"val_sens":>12}{"status":>14}')
        for c in critic:
            st = ('within' if c['val_sensitivity'] < c['threshold']
                  else 'BREACHED')
            print(f'      {c["round"]:>7}{c["threshold"]:>13.5f}'
                  f'{c["val_sensitivity"]:>12.5f}{st:>14}')
        print('\n    val_sensitivity BELOW threshold = the model is not')
        print('    yet closer to its training data than a fresh real')
        print('    sample is. A breach is RTF calling overfitting.')
    else:
        print('\n    [!] no critic rounds parsed - check the log')

    # ── cost extrapolation ──
    banner('COST: measured sensitivity overhead + projection')
    spe = REPORT.get('steps_per_epoch')
    pr, cr = REPORT.get('parent', {}), REPORT.get('child', {})
    if spe and pr.get('wall_seconds'):
        p_ep = pr.get('final_epoch_reached') or args.parent_epochs
        c_ep = cr.get('final_epoch_reached') or args.child_epochs
        p_per_ep = pr['wall_seconds'] / max(p_ep, 1e-9)
        c_per_ep = cr['wall_seconds'] / max(c_ep, 1e-9)
        print(f'    steps/epoch                 : {spe:,}')
        print(f'    parent sec/epoch (n_critic={N_CRITIC}): '
              f'{p_per_ep:.1f}')
        print(f'    child  sec/epoch            : {c_per_ep:.1f}')

        smoke = ROOT / 'results' / 'rtf' / 'smoke_report.json'
        if smoke.exists():
            s = json.loads(smoke.read_text())
            sps = s.get('timing', {}).get('parent_sec_per_step')
            if sps:
                plain = sps * spe
                print(f'\n    parent sec/epoch WITHOUT sensitivity '
                      f'(smoke): {plain:.1f}')
                print(f'    sensitivity overhead factor: '
                      f'{p_per_ep/plain:.2f}x')
                REPORT['sensitivity_overhead_factor'] = p_per_ep / plain

        print(f'\n    {"epochs":>8}{"parent h":>11}{"child h":>10}'
              f'{"total h":>10}')
        proj = {}
        for ep in [30, 50, 75, 100, 150, 200]:
            ph, ch = p_per_ep * ep / 3600, c_per_ep * ep / 3600
            print(f'    {ep:>8}{ph:>11.2f}{ch:>10.2f}{ph+ch:>10.2f}')
            proj[str(ep)] = {'parent_h': ph, 'child_h': ch,
                             'total_h': ph + ch}
        REPORT['projection_hours'] = proj

    # ── chart ──
    try:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for a, (tag, curve) in zip(ax, [('parent', p_curve),
                                        ('child', c_curve)]):
            if curve:
                e = [x for x, _ in curve]
                l = [y for _, y in curve]
                a.plot(e, l, lw=1.2, color='steelblue')
                a.set_title(f'{tag} training loss '
                            f'({REPORT.get(tag, {}).get("verdict", "")})')
                a.set_xlabel('epoch')
                a.set_ylabel('loss')
                a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / 'probe_loss.png', dpi=150,
                    bbox_inches='tight')
        print(f'\n    saved {OUT / "probe_loss.png"}')
    except Exception as e:
        print(f'    chart failed: {e!r}')

    (OUT / 'probe_report.json').write_text(
        json.dumps(REPORT, indent=2, default=float))
    print(f'    saved {OUT / "probe_report.json"}')

    banner('PROBE DONE - epoch number NOT chosen automatically')
    print('    Report the curves and the projection to Brian. He')
    print('    approves the production epoch count before any')
    print('    detached run starts (standing guard).')


if __name__ == '__main__':
    sys.exit(main())
