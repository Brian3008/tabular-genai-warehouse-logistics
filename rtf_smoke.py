"""
rtf_smoke.py
============
Day-1 smoke test for the REaLTabFormer bench. Proves the environment
and the API BEFORE any multi-day training is launched, and measures
sec/step so the epoch count is chosen from data instead of guessed.

GATES (each reports PASS/FAIL independently)
--------------------------------------------
 G1  environment: imports, CUDA, sm_120, pinned versions
 G2  data: smoke tables load, dtype contract holds, join is sound
 G3  parent (tabular) model fits; sec/step measured
 G4  child (relational) model fits; sec/step measured; ZERO training
     baskets dropped by output_max_length (asserted from RTF's own
     warning stream, not from arithmetic)
 G5  sampling works end to end (parent -> children)
 G6  ORDER GATE - planted-fault fixture. THE load-bearing one:
     is_early_in_cart is derived from the generated child row's
     POSITION, so RTF must return child rows in generation order.
     A fixture is trained where the child value IS the position
     (1,2,3,...). If order is preserved the generated baskets come
     back sorted; if it is scrambled, a basket of size k is sorted
     only 1/k! of the time. Planted signal + measured null, per the
     standing rules - not an eyeball check.
 G7  item-discard counter. rtf_sampler.py:409 SILENTLY discards a
     generated item whose column count is wrong (logging.warning
     only). A dropped item shifts every later position and would
     corrupt is_early_in_cart, so the rate is counted and reported.
 G8  flatten: generated baskets -> the 7-column flat schema
 G9  wall-clock projection for the full run

Nothing here trains a production model and nothing is scored.

READS  (read-only): data/rtf/parent_smoke.csv, data/rtf/child_smoke.csv,
                    data/rtf/prep_manifest.json
WRITES (new only) : results/rtf/smoke_report.json
                    data/rtf/smoke_ckpt/**  (throwaway checkpoints)
                    data/rtf/smoke_sample.csv
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import logging
import math
import shutil
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'rtf'
RES = ROOT / 'results' / 'rtf'
CKPT = DATA / 'smoke_ckpt'

SEED = 20260726

# smoke-scale training
SMOKE_EPOCHS = 3
BATCH_SIZE = 32

# fixture for the order gate
FIX_PARENTS = 600
FIX_BASKET = 5          # p(sorted by chance) = 1/5! = 0.83%
FIX_EPOCHS = 40

REPORT = {'gates': {}, 'timing': {}, 'notes': []}


def gate(name, ok, detail=''):
    REPORT['gates'][name] = {'pass': bool(ok), 'detail': detail}
    print(f"\n    [{'PASS' if ok else 'FAIL'}] {name}"
          + (f'  -  {detail}' if detail else ''))
    return ok


def banner(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


class WarnCatcher(logging.Handler):
    """RTF reports dropped training baskets and discarded generated
    items through warnings/logging, never through a return value. We
    capture BOTH streams so those events cannot pass unnoticed."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


@contextmanager
def capture_warnings():
    handler = WarnCatcher()
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = root.level
    root.setLevel(logging.WARNING)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        yield handler, wlist
    root.removeHandler(handler)
    root.setLevel(prev_level)


def all_messages(handler, wlist):
    return ([str(m) for m in handler.records]
            + [str(w.message) for w in wlist])


def rtf_save(model, path):
    """Work around a REaLTabFormer 0.2.4 bug.

    `save()` does json.dumps(self.__dict__) after manually converting
    ONLY checkpoints_dir and samples_save_dir to posix strings. But
    __init__ also does `self.full_save_dir = Path(full_save_dir)`
    (a 0.2.4 addition, the 'save full model' feature from issue #101),
    and save() never converts it -> every save raises
        TypeError: Object of type WindowsPath is not JSON serializable

    We stringify any OTHER Path attribute, save, then restore the
    originals. checkpoints_dir / samples_save_dir are deliberately
    left as Paths because save() itself calls .as_posix() on them.
    The library is NOT modified.
    """
    handled = {'checkpoints_dir', 'samples_save_dir'}
    stash = {k: v for k, v in model.__dict__.items()
             if isinstance(v, Path) and k not in handled}
    for k, v in stash.items():
        setattr(model, k, v.as_posix())
    try:
        model.save(str(path))
    finally:
        for k, v in stash.items():
            setattr(model, k, v)


def steps_per_epoch(model, n_rows):
    """RTF defaults to gradient_accumulation_steps=4, so an optimizer
    STEP consumes batch_size * 4 rows, not batch_size. Measured
    against the smoke run: 2,000 rows / (32*4) = 16 steps/epoch,
    x3 epochs = 48 -> exactly the 48 steps observed."""
    ga = int(model.training_args_kwargs.get(
        'gradient_accumulation_steps', 1))
    bs = int(model.training_args_kwargs.get(
        'per_device_train_batch_size', model.batch_size))
    return math.ceil(n_rows / (bs * ga)), bs, ga


# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-fixture', action='store_true',
                    help='skip G6/G7 (order gate) - NOT for a real run')
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    if CKPT.exists():
        shutil.rmtree(CKPT)
    CKPT.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('RTF DAY-1 SMOKE TEST')
    print('=' * 70)

    # ══════════════════════════════════════════
    banner('G1  environment')
    import torch
    import transformers
    import datasets
    from realtabformer import REaLTabFormer

    versions = {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'datasets': datasets.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
    }
    for k, v in versions.items():
        print(f'    {k:<16} {v}')
    REPORT['versions'] = versions

    cuda_ok = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability(0) if cuda_ok else None
    if cuda_ok:
        print(f'    device           {torch.cuda.get_device_name(0)}')
        print(f'    capability       sm_{cap[0]}{cap[1]}')
    env_ok = (cuda_ok and cap == (12, 0)
              and transformers.__version__ == '4.57.6')
    gate('G1 environment', env_ok,
         f'cuda={cuda_ok} cap={cap} transformers='
         f'{transformers.__version__}')
    if not env_ok:
        finish()
        return 1

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ══════════════════════════════════════════
    banner('G2  smoke data + dtype contract')

    manifest = json.loads((DATA / 'prep_manifest.json').read_text())
    dtypes = manifest['rtf_dtypes']

    def load(path, cols):
        d = pd.read_csv(path, dtype={c: t for c, t in dtypes.items()
                                     if c in cols and t == 'str'})
        for c in cols:
            if dtypes.get(c) == 'float64':
                d[c] = d[c].astype('float64')
        return d[cols]

    P_COLS = manifest['parent_cols']
    C_COLS = manifest['child_cols']
    parent = load(DATA / 'parent_smoke.csv', P_COLS)
    child = load(DATA / 'child_smoke.csv', C_COLS)

    print(f'    parent {parent.shape}   child {child.shape}')
    print('\n    dtypes as loaded:')
    for c in P_COLS:
        print(f'      parent.{c:<24} {parent[c].dtype}')
    for c in C_COLS:
        print(f'      child.{c:<25} {child[c].dtype}')

    # The dtype contract is load-bearing: process_data uses
    # select_dtypes(include=np.number), so an int column would be
    # tokenised as a NUMBER instead of a category.
    cat_ok = all(parent[c].dtype == object for c in
                 ['order_dow', 'order_hour_of_day', 'order_size_grp'])
    cat_ok &= all(child[c].dtype == object
                  for c in ['aisle_id', 'reordered'])
    num_ok = parent['days_since_prior_order'].dtype == np.float64
    join_ok = (parent['order_id'].is_unique
               and set(child['order_id']) <= set(parent['order_id']))

    gate('G2 data + dtypes', cat_ok and num_ok and join_ok,
         f'categoricals=object:{cat_ok} dspo=float:{num_ok} '
         f'join:{join_ok}')

    # ══════════════════════════════════════════
    banner('G3  parent (tabular) model fits')

    p_fit = parent.drop(columns=['order_id'])
    parent_model = REaLTabFormer(
        model_type='tabular',
        batch_size=BATCH_SIZE,
        epochs=SMOKE_EPOCHS,
        random_state=SEED,
        checkpoints_dir=str(CKPT / 'parent'),
        samples_save_dir=str(CKPT / 'parent_samples'),
        logging_steps=50,
        save_strategy='no',
        report_to=[],
    )

    t0 = time.time()
    try:
        # gen_kwargs={} / save_full_every_epoch=0 work around RTF
        # issue #103: 0.2.4's defaults (gen_kwargs=None) crash fit().
        # n_critic=0 here measures RAW step time; the sensitivity
        # (Q_delta) path is exercised separately in G3b.
        p_trainer = parent_model.fit(
            p_fit, device='cuda', n_critic=0,
            save_full_every_epoch=0, gen_kwargs={})
        p_secs = time.time() - t0
        p_steps = int(p_trainer.state.global_step)
        p_loss = [h for h in p_trainer.state.log_history if 'loss' in h]
        ok = p_steps > 0
    except Exception as e:
        p_secs, p_steps, p_loss, ok = time.time() - t0, 0, [], False
        REPORT['notes'].append(f'G3 parent fit failed: {e!r}')
        print(f'    EXCEPTION: {e!r}')

    if ok:
        print(f'\n    {p_steps} steps in {p_secs:.1f}s  '
              f'-> {p_secs/p_steps*1000:.1f} ms/step')
        REPORT['timing']['parent_sec_per_step'] = p_secs / p_steps
        REPORT['timing']['parent_rows_smoke'] = int(len(p_fit))
        if p_loss:
            print(f'    loss {p_loss[0].get("loss")} -> '
                  f'{p_loss[-1].get("loss")}')
    gate('G3 parent fit', ok, f'{p_steps} steps')

    # ══════════════════════════════════════════
    banner('G4  child (relational) model fits')

    parent_dir = CKPT / 'parent_model'
    child_ok = False
    c_secs = c_steps = 0
    dropped_msgs = []
    try:
        rtf_save(parent_model, parent_dir)
        # RTF writes id<timestamp>/ under the given directory
        saved = sorted(parent_dir.glob('id*'))
        parent_path = saved[-1] if saved else parent_dir
        print(f'    parent model saved -> {parent_path}')

        child_model = REaLTabFormer(
            model_type='relational',
            parent_realtabformer_path=str(parent_path),
            batch_size=BATCH_SIZE,
            epochs=SMOKE_EPOCHS,
            random_state=SEED,
            output_max_length=manifest['token_budget'][
                'output_max_length'],
            checkpoints_dir=str(CKPT / 'child'),
            samples_save_dir=str(CKPT / 'child_samples'),
            logging_steps=50,
            save_strategy='no',
            report_to=[],
        )

        t0 = time.time()
        with capture_warnings() as (h, wl):
            c_trainer = child_model.fit(
                df=child, in_df=parent, join_on='order_id',
                device='cuda')
            msgs = all_messages(h, wl)
        c_secs = time.time() - t0
        c_steps = int(c_trainer.state.global_step)

        dropped_msgs = [m for m in msgs
                        if 'has been removed from the training data' in m]
        child_ok = c_steps > 0

        print(f'\n    {c_steps} steps in {c_secs:.1f}s  '
              f'-> {c_secs/c_steps*1000:.1f} ms/step')
        print(f'    relational_max_length = '
              f'{child_model.relational_max_length} tokens')
        REPORT['timing']['child_sec_per_step'] = c_secs / c_steps
        REPORT['timing']['child_examples_smoke'] = int(len(parent))
        REPORT['relational_max_length'] = int(
            child_model.relational_max_length)
        c_loss = [x for x in c_trainer.state.log_history if 'loss' in x]
        if c_loss:
            print(f'    loss {c_loss[0].get("loss")} -> '
                  f'{c_loss[-1].get("loss")}')
    except Exception as e:
        REPORT['notes'].append(f'G4 child fit failed: {e!r}')
        print(f'    EXCEPTION: {e!r}')

    gate('G4 child fit', child_ok, f'{c_steps} steps')
    gate('G4b zero baskets dropped by output_max_length',
         child_ok and len(dropped_msgs) == 0,
         'no removal warning' if not dropped_msgs
         else f'DROPPED: {dropped_msgs}')

    # ══════════════════════════════════════════
    banner('G5  sampling end to end')

    sample_ok = False
    flat = None
    discard_msgs = []
    try:
        if not child_ok:
            raise RuntimeError('skipped - child model did not train')
        n_par = 40
        with capture_warnings() as (h, wl):
            syn_parent = parent_model.sample(
                n_samples=n_par, gen_batch=64, device='cuda')
            syn_parent = syn_parent.reset_index(drop=True)
            syn_parent['order_id'] = [
                f'S{i:06d}' for i in range(len(syn_parent))]

            syn_child = child_model.sample(
                input_unique_ids=syn_parent['order_id'],
                input_df=syn_parent.drop('order_id', axis=1),
                gen_batch=16, device='cuda')
            msgs = all_messages(h, wl)

        discard_msgs = [m for m in msgs if 'Discarding this observation' in m]

        print(f'    parents generated : {len(syn_parent)}')
        print(f'    child rows        : {len(syn_child)}')
        print(f'    child index name  : {syn_child.index.name}')
        print(f'    child columns     : {list(syn_child.columns)}')
        sample_ok = len(syn_child) > 0
    except Exception as e:
        REPORT['notes'].append(f'G5 sampling failed: {e!r}')
        print(f'    EXCEPTION: {e!r}')

    gate('G5 sampling', sample_ok)
    gate('G7 item-discard rate',
         sample_ok and len(discard_msgs) == 0,
         f'{len(discard_msgs)} items discarded mid-basket'
         if discard_msgs else 'none')
    REPORT['discarded_items_smoke'] = len(discard_msgs)

    # ══════════════════════════════════════════
    banner('G8  flatten generated baskets -> flat 7-column schema')

    if sample_ok:
        try:
            from rtf_prepare_data import flatten
            sc = syn_child.reset_index()
            sc = sc.rename(columns={sc.columns[0]: 'order_id'})
            flat = flatten(syn_parent, sc)
            FLAT7 = ['aisle_id', 'order_dow', 'order_hour_of_day',
                     'is_reorder', 'is_early_in_cart',
                     'days_since_prior_order', 'order_size_grp']
            have = [c for c in FLAT7 if c in flat.columns]
            print(f'    flat rows: {len(flat):,}')
            print(f'    columns present: {len(have)}/7  {have}')
            print(f'\n    is_early_in_cart rate {flat["is_early_in_cart"].mean():.3f}'
                  f'   (real 0.286)')
            print(f'    is_reorder rate       {flat["is_reorder"].mean():.3f}'
                  f'   (real 0.594)')
            flat.to_csv(DATA / 'smoke_sample.csv', index=False)
            gate('G8 flatten', len(have) == 7, f'{len(flat)} rows')
        except Exception as e:
            REPORT['notes'].append(f'G8 flatten failed: {e!r}')
            print(f'    EXCEPTION: {e!r}')
            gate('G8 flatten', False, repr(e))
    else:
        gate('G8 flatten', False, 'skipped - sampling failed')

    # ══════════════════════════════════════════
    banner('G6  ORDER GATE - planted-fault fixture')
    print("""    is_early_in_cart = (child row position <= 3), so RTF
    MUST return child rows in generation order.

    Fixture: the child's aisle_id IS its position (1,2,3,4,5).
    A model that preserves order returns sorted baskets. Under a
    scrambled order a size-5 basket is sorted by chance only
    1/5! = 0.83% of the time. Planted signal vs measured null.""")

    if args.skip_fixture:
        gate('G6 order preserved', False, 'SKIPPED by flag')
    else:
        try:
            rate, n_bask, null_rate = run_order_fixture(
                REaLTabFormer, torch)
            print(f'\n    baskets checked        : {n_bask}')
            print(f'    sorted-by-position rate: {rate:.1%}')
            print(f'    chance rate (1/{FIX_BASKET}!)       : '
                  f'{null_rate:.2%}')
            REPORT['order_gate'] = {
                'sorted_rate': rate, 'n_baskets': n_bask,
                'chance_rate': null_rate}
            gate('G6 order preserved', rate > 0.80,
                 f'{rate:.1%} sorted vs {null_rate:.2%} chance')
        except Exception as e:
            REPORT['notes'].append(f'G6 fixture failed: {e!r}')
            print(f'    EXCEPTION: {e!r}')
            gate('G6 order preserved', False, repr(e))

    # ══════════════════════════════════════════
    banner('G9  wall-clock projection for the FULL run')

    full = json.loads((DATA / 'prep_manifest.json').read_text())
    n_parent = full['parent_rows']

    # RTF defaults gradient_accumulation_steps=4, so an optimizer step
    # eats batch_size*4 rows. Verified against the smoke run.
    spe, bs, ga = steps_per_epoch(parent_model, n_parent)
    smoke_spe, _, _ = steps_per_epoch(parent_model, len(p_fit))

    print(f'    full parent rows         : {n_parent:,}')
    print(f'    batch size               : {bs}')
    print(f'    gradient_accumulation    : {ga}   '
          f'(rows per optimizer step = {bs*ga})')
    print(f'    steps/epoch (both models): {spe:,}')
    print('    (the relational dataset has ONE example per PARENT,')
    print('     not per item - so both models see the same count)')
    print(f'\n    sanity: smoke had {len(p_fit):,} rows -> '
          f'{smoke_spe} steps/epoch x {SMOKE_EPOCHS} epochs = '
          f'{smoke_spe*SMOKE_EPOCHS} steps; observed {p_steps}  '
          f'[{"OK" if smoke_spe*SMOKE_EPOCHS == p_steps else "MISMATCH"}]')
    REPORT['timing']['step_formula_verified'] = bool(
        smoke_spe * SMOKE_EPOCHS == p_steps)

    proj = {}
    EPOCH_GRID = [10, 25, 50, 75, 100, 150, 200]
    for tag in ['parent', 'child']:
        sps = REPORT['timing'].get(f'{tag}_sec_per_step')
        if not sps:
            continue
        print(f'\n    {tag}: {sps*1000:.1f} ms/step')
        print(f'      {"epochs":>8}{"steps":>12}{"hours":>10}')
        for ep in EPOCH_GRID:
            print(f'      {ep:>8}{spe*ep:>12,}'
                  f'{sps*spe*ep/3600:>10.2f}')
        proj[tag] = {str(ep): sps * spe * ep / 3600
                     for ep in EPOCH_GRID}

    if 'parent' in proj and 'child' in proj:
        print(f'\n    COMBINED parent+child wall clock:')
        print(f'      {"epochs":>8}{"hours":>10}')
        for ep in EPOCH_GRID:
            print(f'      {ep:>8}'
                  f'{proj["parent"][str(ep)]+proj["child"][str(ep)]:>10.2f}')
        proj['combined'] = {
            str(ep): proj['parent'][str(ep)] + proj['child'][str(ep)]
            for ep in EPOCH_GRID}

    REPORT['timing']['steps_per_epoch_full'] = spe
    REPORT['timing']['projection_hours'] = proj
    REPORT['timing']['batch_size'] = bs
    REPORT['timing']['gradient_accumulation_steps'] = ga

    finish()
    return 0


# ══════════════════════════════════════════════════════════
def run_order_fixture(REaLTabFormer, torch):
    """Train a tiny relational model where the child value IS the
    position, then measure how often generated baskets come back
    sorted. Returns (sorted_rate, n_baskets, chance_rate)."""
    fix_dir = CKPT / 'fixture'
    if fix_dir.exists():
        shutil.rmtree(fix_dir)
    fix_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)
    pids = [f'F{i:05d}' for i in range(FIX_PARENTS)]
    fp = pd.DataFrame({
        'order_id': pids,
        'grp': rng.choice(['a', 'b'], size=FIX_PARENTS).astype(str),
    })
    rows = []
    for pid in pids:
        for pos in range(1, FIX_BASKET + 1):
            rows.append({'order_id': pid, 'slot': str(pos)})
    fc = pd.DataFrame(rows)

    pm = REaLTabFormer(
        model_type='tabular', batch_size=64, epochs=FIX_EPOCHS,
        random_state=SEED,
        checkpoints_dir=str(fix_dir / 'p'),
        samples_save_dir=str(fix_dir / 'ps'),
        save_strategy='no', report_to=[], logging_steps=100)
    pm.fit(fp.drop(columns=['order_id']), device='cuda', n_critic=0,
           save_full_every_epoch=0, gen_kwargs={})
    rtf_save(pm, fix_dir / 'pmodel')
    saved = sorted((fix_dir / 'pmodel').glob('id*'))
    ppath = saved[-1] if saved else (fix_dir / 'pmodel')

    cm = REaLTabFormer(
        model_type='relational',
        parent_realtabformer_path=str(ppath),
        batch_size=64, epochs=FIX_EPOCHS, random_state=SEED,
        checkpoints_dir=str(fix_dir / 'c'),
        samples_save_dir=str(fix_dir / 'cs'),
        save_strategy='no', report_to=[], logging_steps=100)
    cm.fit(df=fc, in_df=fp, join_on='order_id', device='cuda')

    sp = fp.head(150).copy()
    sc = cm.sample(input_unique_ids=sp['order_id'],
                   input_df=sp.drop('order_id', axis=1),
                   gen_batch=32, device='cuda')
    sc = sc.reset_index()
    sc = sc.rename(columns={sc.columns[0]: 'order_id'})
    sc['slot'] = pd.to_numeric(sc['slot'], errors='coerce')

    sorted_ct = tot = 0
    for _, g in sc.groupby('order_id', sort=False):
        v = g['slot'].dropna().values
        if len(v) < 2:
            continue
        tot += 1
        if np.all(np.diff(v) > 0):
            sorted_ct += 1

    chance = 1.0 / math.factorial(FIX_BASKET)
    return (sorted_ct / tot if tot else 0.0), tot, chance


def finish():
    (RES / 'smoke_report.json').write_text(
        json.dumps(REPORT, indent=2, default=str))
    banner('SMOKE SUMMARY')
    for k, v in REPORT['gates'].items():
        print(f"    [{'PASS' if v['pass'] else 'FAIL'}] {k}")
    if REPORT['notes']:
        print('\n    notes:')
        for n in REPORT['notes']:
            print(f'      - {n}')
    print(f'\n    saved {RES / "smoke_report.json"}')


if __name__ == '__main__':
    sys.exit(main())
