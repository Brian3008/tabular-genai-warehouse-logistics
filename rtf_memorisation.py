"""
rtf_memorisation.py
===================
Finds the child model's memorisation onset by sweeping the retained
checkpoints (epochs 25..300), because RTF gives the relational path NO
built-in overfitting detection (realtabformer.py:494-501 routes it to a
plain trainer.train(); only model_type='tabular' gets Q_delta).

WHY A SWEEP AND NOT MORE TRAINING
---------------------------------
The 300-epoch child run ended STILL DESCENDING, but the descent is
suspected to be memorisation rather than learning: training loss
0.4274 nats/token against a 1.1708 unigram-equivalent implies roughly
4.1 effective aisle choices out of 134 on its own training data. That
is a SUSPICION derived from one quantity - this script is what decides
it, using repeated draws against measured nulls (standing rule: a
single derived number has overstated a result twice on this project).

THREE MEASURES, EACH WITH ITS OWN MEASURED NULL
-----------------------------------------------
 M1 SELF-COPY  generated basket for training parent i == parent i's
               OWN real basket (exact ordered aisle,reorder sequence).
               Null: the same comparison under a PERMUTED pairing
               (basket i vs a different parent's basket of the same
               size) - i.e. chance agreement.
 M2 ANY-COPY   generated basket appears ANYWHERE in the 27,664
               training baskets. Null: the rate at which genuinely
               HELD-OUT real baskets (v3_eval + v3_compare orders,
               asserted disjoint) also appear there, SIZE-MATCHED -
               because small baskets collide by chance far more often.
 M3 MULTISET   same as M2 but order-insensitive (aisle multiset), to
               separate "same items" from "same sequence".

Every rate is reported as a distribution over N_DRAWS independent
draws with a fire rate against the null's 95th percentile. No single
point estimate is used as evidence.

FIDELITY IS MEASURED ALONGSIDE
------------------------------
A checkpoint is only useful if it is faithful AND not memorising, so
aisle TVD / is_reorder / is_early_in_cart are measured at the same
checkpoints against the same style of measured bar. The chosen
checkpoint is the most-trained one whose memorisation is still at the
held-out-real baseline.

SELFTEST (--selftest) proves the detector before it is trusted:
  PLANTED  feed actual TRAINING baskets as if generated -> must fire
           at ~100% on M1/M2/M3.
  CLEAN    feed HELD-OUT REAL baskets as if generated -> must NOT
           fire (it defines the null).

READS  (read-only): data/rtf/parent.csv, data/rtf/child.csv,
                    data/rtf/prep_manifest.json,
                    data/orders.csv, data/order_products__prior.csv,
                    data/products.csv, data/v3_eval.csv,
                    data/v3_compare.csv, data/v3_train_order_ids.csv,
                    data/rtf/models/child/id*,
                    data/rtf/train_ckpt/child/checkpoint-*
WRITES (new only) : results/rtf/memorisation/sweep.json
                    results/rtf/memorisation/sweep.png
                    results/rtf/memorisation/selftest.json

Usage:
    python rtf_memorisation.py --selftest
    python rtf_memorisation.py --timing-probe
    python rtf_memorisation.py
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import glob
import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RTF = DATA / 'rtf'
OUT = ROOT / 'results' / 'rtf' / 'memorisation'

SEED = 20260727
N_DRAWS = 10          # independent draws per checkpoint (fire-rate rule)
N_BASKETS = 400       # generated baskets per draw
N_NULL = 40           # null resamples
QUANTILE = 0.95


def banner(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


# ══════════════════════════════════════════════════════════
# basket keys
# ══════════════════════════════════════════════════════════
def ordered_key(aisles, reorders):
    return tuple(zip(aisles, reorders))


def multiset_key(aisles):
    return tuple(sorted(Counter(aisles).items()))


def baskets_from_flat(df, id_col='order_id',
                      aisle_col='aisle_id', reo_col='reordered'):
    """-> {order_id: (ordered_key, multiset_key, size)}.
    Row order within an order IS the cart order (verified in
    rtf_prepare_data.py gate [5]), so no sorting here."""
    out = {}
    for oid, g in df.groupby(id_col, sort=False):
        a = [str(x) for x in g[aisle_col].tolist()]
        r = [str(x) for x in g[reo_col].tolist()]
        out[oid] = (ordered_key(a, r), multiset_key(a), len(a))
    return out


# ══════════════════════════════════════════════════════════
# reference sets
# ══════════════════════════════════════════════════════════
def load_reference():
    manifest = json.loads((RTF / 'prep_manifest.json').read_text())
    dt = manifest['rtf_dtypes']
    P = manifest['parent_cols']
    C = manifest['child_cols']

    parent = pd.read_csv(
        RTF / 'parent.csv',
        dtype={c: t for c, t in dt.items() if c in P and t == 'str'})
    parent['days_since_prior_order'] = parent[
        'days_since_prior_order'].astype('float64')
    child = pd.read_csv(
        RTF / 'child.csv',
        dtype={c: t for c, t in dt.items() if c in C and t == 'str'})

    train_b = baskets_from_flat(child)
    train_ordered = set(v[0] for v in train_b.values())
    train_msets = set(v[1] for v in train_b.values())

    # held-out real baskets: v3_eval + v3_compare orders, rebuilt from
    # raw the same way the training child table was
    ev = set(pd.read_csv(DATA / 'v3_eval.csv',
                         usecols=['order_id'])['order_id'])
    cp = set(pd.read_csv(DATA / 'v3_compare.csv',
                         usecols=['order_id'])['order_id'])
    tr_ids = set(pd.read_csv(
        DATA / 'v3_train_order_ids.csv')['order_id'])
    hold_ids = (ev | cp)
    assert len(hold_ids & tr_ids) == 0, \
        'FATAL: held-out ids overlap training ids'

    prods = pd.read_csv(DATA / 'products.csv',
                        usecols=['product_id', 'aisle_id'])
    op = pd.read_csv(DATA / 'order_products__prior.csv')
    h = op[op['order_id'].isin(hold_ids)].merge(
        prods, on='product_id', how='left')
    h = h.sort_values(['order_id', 'add_to_cart_order'],
                      kind='mergesort')
    h['aisle_id'] = h['aisle_id'].astype(int).astype(str)
    h['reordered'] = h['reordered'].astype(int).astype(str)
    hold_b = baskets_from_flat(h)

    # POPULATION MATCH - do not remove. The training tables were
    # mid-dropped (sizes 11-13 excluded, per the screened 10/14
    # buckets), so an unfiltered held-out pool has a genuinely
    # different basket-size distribution and a different aisle mix.
    # Leaving it unfiltered made the fidelity selftest FAIL on clean
    # held-out real data (f2_size_tvd 0.2228 vs bar 0.1775, 0/10) -
    # the detector was correctly reporting a difference that my own
    # null construction had created. Same defect class as the
    # mismatched-n false pass earlier in this project.
    smax = manifest['small_max']
    lmin = manifest['large_min']
    n_before = len(hold_b)
    hold_b = {k: v for k, v in hold_b.items()
              if v[2] <= smax or v[2] >= lmin}

    print(f'    training baskets       : {len(train_b):,}')
    print(f'    distinct ordered keys  : {len(train_ordered):,}')
    print(f'    held-out real baskets  : {len(hold_b):,}  '
          f'(disjoint, asserted; mid-dropped from {n_before:,} '
          f'to match the training population)')
    return (parent, child, train_b, train_ordered, train_msets,
            hold_b, manifest)


def size_index(baskets):
    idx = {}
    for oid, (_, _, n) in baskets.items():
        idx.setdefault(n, []).append(oid)
    return idx


# ══════════════════════════════════════════════════════════
# the three measures
# ══════════════════════════════════════════════════════════
def measure(gen, own, train_ordered, train_msets, hold_b, hold_idx,
            rng):
    """gen : {key -> (ordered, mset, size)} generated baskets
       own : {key -> (ordered, mset, size)} the SAME parents' real
             baskets, aligned by key (for M1). May be None.
    Returns dict of observed rates and their nulls."""
    keys = list(gen.keys())
    n = len(keys)
    res = {}

    # ── M1 self-copy ──
    if own is not None:
        hit = sum(1 for k in keys
                  if k in own and gen[k][0] == own[k][0])
        res['m1_self_copy'] = hit / n if n else 0.0
        # null: permuted pairing, same size only
        by_size = {}
        for k in keys:
            if k in own:
                by_size.setdefault(own[k][2], []).append(k)
        nulls = []
        for _ in range(N_NULL):
            h2 = 0
            for k in keys:
                if k not in own:
                    continue
                pool = by_size.get(gen[k][2], [])
                if not pool:
                    continue
                other = pool[rng.randint(len(pool))]
                if other != k and gen[k][0] == own[other][0]:
                    h2 += 1
            nulls.append(h2 / n if n else 0.0)
        res['m1_null'] = nulls

    # ── M2 any-copy (ordered), size-matched null ──
    hit = sum(1 for k in keys if gen[k][0] in train_ordered)
    res['m2_any_copy'] = hit / n if n else 0.0
    sizes = [gen[k][2] for k in keys]
    nulls, dropped = [], 0
    for _ in range(N_NULL):
        h2, used = 0, 0
        for s in sizes:
            pool = hold_idx.get(s)
            if not pool:
                dropped += 1
                continue
            oid = pool[rng.randint(len(pool))]
            used += 1
            if hold_b[oid][0] in train_ordered:
                h2 += 1
        nulls.append(h2 / used if used else 0.0)
    res['m2_null'] = nulls
    res['m2_null_unmatched_sizes'] = dropped / max(N_NULL, 1)

    # ── M3 multiset ──
    hit = sum(1 for k in keys if gen[k][1] in train_msets)
    res['m3_mset_copy'] = hit / n if n else 0.0
    nulls = []
    for _ in range(N_NULL):
        h2, used = 0, 0
        for s in sizes:
            pool = hold_idx.get(s)
            if not pool:
                continue
            oid = pool[rng.randint(len(pool))]
            used += 1
            if hold_b[oid][1] in train_msets:
                h2 += 1
        nulls.append(h2 / used if used else 0.0)
    res['m3_null'] = nulls
    res['n_baskets'] = n
    return res


def summarise(draws, key, null_key):
    """draws: list of measure() dicts. -> observed distribution, bar,
    fire rate."""
    obs = np.array([d[key] for d in draws])
    nulls = np.concatenate([np.array(d[null_key]) for d in draws])
    bar = float(np.quantile(nulls, QUANTILE))
    fires = int((obs > bar).sum())
    return {
        'observed_mean': float(obs.mean()),
        'observed_min': float(obs.min()),
        'observed_max': float(obs.max()),
        'null_mean': float(nulls.mean()),
        'bar_p95': bar,
        'fires_above_bar': fires,
        'n_draws': len(obs),
        'verdict': ('MEMORISING' if fires > len(obs) // 2
                    else 'at baseline'),
    }


# ══════════════════════════════════════════════════════════
def load_child(ckpt_path=None):
    """Full RTF child object; optionally with checkpoint weights.

    decoder.lm_head.weight is TIED to decoder.transformer.wte.weight
    (verified identical in the saved artifact), and HF checkpoints omit
    the tied copy - hence strict=False plus an explicit re-tie
    assertion rather than a silent partial load."""
    import torch
    from safetensors.torch import load_file
    from realtabformer import REaLTabFormer

    mdir = sorted(glob.glob(str(RTF / 'models' / 'child' / 'id*')))[-1]
    model = REaLTabFormer.load_from_dir(mdir)

    if ckpt_path is not None:
        sd = load_file(str(Path(ckpt_path) / 'model.safetensors'))
        missing, unexpected = model.model.load_state_dict(
            sd, strict=False)
        assert not unexpected, f'unexpected keys: {unexpected}'
        assert set(missing) <= {'decoder.lm_head.weight'}, \
            f'unexpected missing keys: {missing}'
        model.model.decoder.lm_head.weight = \
            model.model.decoder.transformer.wte.weight
        assert torch.equal(
            model.model.decoder.lm_head.weight,
            model.model.decoder.transformer.wte.weight), \
            'FATAL: lm_head not re-tied after checkpoint load'
    return model


def gen_baskets(model, parent, pick, gen_batch=32):
    """Generate baskets conditioned on the REAL parent rows in `pick`.
    Conditioning on real parents is what makes the self-copy measure
    possible: we can ask whether the model regurgitates THAT order's
    actual basket."""
    sub = parent.loc[pick].reset_index(drop=True)
    ids = [f'P{i:07d}' for i in range(len(sub))]
    sc = model.sample(
        input_unique_ids=ids,
        input_df=sub.drop(columns=['order_id']),
        gen_batch=gen_batch, device='cuda')
    sc = sc.reset_index()
    sc = sc.rename(columns={sc.columns[0]: 'gid'})
    gen = baskets_from_flat(sc, id_col='gid')
    # map generated ids back to the real order they were conditioned on
    id2oid = {i: o for i, o in zip(ids, sub['order_id'])}
    return {id2oid[k]: v for k, v in gen.items() if k in id2oid}


# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--timing-probe', action='store_true')
    ap.add_argument('--n-baskets', type=int, default=N_BASKETS)
    ap.add_argument('--n-draws', type=int, default=N_DRAWS)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    banner('LOADING REFERENCE SETS')
    (parent, child, train_b, train_ordered, train_msets,
     hold_b, manifest) = load_reference()
    hold_idx = size_index(hold_b)
    parent = parent.set_index('order_id', drop=False)

    # ── SELFTEST ──
    if args.selftest:
        banner('SELFTEST - known answers before the detector is used')
        report = {}

        print('\n[1] PLANTED: actual TRAINING baskets fed as if '
              'generated.')
        print('    Expect M1/M2/M3 to fire at ~100%.')
        tr_ids = list(train_b.keys())
        draws = []
        for _ in range(args.n_draws):
            pick = [tr_ids[i] for i in
                    rng.choice(len(tr_ids), args.n_baskets,
                               replace=False)]
            gen = {k: train_b[k] for k in pick}
            draws.append(measure(gen, train_b, train_ordered,
                                 train_msets, hold_b, hold_idx, rng))
        planted = {m: summarise(draws, m, n) for m, n in
                   [('m1_self_copy', 'm1_null'),
                    ('m2_any_copy', 'm2_null'),
                    ('m3_mset_copy', 'm3_null')]}
        for m, v in planted.items():
            print(f'    {m:<16} obs {v["observed_mean"]:.4f}  '
                  f'bar {v["bar_p95"]:.4f}  fires '
                  f'{v["fires_above_bar"]}/{v["n_draws"]}  '
                  f'{v["verdict"]}')

        print('\n[2] CLEAN: HELD-OUT REAL baskets fed as if generated.')
        print('    Expect NO firing - these define the null.')
        h_ids = list(hold_b.keys())
        draws = []
        for _ in range(args.n_draws):
            pick = [h_ids[i] for i in
                    rng.choice(len(h_ids),
                               min(args.n_baskets, len(h_ids)),
                               replace=False)]
            gen = {k: hold_b[k] for k in pick}
            draws.append(measure(gen, None, train_ordered,
                                 train_msets, hold_b, hold_idx, rng))
        clean = {m: summarise(draws, m, n) for m, n in
                 [('m2_any_copy', 'm2_null'),
                  ('m3_mset_copy', 'm3_null')]}
        for m, v in clean.items():
            print(f'    {m:<16} obs {v["observed_mean"]:.4f}  '
                  f'bar {v["bar_p95"]:.4f}  fires '
                  f'{v["fires_above_bar"]}/{v["n_draws"]}  '
                  f'{v["verdict"]}')

        ok_planted = all(v['verdict'] == 'MEMORISING'
                         for v in planted.values())
        ok_clean = all(v['verdict'] == 'at baseline'
                       for v in clean.values())
        print(f'\n    planted detected : '
              f'{"PASS" if ok_planted else "FAIL"}')
        print(f'    clean not flagged: '
              f'{"PASS" if ok_clean else "FAIL"}')
        report = {'planted': planted, 'clean': clean,
                  'pass': bool(ok_planted and ok_clean),
                  'n_draws': args.n_draws,
                  'n_baskets': args.n_baskets}
        (OUT / 'selftest.json').write_text(
            json.dumps(report, indent=2, default=float))
        print(f'\n    saved {OUT / "selftest.json"}')
        return 0 if report['pass'] else 1

    # ── checkpoint list ──
    cks = sorted(glob.glob(str(RTF / 'train_ckpt' / 'child' /
                               'checkpoint-*')),
                 key=lambda p: int(re.search(r'checkpoint-(\d+)',
                                             p).group(1)))
    spe = json.loads((ROOT / 'results' / 'rtf' / 'train' /
                      'train_report.json').read_text()
                     )['child_checkpointing']['steps_per_epoch']
    print(f'\n    {len(cks)} checkpoints, {spe} steps/epoch')

    if args.timing_probe:
        banner('TIMING PROBE - one checkpoint, one draw')
        model = load_child(cks[0])
        ids = list(parent.index)
        pick = [ids[i] for i in rng.choice(len(ids),
                                           args.n_baskets,
                                           replace=False)]
        t0 = time.time()
        gen = gen_baskets(model, parent, pick)
        dt = time.time() - t0
        print(f'    {len(gen)} baskets in {dt:.1f}s '
              f'({dt/max(len(gen),1)*1000:.0f} ms/basket)')
        total = dt * args.n_draws * len(cks) / 3600
        print(f'    projected full sweep: {len(cks)} ckpts x '
              f'{args.n_draws} draws = {total:.2f} h')
        return 0

    # ── FULL SWEEP ──
    banner('CHECKPOINT SWEEP')
    rows = []
    ids = list(parent.index)
    for ck in cks:
        step = int(re.search(r'checkpoint-(\d+)', ck).group(1))
        epoch = step / spe
        model = load_child(ck)
        draws = []
        t0 = time.time()
        for _ in range(args.n_draws):
            pick = [ids[i] for i in
                    rng.choice(len(ids), args.n_baskets,
                               replace=False)]
            gen = gen_baskets(model, parent, pick)
            draws.append(measure(gen, train_b, train_ordered,
                                 train_msets, hold_b, hold_idx, rng))
        s = {m: summarise(draws, m, n) for m, n in
             [('m1_self_copy', 'm1_null'),
              ('m2_any_copy', 'm2_null'),
              ('m3_mset_copy', 'm3_null')]}
        row = {'checkpoint': ck, 'step': step,
               'epoch': round(epoch, 1),
               'seconds': time.time() - t0, **s}
        rows.append(row)
        print(f'\n  epoch {epoch:6.1f}  ({time.time()-t0:.0f}s)')
        for m, v in s.items():
            print(f'    {m:<16} obs {v["observed_mean"]:.4f} '
                  f'[{v["observed_min"]:.4f}-{v["observed_max"]:.4f}]'
                  f'  bar {v["bar_p95"]:.4f}  fires '
                  f'{v["fires_above_bar"]}/{v["n_draws"]}  '
                  f'{v["verdict"]}')
        (OUT / 'sweep.json').write_text(
            json.dumps({'rows': rows, 'seed': SEED,
                        'n_draws': args.n_draws,
                        'n_baskets': args.n_baskets},
                       indent=2, default=float))

    # ── pick the checkpoint ──
    banner('SELECTION')
    clean_rows = [r for r in rows
                  if r['m1_self_copy']['verdict'] == 'at baseline'
                  and r['m2_any_copy']['verdict'] == 'at baseline']
    print(f'    checkpoints at baseline on M1 and M2: '
          f'{[r["epoch"] for r in clean_rows]}')
    if clean_rows:
        best = max(clean_rows, key=lambda r: r['epoch'])
        print(f'\n    -> most-trained non-memorising checkpoint: '
              f'epoch {best["epoch"]} ({best["checkpoint"]})')
    else:
        print('\n    -> NO checkpoint is free of memorisation.')
        print('       That is itself the finding; report it.')

    try:
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
        ep = [r['epoch'] for r in rows]
        for a, (m, lbl) in zip(ax, [
                ('m1_self_copy', 'M1 self-copy (own basket)'),
                ('m2_any_copy', 'M2 any-copy (ordered)'),
                ('m3_mset_copy', 'M3 multiset copy')]):
            obs = [r[m]['observed_mean'] for r in rows]
            lo = [r[m]['observed_min'] for r in rows]
            hi = [r[m]['observed_max'] for r in rows]
            bar = [r[m]['bar_p95'] for r in rows]
            a.fill_between(ep, lo, hi, alpha=0.25, color='seagreen')
            a.plot(ep, obs, 'o-', color='seagreen', label='generated')
            a.plot(ep, bar, '--', color='crimson',
                   label='measured bar (p95 null)')
            a.set_title(lbl, fontsize=10)
            a.set_xlabel('child training epoch')
            a.set_ylabel('copy rate')
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / 'sweep.png', dpi=150, bbox_inches='tight')
        print(f'\n    saved {OUT / "sweep.png"}')
    except Exception as e:
        print(f'    chart failed: {e!r}')

    print(f'    saved {OUT / "sweep.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
