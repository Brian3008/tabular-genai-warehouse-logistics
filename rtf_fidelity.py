"""
rtf_fidelity.py
===============
The other half of the checkpoint sweep. rtf_memorisation.py found that
the child model copies training baskets from roughly epoch 50-75
onward; this script asks whether the checkpoints that are STILL CLEAN
are actually any good.

Together the two produce the fidelity-vs-memorisation tradeoff curve,
which is what decides whether REaLTabFormer can be used here at all.
If the non-memorising checkpoints are also unfaithful, then there is
no usable operating point - and that is a finding about the
architecture, not a tuning failure.

METRICS (all with MEASURED bars, all as fire rates over draws)
--------------------------------------------------------------
 F1 aisle TVD        item-level aisle_id distribution vs real
 F2 basket-size TVD  generated basket length distribution vs real
 F3 is_reorder       |generated rate - real rate|
 F4 is_early_in_cart |generated rate - real rate|   (position <= 3)

Every bar comes from disjoint half-splits of the REAL data at the SAME
sample size as the observed comparison (TVD is size-dependent - a
mismatched n produced a false pass on this project once already).
A metric PASSES when the observed value is below the bar on a MAJORITY
of independent draws.

SELFTEST (--selftest):
  CLEAN    held-out real baskets scored as if generated -> must PASS
  PLANTED  aisles resampled uniformly, reorder flipped -> must FAIL

READS  (read-only): data/rtf/parent.csv, data/rtf/child.csv,
                    data/rtf/prep_manifest.json,
                    data/rtf/models/child/id*,
                    data/rtf/train_ckpt/child/checkpoint-*
WRITES (new only) : results/rtf/memorisation/fidelity.json
                    results/rtf/memorisation/fidelity.png
                    results/rtf/memorisation/fidelity_selftest.json

Usage:
    python rtf_fidelity.py --selftest
    python rtf_fidelity.py --epochs 25 50 75 100 200 300
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rtf_memorisation import (RTF, ROOT, gen_baskets, load_child,
                              banner)

OUT = ROOT / 'results' / 'rtf' / 'memorisation'
SEED = 20260727
N_DRAWS = 10
N_BASKETS = 400
N_NULL = 40
QUANTILE = 0.95


def tvd_counts(a, b):
    idx = sorted(set(a.index) | set(b.index), key=str)
    a = a.reindex(idx, fill_value=0.0)
    b = b.reindex(idx, fill_value=0.0)
    return float(0.5 * np.abs(a - b).sum())


def dist(vals):
    return pd.Series(vals).value_counts(normalize=True)


def flatten_gen(gen):
    """gen: {oid: (ordered_key, mset_key, size)}
    ordered_key is a tuple of (aisle, reordered) in cart order, so the
    item index within the tuple IS add_to_cart_order - the same rule
    rtf_prepare_data.flatten() uses, verified there against v3_train."""
    aisles, reorders, early, sizes = [], [], [], []
    for _, (ok, _, n) in gen.items():
        sizes.append(n)
        for pos, (a, r) in enumerate(ok, start=1):
            aisles.append(a)
            reorders.append(int(r))
            early.append(1 if pos <= 3 else 0)
    return (np.array(aisles), np.array(reorders),
            np.array(early), np.array(sizes))


def real_pools(child, train_b):
    aisles = child['aisle_id'].astype(str).values
    reorders = child['reordered'].astype(int).values
    pos = child.groupby('order_id', sort=False).cumcount().values + 1
    early = (pos <= 3).astype(int)
    sizes = np.array([v[2] for v in train_b.values()])
    return aisles, reorders, early, sizes


def measure_fidelity(gen, real, rng):
    ga, gr, ge, gs = gen
    ra, rr, re_, rs = real
    n_item, n_bask = len(ga), len(gs)
    out = {}

    # F1 aisle TVD, matched at n_item per side
    obs = tvd_counts(dist(ga),
                     dist(ra[rng.choice(len(ra), n_item,
                                        replace=False)]))
    nulls = []
    for _ in range(N_NULL):
        p = rng.permutation(len(ra))
        nulls.append(tvd_counts(dist(ra[p[:n_item]]),
                                dist(ra[p[n_item:2 * n_item]])))
    out['f1_aisle_tvd'] = (obs, nulls)

    # F2 basket-size TVD, matched at n_bask per side
    obs = tvd_counts(dist(gs),
                     dist(rs[rng.choice(len(rs), n_bask,
                                        replace=False)]))
    nulls = []
    for _ in range(N_NULL):
        p = rng.permutation(len(rs))
        nulls.append(tvd_counts(dist(rs[p[:n_bask]]),
                                dist(rs[p[n_bask:2 * n_bask]])))
    out['f2_size_tvd'] = (obs, nulls)

    # F3/F4 rate gaps, matched at n_item per side
    for key, gv, rv in [('f3_is_reorder', gr, rr),
                        ('f4_is_early_in_cart', ge, re_)]:
        obs = abs(gv.mean() -
                  rv[rng.choice(len(rv), n_item,
                                replace=False)].mean())
        nulls = []
        for _ in range(N_NULL):
            p = rng.permutation(len(rv))
            nulls.append(abs(rv[p[:n_item]].mean() -
                             rv[p[n_item:2 * n_item]].mean()))
        out[key] = (obs, nulls)
    return out


def summarise(draws, key):
    obs = np.array([d[key][0] for d in draws])
    nulls = np.concatenate([np.array(d[key][1]) for d in draws])
    bar = float(np.quantile(nulls, QUANTILE))
    below = int((obs < bar).sum())
    return {'observed_mean': float(obs.mean()),
            'observed_min': float(obs.min()),
            'observed_max': float(obs.max()),
            'null_mean': float(nulls.mean()),
            'bar_p95': bar,
            'fires_below_bar': below,
            'n_draws': len(obs),
            'verdict': ('MATCHES REAL' if below > len(obs) // 2
                        else 'DIFFERS')}


KEYS = ['f1_aisle_tvd', 'f2_size_tvd', 'f3_is_reorder',
        'f4_is_early_in_cart']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--epochs', type=int, nargs='+',
                    default=[25, 50, 75, 100, 200, 300])
    ap.add_argument('--n-baskets', type=int, default=N_BASKETS)
    ap.add_argument('--n-draws', type=int, default=N_DRAWS)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    from rtf_memorisation import load_reference, baskets_from_flat
    banner('LOADING REFERENCE SETS')
    (parent, child, train_b, train_ordered, train_msets,
     hold_b, manifest) = load_reference()
    real = real_pools(child, train_b)
    parent = parent.set_index('order_id', drop=False)

    if args.selftest:
        banner('SELFTEST')
        report = {}

        print('\n[1] CLEAN: held-out REAL baskets scored as generated.')
        draws = []
        h_ids = list(hold_b.keys())
        for _ in range(args.n_draws):
            pick = [h_ids[i] for i in rng.choice(
                len(h_ids), args.n_baskets, replace=False)]
            draws.append(measure_fidelity(
                flatten_gen({k: hold_b[k] for k in pick}), real, rng))
        clean = {k: summarise(draws, k) for k in KEYS}
        for k, v in clean.items():
            print(f'    {k:<22} obs {v["observed_mean"]:.4f}  bar '
                  f'{v["bar_p95"]:.4f}  {v["fires_below_bar"]}/'
                  f'{v["n_draws"]}  {v["verdict"]}')

        print('\n[2] PLANTED: aisles resampled uniformly, reorder '
              'flipped.')
        draws = []
        uniq = np.unique(real[0])
        for _ in range(args.n_draws):
            pick = [h_ids[i] for i in rng.choice(
                len(h_ids), args.n_baskets, replace=False)]
            ga, gr, ge, gs = flatten_gen({k: hold_b[k] for k in pick})
            ga = uniq[rng.randint(0, len(uniq), len(ga))]
            gr = 1 - gr
            draws.append(measure_fidelity((ga, gr, ge, gs), real, rng))
        planted = {k: summarise(draws, k) for k in KEYS}
        for k, v in planted.items():
            print(f'    {k:<22} obs {v["observed_mean"]:.4f}  bar '
                  f'{v["bar_p95"]:.4f}  {v["fires_below_bar"]}/'
                  f'{v["n_draws"]}  {v["verdict"]}')

        ok_clean = all(clean[k]['verdict'] == 'MATCHES REAL'
                       for k in KEYS)
        ok_planted = all(planted[k]['verdict'] == 'DIFFERS'
                         for k in ['f1_aisle_tvd', 'f3_is_reorder'])
        print(f'\n    clean passes  : '
              f'{"PASS" if ok_clean else "FAIL"}')
        print(f'    planted caught: '
              f'{"PASS" if ok_planted else "FAIL"}')
        report = {'clean': clean, 'planted': planted,
                  'pass': bool(ok_clean and ok_planted)}
        (OUT / 'fidelity_selftest.json').write_text(
            json.dumps(report, indent=2, default=float))
        print(f'\n    saved {OUT / "fidelity_selftest.json"}')
        return 0 if report['pass'] else 1

    spe = json.loads((ROOT / 'results' / 'rtf' / 'train' /
                      'train_report.json').read_text()
                     )['child_checkpointing']['steps_per_epoch']
    all_ck = {int(re.search(r'checkpoint-(\d+)', p).group(1)) // spe: p
              for p in glob.glob(str(RTF / 'train_ckpt' / 'child' /
                                     'checkpoint-*'))}

    banner('FIDELITY BY CHECKPOINT')
    rows = []
    ids = list(parent.index)
    for ep in args.epochs:
        if ep not in all_ck:
            print(f'    epoch {ep}: no checkpoint, skipping')
            continue
        model = load_child(all_ck[ep])
        draws = []
        for _ in range(args.n_draws):
            pick = [ids[i] for i in rng.choice(
                len(ids), args.n_baskets, replace=False)]
            gen = gen_baskets(model, parent, pick)
            draws.append(measure_fidelity(flatten_gen(gen), real, rng))
        s = {k: summarise(draws, k) for k in KEYS}
        n_ok = sum(1 for k in KEYS if s[k]['verdict'] == 'MATCHES REAL')
        rows.append({'epoch': ep, 'n_matching': n_ok, **s})
        print(f'\n  epoch {ep}   ({n_ok}/4 match real)')
        for k in KEYS:
            v = s[k]
            print(f'    {k:<22} obs {v["observed_mean"]:.4f} '
                  f'[{v["observed_min"]:.4f}-{v["observed_max"]:.4f}]'
                  f'  bar {v["bar_p95"]:.4f}  '
                  f'{v["fires_below_bar"]}/{v["n_draws"]}  '
                  f'{v["verdict"]}')
        (OUT / 'fidelity.json').write_text(
            json.dumps({'rows': rows, 'seed': SEED,
                        'n_draws': args.n_draws,
                        'n_baskets': args.n_baskets},
                       indent=2, default=float))

    try:
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
        ep = [r['epoch'] for r in rows]
        for a, k in zip(ax, KEYS):
            a.fill_between(ep, [r[k]['observed_min'] for r in rows],
                           [r[k]['observed_max'] for r in rows],
                           alpha=0.25, color='steelblue')
            a.plot(ep, [r[k]['observed_mean'] for r in rows], 'o-',
                   color='steelblue', label='generated')
            a.plot(ep, [r[k]['bar_p95'] for r in rows], '--',
                   color='crimson', label='measured bar')
            a.set_title(k, fontsize=10)
            a.set_xlabel('child training epoch')
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / 'fidelity.png', dpi=150,
                    bbox_inches='tight')
        print(f'\n    saved {OUT / "fidelity.png"}')
    except Exception as e:
        print(f'    chart failed: {e!r}')
    print(f'    saved {OUT / "fidelity.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
