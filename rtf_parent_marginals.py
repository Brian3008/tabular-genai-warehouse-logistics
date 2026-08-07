"""
rtf_parent_marginals.py
=======================
Answers one question, with a MEASURED bar: at the epoch count RTF's
Q_delta detector tolerates (~30), does the PARENT model actually
reproduce the order-level marginals?

WHY THIS MATTERS (the architecture tension)
-------------------------------------------
freeze_parent_model=True is RTF's default, so the child's ENCODER is
the frozen parent GPT-2. The parent therefore has two jobs at once:
  1. generate order-level columns (dow, hour, days_since_prior_order,
     order_size_grp) faithfully, and
  2. serve as a frozen representation the child conditions its whole
     basket on.
Q_delta caps job 1's training on MEMORISATION grounds. If the parent
is still under-fit on the marginals at that cap, the two requirements
are in direct conflict and that conflict is a finding about the
architecture, not something to tune away.

MEASURED BAR, NOT A GUESSED ONE
-------------------------------
TVD is sample-size dependent, so every comparison is made at the SAME
size m. With n real parent rows:
    m        = n // 2
    observed = TVD(real_A(m), synth(m))          over 10 fresh draws
    null     = TVD(real_A(m), real_B(m))         over N_NULL disjoint
                                                 half-splits of the
                                                 REAL table only
    bar      = 95th percentile of the null
A column "passes" only if the observed TVD fires BELOW the bar on a
MAJORITY of draws. Fire rates are reported, never a single point
estimate (standing rule).

SELFTEST (--selftest) verifies the detector on known answers before it
is trusted: two disjoint halves of the real data must report NO
difference, and a deliberately corrupted copy must be detected.

READS  (read-only): data/rtf/parent.csv, data/rtf/prep_manifest.json,
                    data/rtf/probe_ckpt/parent_model/id*  (probe model)
WRITES (new only) : results/rtf/probe/parent_marginals.json
                    results/rtf/probe/parent_marginals.png

Usage:
    python rtf_parent_marginals.py --selftest
    python rtf_parent_marginals.py
    python rtf_parent_marginals.py --model-dir <path to id*>
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'rtf'
OUT = ROOT / 'results' / 'rtf' / 'probe'

SEED = 20260726
N_NULL = 40          # null half-splits
N_OBS_DRAWS = 10     # observed draws (fire-rate rule)
QUANTILE = 0.95

COLS = ['order_dow', 'order_hour_of_day',
        'days_since_prior_order', 'order_size_grp']


def tvd(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance between two categorical samples."""
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    idx = sorted(set(pa.index) | set(pb.index), key=str)
    pa = pa.reindex(idx, fill_value=0.0)
    pb = pb.reindex(idx, fill_value=0.0)
    return float(0.5 * np.abs(pa - pb).sum())


def as_cat(s: pd.Series, col: str) -> pd.Series:
    """Common discretisation for both sides. dspo is integer-valued
    0-30 in this data, so rounding to int is lossless for the real
    side and bins the generated side identically."""
    if col == 'days_since_prior_order':
        return pd.to_numeric(s, errors='coerce').fillna(0)\
            .round().astype(int).astype(str)
    return s.astype(str)


def null_bar(real: pd.DataFrame, col: str, m: int, rng):
    """Null TVD distribution from disjoint half-splits of the REAL
    table only. Same m on both sides as the observed comparison."""
    vals = []
    n = len(real)
    for _ in range(N_NULL):
        perm = rng.permutation(n)
        A = real.iloc[perm[:m]]
        B = real.iloc[perm[m:2 * m]]
        vals.append(tvd(as_cat(A[col], col), as_cat(B[col], col)))
    return np.array(vals)


def observed_draws(real: pd.DataFrame, synth: pd.DataFrame,
                   col: str, m: int, rng):
    """Observed TVD over independent draws, matched at m per side."""
    vals = []
    for _ in range(N_OBS_DRAWS):
        A = real.iloc[rng.permutation(len(real))[:m]]
        S = synth.iloc[rng.permutation(len(synth))[:m]]
        vals.append(tvd(as_cat(A[col], col), as_cat(S[col], col)))
    return np.array(vals)


def assess(real, synth, label, rng):
    m = min(len(real) // 2, len(synth))
    rows = []
    print(f'\n    comparison size m = {m:,} per side '
          f'(real {len(real):,}, synth {len(synth):,})')
    print(f'\n    {"column":<26}{"real":>9}{"synth":>9}'
          f'{"obsTVD":>9}{"bar":>8}{"fires<bar":>11}')
    print('    ' + '-' * 72)
    for col in COLS:
        nb = null_bar(real, col, m, rng)
        bar = float(np.quantile(nb, QUANTILE))
        ob = observed_draws(real, synth, col, m, rng)
        below = int((ob < bar).sum())

        if col == 'order_size_grp':
            r_val = float((real[col] == 'small').mean())
            s_val = float((synth[col] == 'small').mean())
        else:
            r_val = float(pd.to_numeric(
                real[col], errors='coerce').mean())
            s_val = float(pd.to_numeric(
                synth[col], errors='coerce').mean())
        r_txt = f'{r_val:.3f}'
        s_txt = f'{s_val:.3f}'

        print(f'    {col:<26}{r_txt:>9}{s_txt:>9}'
              f'{ob.mean():>9.4f}{bar:>8.4f}'
              f'{below:>8}/{N_OBS_DRAWS}')
        rows.append({
            'column': col, 'label': label,
            'real_summary': r_val, 'synth_summary': s_val,
            'observed_tvd_mean': float(ob.mean()),
            'observed_tvd_min': float(ob.min()),
            'observed_tvd_max': float(ob.max()),
            'null_mean': float(nb.mean()),
            'bar_p95': bar,
            'fires_below_bar': below,
            'n_draws': N_OBS_DRAWS,
            'verdict': ('MATCHES REAL'
                        if below > N_OBS_DRAWS // 2 else 'DIFFERS'),
            'm': int(m),
        })
    return rows


def selftest(real, rng):
    print('=' * 70)
    print('SELFTEST - known answers before the detector is trusted')
    print('=' * 70)
    n = len(real)
    perm = rng.permutation(n)
    half = n // 2
    A = real.iloc[perm[:half]].reset_index(drop=True)
    B = real.iloc[perm[half:2 * half]].reset_index(drop=True)

    print('\n[1] CLEAN: two disjoint halves of the REAL table.')
    print('    Expect every column to report MATCHES REAL.')
    clean = assess(A, B, 'selftest_clean', rng)

    print('\n[2] PLANTED FAULT: hours shifted +4, dow collapsed to')
    print('    two values, dspo halved. Expect DIFFERS.')
    bad = B.copy()
    bad['order_hour_of_day'] = (
        (pd.to_numeric(bad['order_hour_of_day']) + 4) % 24).astype(str)
    bad['order_dow'] = np.where(
        np.arange(len(bad)) % 2 == 0, '0', '3')
    bad['days_since_prior_order'] = (
        pd.to_numeric(bad['days_since_prior_order']) / 2)
    planted = assess(A, bad, 'selftest_planted', rng)

    ok_clean = all(r['verdict'] == 'MATCHES REAL' for r in clean)
    detected = sum(1 for r in planted
                   if r['column'] != 'order_size_grp'
                   and r['verdict'] == 'DIFFERS')
    print(f'\n    clean: all columns match      -> '
          f'{"PASS" if ok_clean else "FAIL"}')
    print(f'    planted: {detected}/3 corrupted cols detected -> '
          f'{"PASS" if detected == 3 else "FAIL"}')
    return ok_clean and detected == 3, clean, planted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--n-samples', type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    manifest = json.loads((DATA / 'prep_manifest.json').read_text())
    dtypes = manifest['rtf_dtypes']
    P_COLS = manifest['parent_cols']
    real = pd.read_csv(
        DATA / 'parent.csv',
        dtype={c: t for c, t in dtypes.items()
               if c in P_COLS and t == 'str'})
    real['days_since_prior_order'] = real[
        'days_since_prior_order'].astype('float64')
    real = real[P_COLS]
    print(f'real parent rows: {len(real):,}')

    report = {'seed': SEED, 'n_null': N_NULL,
              'n_obs_draws': N_OBS_DRAWS, 'quantile': QUANTILE,
              'real_rows': int(len(real))}

    if args.selftest:
        ok, clean, planted = selftest(real, rng)
        report['selftest'] = {'pass': ok, 'clean': clean,
                              'planted': planted}
        (OUT / 'parent_marginals_selftest.json').write_text(
            json.dumps(report, indent=2, default=float))
        print(f'\nsaved {OUT / "parent_marginals_selftest.json"}')
        return 0 if ok else 1

    # ── locate the trained parent model ──
    if args.model_dir:
        mdir = Path(args.model_dir)
    else:
        cands = sorted((DATA / 'probe_ckpt' / 'parent_model')
                       .glob('id*'))
        if not cands:
            raise SystemExit(
                'No parent model found. Pass --model-dir.')
        mdir = cands[-1]
    print(f'parent model: {mdir}')

    from realtabformer import REaLTabFormer
    model = REaLTabFormer.load_from_dir(str(mdir))
    n_gen = args.n_samples or len(real)
    print(f'generating {n_gen:,} parent rows ...')
    synth = model.sample(n_samples=n_gen, gen_batch=512,
                         device='cuda')
    synth = synth.reset_index(drop=True)
    print(f'generated {len(synth):,} rows, columns '
          f'{list(synth.columns)}')

    for c in COLS:
        assert c in synth.columns, f'FATAL: {c} missing from output'

    print('\n' + '=' * 70)
    print(f'PARENT MARGINALS at the probe checkpoint '
          f'({mdir.name[:12]}...)')
    print('=' * 70)
    rows = assess(real, synth, 'parent_probe', rng)
    report['results'] = rows
    report['model_dir'] = str(mdir)
    report['n_generated'] = int(len(synth))

    n_pass = sum(1 for r in rows if r['verdict'] == 'MATCHES REAL')
    print(f'\n    {n_pass}/{len(rows)} order-level columns '
          f'indistinguishable from real at the measured bar')
    report['n_columns_matching'] = n_pass

    # basket-size mix is the conditioning axis - call it out
    ss = (synth['order_size_grp'] == 'small').mean()
    rs = (real['order_size_grp'] == 'small').mean()
    print(f'\n    order_size_grp small share: real {rs:.4f}  '
            f'synth {ss:.4f}  (delta {ss-rs:+.4f})')
    print('    This is the CONDITIONING AXIS. If the parent gets the')
    print('    small/large mix wrong, every downstream basket-size')
    print('    comparison loses power (bench contract rule 3).')

    try:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        for a, col in zip(ax, COLS):
            r = as_cat(real[col], col).value_counts(normalize=True)
            s = as_cat(synth[col], col).value_counts(normalize=True)
            idx = sorted(set(r.index) | set(s.index),
                         key=lambda x: (len(x), x))
            r = r.reindex(idx, fill_value=0)
            s = s.reindex(idx, fill_value=0)
            x = np.arange(len(idx))
            a.bar(x - 0.2, r.values, 0.4, label='real',
                  color='steelblue')
            a.bar(x + 0.2, s.values, 0.4, label='synth',
                  color='seagreen')
            a.set_title(col, fontsize=10)
            a.set_xticks(x)
            a.set_xticklabels(idx, rotation=90, fontsize=6)
            a.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT / 'parent_marginals.png', dpi=150,
                    bbox_inches='tight')
        print(f'\n    saved {OUT / "parent_marginals.png"}')
    except Exception as e:
        print(f'    chart failed: {e!r}')

    (OUT / 'parent_marginals.json').write_text(
        json.dumps(report, indent=2, default=float))
    print(f'    saved {OUT / "parent_marginals.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
