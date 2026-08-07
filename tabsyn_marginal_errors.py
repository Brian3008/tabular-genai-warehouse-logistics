"""tabsyn_marginal_errors.py - marginal errors for BOTH generators on ONE
declared basis, with a measured bar and a repeated-draw fire rate.

WHY THIS SCRIPT EXISTS (audit finding, Jul 27 2026)
---------------------------------------------------
The marginal-error row of results/tabsyn/comparison_ctgan_vs_tabsyn.md
quoted CTGAN 4.8%/3.0% and TabSyn 0.9%/1.5%. Those two columns were on
DIFFERENT bases (CTGAN vs the full fitted training population; TabSyn vs
an n=9000 seeded subsample of v3_eval) and NEITHER was written to a file
by any script - they existed only in console output. Same defect class as
the DCR n=4000/n=5000 trap (project notes). This script puts both generators
on one basis and saves the numbers.

DESIGN (project standing rules)
-------------------------------
Rule 1 - thresholds MEASURED, never guessed: the bar is the 95th
  percentile of |rate_a - rate_b| over 40 disjoint matched half-splits of
  the REAL population. Nothing is hard-coded.
Rule 2 - detector verified against a known-answer fixture: --selftest
  runs clean data (must fire 0%) and a planted +5pp shift (must fire
  100%) before any real number is trusted.
Rule 3 - sample sizes MATCH: one n_side is used for every side of every
  comparison, in the null AND in the observed draws. A marginal rate is
  far less size-sensitive than TVD/Gini, but the project has been burned
  by mismatched n once already, so it is matched here too.
Rule 5 - fire rate is the primary evidence: 10 independent observed
  draws, majority vote. The deterministic full-population point estimate
  is reported ALONGSIDE its fire rate, never instead of it.

BASIS (primary): the training population both generators were fitted to -
  data/v3_train.csv with order_size_grp RECOMPUTED at the screened 10/14
  buckets and mid dropped (272,820 rows). This is the basis CTGAN's
  recorded 4.8%/3.0% uses, so the known-answer gate can check it.
BASIS (secondary): held-out data/v3_eval.csv, same 10/14 mid-drop so the
  real side covers the same population the generators model (project notes
  caveat 3). Reported for transparency; conclusions agree.

KNOWN-ANSWER GATE: the full-population CTGAN errors must reproduce the
  recorded 0.048 / 0.030 (v3_training_report.json / v3_diagnosis.json) to
  <= 0.0005, or the script exits non-zero and writes nothing.

READS (read-only): data/v3_train.csv, data/v3_eval.csv,
                   data/synthetic_v3.csv, data/tabsyn/synthetic_tabsyn.csv
WRITES: <--out-dir>/marginal_errors.json  (or marginal_errors_selftest.json)

Usage:
  python tabsyn_marginal_errors.py --selftest --out-dir results/tabsyn
  python tabsyn_marginal_errors.py --out-dir results/tabsyn
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in (
        'utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
SMALL_MAX = 10
LARGE_MIN = 14
COND_COL = 'order_size_grp'
MARGINALS = ['is_reorder', 'is_early_in_cart']

N_NULL_DRAWS = 40
N_TEST_DRAWS = 10
PERCENTILE = 95
N_SIDE_CAP = 50_000

# recorded CTGAN figures the gate must reproduce
RECORDED_CTGAN = {'is_reorder': 0.048, 'is_early_in_cart': 0.030}
GATE_TOL = 0.0005


def fitted_population(df):
    """Recompute order_size_grp at the SCREENED 10/14 buckets and drop
    mid - never the stored 33/67 column (project-notes caveat 4)."""
    d = df.copy()
    n_items = d['order_id'].map(d.groupby('order_id').size())
    grp = np.where(n_items <= SMALL_MAX, 'small',
                   np.where(n_items >= LARGE_MIN, 'large', 'mid'))
    d[COND_COL] = grp
    return d[d[COND_COL] != 'mid'].reset_index(drop=True)


def measure_bar(real, n_side, rs):
    """Noise floor: |rate_a - rate_b| over disjoint matched half-splits of
    the REAL data. Two halves of the same population differ only by
    sampling noise, so this is what 'no error' actually looks like."""
    out = {c: [] for c in MARGINALS}
    n = len(real)
    for _ in range(N_NULL_DRAWS):
        idx = rs.permutation(n)
        a = real.iloc[idx[:n_side]]
        b = real.iloc[idx[n_side:2 * n_side]]
        for c in MARGINALS:
            out[c].append(abs(a[c].mean() - b[c].mean()))
    return ({c: float(np.percentile(out[c], PERCENTILE)) for c in MARGINALS},
            {c: float(np.mean(out[c])) for c in MARGINALS})


def observed_draws(real, synth, n_side, rs):
    """|rate_synth - rate_real| over N_TEST_DRAWS independent draws, both
    sides at the SAME n_side used to build the bar."""
    out = {c: [] for c in MARGINALS}
    for _ in range(N_TEST_DRAWS):
        r = real.iloc[rs.choice(len(real), n_side, replace=False)]
        s = synth.iloc[rs.choice(len(synth), n_side, replace=False)]
        for c in MARGINALS:
            out[c].append(abs(s[c].mean() - r[c].mean()))
    return {c: [float(v) for v in out[c]] for c in MARGINALS}


def score(real, synth, label):
    rs = np.random.RandomState(SEED)
    n_side = min(len(real) // 2, len(synth), N_SIDE_CAP)
    bar, bar_mean = measure_bar(real, n_side, rs)
    draws = observed_draws(real, synth, n_side, rs)

    res = {'n_side': int(n_side), 'n_real_pop': int(len(real)),
           'n_synth_pop': int(len(synth)), 'bar': bar, 'bar_mean': bar_mean,
           'per_column': {}}
    for c in MARGINALS:
        # deterministic full-population point estimate (the recorded basis)
        pt = abs(synth[c].mean() - real[c].mean())
        fired = [d > bar[c] for d in draws[c]]
        res['per_column'][c] = {
            'real_rate_full': float(real[c].mean()),
            'synth_rate_full': float(synth[c].mean()),
            'point_estimate': float(pt),
            'observed_draws': draws[c],
            'observed_mean': float(np.mean(draws[c])),
            'fire_rate': float(np.mean(fired)),
            'fails': bool(np.mean(fired) > 0.5),
        }
        print(f"    {label:7s} {c:20s} point {pt*100:5.2f}pp  "
              f"draw-mean {np.mean(draws[c])*100:5.2f}pp  "
              f"bar {bar[c]*100:4.2f}pp  fires {np.mean(fired):.0%}  "
              f"-> {'FAIL' if np.mean(fired) > 0.5 else 'PASS'}")
    return res


def selftest():
    """Known-answer fixture: the detector must stay quiet on clean data
    and fire on a planted shift. Nothing is trusted until this passes."""
    rs = np.random.RandomState(7)
    n = 60_000
    base = pd.DataFrame({
        'is_reorder': rs.binomial(1, 0.594, n),
        'is_early_in_cart': rs.binomial(1, 0.286, n)})
    clean_a, clean_b = base.iloc[:30_000], base.iloc[30_000:]

    # Planted fault: an EXACT +5pp shift on is_reorder only. Drawing from
    # p=0.644 rather than flipping rows - flipping 5% of rows moves an
    # already-0.594 rate by only ~2pp, which sits so close to the bar that
    # a pass would be luck rather than evidence the detector works.
    planted = clean_b.copy()
    planted['is_reorder'] = rs.binomial(1, 0.644, len(planted))

    print("  [fixture A] clean: two halves of the same distribution")
    a = score(clean_a, clean_b, 'clean')
    print("  [fixture B] planted: +5pp is_reorder shift")
    b = score(clean_a, planted, 'planted')

    quiet = all(not a['per_column'][c]['fails'] for c in MARGINALS)
    caught = b['per_column']['is_reorder']['fails']
    unharmed = not b['per_column']['is_early_in_cart']['fails']
    ok = quiet and caught and unharmed
    print(f"\n  clean quiet: {quiet} | planted caught: {caught} | "
          f"untouched column unaffected: {unharmed}")
    return ok, {'clean_quiet': quiet, 'planted_caught': caught,
                'untouched_column_unaffected': unharmed, 'passed': ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    assert 'results/tabsyn' in os.path.normpath(
        args.out_dir).replace('\\', '/'), \
        'FATAL: out-dir must live under results/tabsyn (protection rules)'
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 66)
    print("SELFTEST - planted-fault fixture (standing rule 2)")
    print("=" * 66)
    ok, st = selftest()
    if args.selftest:
        out = os.path.join(args.out_dir, 'marginal_errors_selftest.json')
        json.dump(st, open(out, 'w'), indent=2)
        print(f"\nSaved {out}")
        sys.exit(0 if ok else 1)
    assert ok, 'FATAL: selftest failed - detector not trusted'

    train = fitted_population(pd.read_csv('data/v3_train.csv'))
    evl = fitted_population(pd.read_csv('data/v3_eval.csv'))
    ctgan = pd.read_csv('data/synthetic_v3.csv')
    tabsyn = pd.read_csv('data/tabsyn/synthetic_tabsyn.csv')

    print("\n" + "=" * 66)
    print("PRIMARY BASIS - fitted training population "
          f"(10/14, mid dropped, {len(train):,} rows)")
    print("=" * 66)
    prim = {'ctgan': score(train, ctgan, 'CTGAN'),
            'tabsyn': score(train, tabsyn, 'TabSyn')}

    print("\n" + "=" * 66)
    print("KNOWN-ANSWER GATE - CTGAN full-population errors vs recorded")
    print("=" * 66)
    gate = {}
    for c in MARGINALS:
        got = prim['ctgan']['per_column'][c]['point_estimate']
        d = abs(got - RECORDED_CTGAN[c])
        gate[c] = {'recorded': RECORDED_CTGAN[c], 'reproduced': got,
                   'abs_diff': d, 'ok': bool(d <= GATE_TOL)}
        print(f"    {c:20s} recorded {RECORDED_CTGAN[c]:.3f}  "
              f"got {got:.4f}  diff {d:.5f}  "
              f"{'PASS' if d <= GATE_TOL else 'FAIL'}")
    if not all(g['ok'] for g in gate.values()):
        print("\n*** GATE FAILED - nothing written ***")
        sys.exit(1)

    print("\n" + "=" * 66)
    print("SECONDARY BASIS - held-out v3_eval "
          f"(same 10/14 mid-drop, {len(evl):,} rows)")
    print("=" * 66)
    sec = {'ctgan': score(evl, ctgan, 'CTGAN'),
           'tabsyn': score(evl, tabsyn, 'TabSyn')}

    ratios = {}
    for basis, blk in (('training', prim), ('held_out_eval', sec)):
        ratios[basis] = {
            c: float(blk['ctgan']['per_column'][c]['point_estimate']
                     / blk['tabsyn']['per_column'][c]['point_estimate'])
            for c in MARGINALS}

    payload = {
        'purpose': 'single-basis marginal errors for CTGAN v3 and TabSyn',
        'created': '2026-07-27',
        'basis_primary': {
            'reference': 'data/v3_train.csv, order_size_grp recomputed at '
                         '10/14, mid dropped (the population both '
                         'generators were fitted to)',
            'n_real_pop': int(len(train))},
        'basis_secondary': {
            'reference': 'data/v3_eval.csv, same 10/14 mid-drop',
            'n_real_pop': int(len(evl))},
        'design': {
            'n_null_draws': N_NULL_DRAWS, 'n_test_draws': N_TEST_DRAWS,
            'percentile': PERCENTILE, 'seed': SEED,
            'matched_n': 'one n_side used on every side of null AND '
                         'observed draws',
            'rule': 'column FAILS if |rate error| exceeds the measured bar '
                    'on a majority of draws'},
        'known_answer_gate': gate,
        'selftest': st,
        'training_basis': prim,
        'held_out_eval_basis': sec,
        'ctgan_over_tabsyn_error_ratio': ratios,
        'scored_files': {'ctgan': 'data/synthetic_v3.csv',
                         'tabsyn': 'data/tabsyn/synthetic_tabsyn.csv'},
    }
    out = os.path.join(args.out_dir, 'marginal_errors.json')
    json.dump(payload, open(out, 'w'), indent=2)

    print("\n" + "=" * 66)
    print("MATCHED-BASIS RATIO (CTGAN error / TabSyn error)")
    print("=" * 66)
    for basis in ratios:
        print(f"    {basis:14s} " + ', '.join(
            f'{c} {ratios[basis][c]:.1f}x' for c in MARGINALS))
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
