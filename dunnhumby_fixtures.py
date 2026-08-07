"""
dunnhumby_fixtures.py - known-answer verification of the seasonal
detector, on SYNTHETIC data only. Requires no real Dunnhumby files.

A detector that has never been shown to fire on a known-bad case, and
stay quiet on a known-good one, is not a detector. This project has
been burned by unverified checkers repeatedly.

  F1 FLAT          no seasonal effect at all
                   -> raw quiet AND size-controlled quiet
  F2 REAL SEASONAL category mix shifts in the window, basket sizes
                   held IDENTICAL
                   -> raw FIRES and size-controlled FIRES
                      (a genuine seasonal effect must survive the
                       control, or the control is destroying signal)
  F3 SIZE-DRIVEN   category mix does NOT depend on the week at all;
     (THE KEY ONE) it depends only on BASKET SIZE, and the window
                   simply contains bigger baskets
                   -> raw FIRES (this is the false positive we are
                      guarding against) but size-controlled QUIET

F3 is the fixture that earns the basket-size control its place. It
plants exactly the confound this project already found on Instacart -
basket size masquerading as calendar - and the control must strip it.
If F3's controlled test fires, the control does not work and no
seasonal claim from this pipeline can be trusted.

Writes: results/dunnhumby/fixtures.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import dunnhumby_signal_search as ss

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

K = 50
N_BASKETS = 24000
WINDOW = set(range(20, 26))       # 6-week window
ALL_WEEKS = set(range(1, 51))
SEED = 4242


def _cat_probs_for_size(s, lo, hi):
    """p(category | basket size). Low-size baskets favour the first
    half of the catalogue, high-size the second. Used ONLY by F3."""
    a = np.clip((s - lo) / max(hi - lo, 1), 0.0, 1.0)
    p = np.empty(K)
    half = K // 2
    p[:half] = (1.0 - a) * 2.0 + 0.2
    p[half:] = a * 2.0 + 0.2
    return p / p.sum()


def build(case, seed=SEED):
    """Returns a dataframe with basket_id, week_of_year, code."""
    rng = np.random.RandomState(seed)
    base_p = np.full(K, 1.0 / K)
    # a shifted mix for F2: move mass onto 10 categories
    shift_p = base_p.copy()
    shift_p[:10] *= 3.0
    shift_p /= shift_p.sum()

    rows_b, rows_w, rows_c = [], [], []
    bid = 0
    for _ in range(N_BASKETS):
        woy = int(rng.choice(sorted(ALL_WEEKS)))
        in_win = woy in WINDOW

        if case == "F3":
            # window baskets are BIGGER; category mix depends only on
            # size, never on the week
            size = int(rng.poisson(11 if in_win else 5)) + 1
            p = _cat_probs_for_size(size, 1, 16)
        elif case == "F2":
            # sizes IDENTICAL; the mix itself shifts in the window
            size = int(rng.poisson(6)) + 1
            p = shift_p if in_win else base_p
        else:  # F1 flat
            size = int(rng.poisson(6)) + 1
            p = base_p

        codes = rng.choice(K, size=size, p=p)
        rows_b.append(np.full(size, bid))
        rows_w.append(np.full(size, woy))
        rows_c.append(codes)
        bid += 1

    return pd.DataFrame({
        "basket_id": np.concatenate(rows_b),
        "week_of_year": np.concatenate(rows_w),
        "code": np.concatenate(rows_c),
    })


def run_case(case, expect_raw, expect_ctl):
    df = build(case)
    panel = ss.Panel(df, K)
    ia = panel.select_weeks(WINDOW)
    ib = panel.select_weeks(ALL_WEEKS - WINDOW)

    sz_a = panel.sizes[ia].mean()
    sz_b = panel.sizes[ib].mean()

    rng = np.random.RandomState(SEED + 1)
    raw = ss.raw_measure(panel, ia, ib, K,
                         rng, ss.N_NULL_DRAWS, ss.N_TEST_DRAWS)
    rng = np.random.RandomState(SEED + 2)
    ctl = ss.controlled_measure(panel, ia, ib, K, rng,
                                ss.N_NULL_DRAWS, ss.N_TEST_DRAWS)

    print(f"\n  [{case}]  mean basket size: window {sz_a:.2f} vs "
          f"baseline {sz_b:.2f}")
    res = {"mean_size_window": float(sz_a),
           "mean_size_baseline": float(sz_b)}
    ok = True
    for nm, r, exp in (("raw", raw, expect_raw),
                       ("size-controlled", ctl, expect_ctl)):
        if r is None:
            print(f"    {nm:<16} INSUFFICIENT DATA -> FAIL")
            ok = False
            continue
        got = r["effect"]
        good = (got == exp)
        ok &= good
        print(f"    {nm:<16} observed {r['observed_mean']:.5f}  bar "
              f"{r['bar']:.5f}  fires {r['fires']}/{r['n_draws']}  "
              f"-> {'EFFECT' if got else 'quiet':<7} "
              f"(expected {'EFFECT' if exp else 'quiet'})  "
              f"{'PASS' if good else 'FAIL'}")
        res[nm] = {"observed_mean": r["observed_mean"], "bar": r["bar"],
                   "fires": r["fires"], "n_draws": r["n_draws"],
                   "effect": got, "expected": exp, "PASS": good}
    res["PASS"] = bool(ok)
    return res


def main():
    print("=" * 68)
    print("DUNNHUMBY DETECTOR FIXTURES (synthetic data, no real files)")
    print("=" * 68)
    out = {}
    out["F1_flat"] = run_case("F1", expect_raw=False, expect_ctl=False)
    out["F2_real_seasonal"] = run_case("F2", expect_raw=True,
                                       expect_ctl=True)
    out["F3_size_driven"] = run_case("F3", expect_raw=True,
                                     expect_ctl=False)

    allpass = all(v["PASS"] for v in out.values())
    out["ALL_PASS"] = bool(allpass)
    print("\n" + "=" * 68)
    for k, v in out.items():
        if k == "ALL_PASS":
            continue
        print(f"  {k:<22} {'PASS' if v['PASS'] else 'FAIL'}")
    print(f"  {'ALL':<22} {'PASS' if allpass else 'FAIL'}")
    print("=" * 68)
    if allpass:
        print("  F3 passing is the important one: a basket-size-driven")
        print("  pseudo-seasonal effect fires RAW and is stripped by")
        print("  the control. The control works.")
    os.makedirs(ss.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(ss.RESULTS_DIR, "fixtures.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"-> {os.path.join(ss.RESULTS_DIR, 'fixtures.json')}")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
