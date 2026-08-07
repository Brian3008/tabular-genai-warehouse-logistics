"""
dunnhumby_signal_search.py - THE HARD GATE.

Does seasonal period actually shift the category mix on Dunnhumby,
against a MEASURED noise floor, in BOTH years?

If it does not, we stop and that is the finding: it replicates the
Instacart "calendar carries no signal" result on a second dataset and
a second retailer, which is a stronger claim than a positive.

DESIGN RULES (inherited from test_real_fleet_effect.py, this project's
audited template - NOT from signal_search.py, whose `analyse()`
compares groups at whatever size they happen to have, the mismatched-n
pattern that produced a false pass here once already)
--------------------------------------------------------------------
1. SIZE-MATCHED. One budget, computed once, used identically in the
   null and the observed test. TVD is sample-size dependent.
2. THE NULL IS MEASURED, NOT GUESSED. Split a SINGLE period in two -
   same season on both sides, so the difference is pure sampling
   noise. Repeat N_NULL_DRAWS times. Bar = 95th percentile.
3. THE NULL IS CLUSTERED BY BASKET, NOT BY ITEM. Items within a
   shopping trip are correlated, so an item-level null understates the
   true variance and turns basket-clustering noise into a "seasonal
   effect". This was NOT hypothetical: the first version split items
   and its own Case-A fixture fired 10/10 on two halves of the SAME
   year. The sampling unit is the basket, in the null and the observed
   test alike.
4. CONSISTENT ESTIMATOR NOISE. Identical sampling scheme both sides.
5. MAJORITY VOTE. A 95th-pct bar fires ~5% of the time by
   construction, so one draw proves nothing. An effect is claimed only
   if it fires on a MAJORITY of N_TEST_DRAWS.
6. REPLICATION. The window is DISCOVERED in year 1 and CONFIRMED in
   year 2. Year 2 is untouched during selection - the disjointness
   discipline this project applies to train/eval splits, moved onto
   the seasonal axis. Scanning all windows and reporting the best
   would be p-hacking.
7. THE BASKET-SIZE CONTROL. This project's own finding is that basket
   SIZE, not calendar, carries category signal. Holiday baskets are
   bigger, so a raw seasonal shift may be the basket-size effect in a
   calendar costume. EVERY test runs twice: raw, and with the periods
   matched on their basket-size histogram. A shift that dies under the
   control is not a seasonal shift.
8. PANEL RAMP-UP EXCLUDED. Dunnhumby onboarded households
   progressively; year 1's early weeks hold a fraction of year 2's
   households and stores. Those weeks measure panel composition, not
   season. Cutoff measured from the data, dropped from BOTH years.
9. FIXTURE-VERIFIED. The selftest runs on REAL data before the real
   test: Case A (two halves of one period) must report NO effect;
   Case B (planted category shift) must DETECT it.

Reads:  data/dunnhumby/dj_items.csv
Writes: results/dunnhumby/signal_search.json
        results/dunnhumby/window_profile.csv

Usage:  python dunnhumby_signal_search.py --selftest
        python dunnhumby_signal_search.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join("data", "dunnhumby")
RESULTS_DIR = os.path.join("results", "dunnhumby")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_NULL_DRAWS = 40      # draws building the null distribution
N_TEST_DRAWS = 10      # observed draws; majority vote over these
PCTL = 95              # bar = this percentile of the null
SEED = 20260729
WINDOW_LENGTHS = (4, 6)
BUDGET_FRAC = 0.45     # item budget per side; < 0.5 so a half-split
                       # of the smaller side always clears it


def tvd(c1, c2, K):
    p = np.bincount(c1, minlength=K).astype(float)
    q = np.bincount(c2, minlength=K).astype(float)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def size_bin(n):
    """Bins for the basket-size control. Exact-size matching starves
    in the tail; bins keep the tail represented with only a small
    residual size difference (reported)."""
    if n <= 19:
        return n
    if n <= 24:
        return 20
    if n <= 29:
        return 25
    if n <= 39:
        return 30
    return 40


class Panel:
    """Item codes grouped by basket."""

    def __init__(self, df, K, codes=None):
        self.K = K
        df = df.sort_values("basket_id", kind="mergesort")
        self.codes = (df["code"].to_numpy() if codes is None else codes)
        bid = df["basket_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
        self.starts = starts
        self.ends = np.r_[starts[1:], len(bid)]
        self.sizes = self.ends - starts
        self.bins = np.array([size_bin(s) for s in self.sizes])
        self.woy = df["week_of_year"].to_numpy()[starts]
        self.n_baskets = len(starts)

    def copy_with_codes(self, codes):
        p = object.__new__(Panel)
        p.__dict__.update(self.__dict__)
        p.codes = codes
        return p

    def items_of(self, basket_idx):
        if len(basket_idx) == 0:
            return np.empty(0, dtype=self.codes.dtype)
        return np.concatenate(
            [self.codes[self.starts[i]:self.ends[i]] for i in basket_idx])

    def select_weeks(self, weeks):
        return np.flatnonzero(np.isin(self.woy, list(weeks)))


def _items_capped(panel, basket_idx, n_items, rng):
    """Whole baskets in random order until n_items, then truncate.
    Taking WHOLE baskets preserves the within-trip correlation that
    makes an item-level null invalid (design rule 3)."""
    if len(basket_idx) == 0:
        return None
    order = rng.permutation(basket_idx)
    csum = np.cumsum(panel.sizes[order])
    if csum[-1] < n_items:
        return None
    k = int(np.searchsorted(csum, n_items, side="left")) + 1
    return panel.items_of(order[:min(k, len(order))])[:n_items]


# ══════════════════════════════════════════════════════════
# RAW TEST  (basket-clustered)
# ══════════════════════════════════════════════════════════
def raw_measure(panel, idx_a, idx_b, K, rng, n_null, n_test):
    items_a = int(panel.sizes[idx_a].sum())
    items_b = int(panel.sizes[idx_b].sum())
    N = int(BUDGET_FRAC * min(items_a, items_b))
    if N < 500:
        return None
    null = []
    for i in range(n_null):
        src = idx_b if i % 2 == 0 else idx_a
        order = rng.permutation(src)
        half = len(order) // 2
        a = _items_capped(panel, order[:half], N, rng)
        b = _items_capped(panel, order[half:], N, rng)
        if a is None or b is None:
            continue
        null.append(tvd(a, b, K))
    if len(null) < n_null // 2:
        return None
    bar = float(np.percentile(null, PCTL))
    obs = []
    for _ in range(n_test):
        a = _items_capped(panel, idx_a, N, rng)
        b = _items_capped(panel, idx_b, N, rng)
        if a is None or b is None:
            continue
        obs.append(tvd(a, b, K))
    if not obs:
        return None
    fires = int(sum(o > bar for o in obs))
    return {"N_items_per_side": int(N), "bar": bar,
            "null_mean": float(np.mean(null)),
            "null_n": len(null),
            "observed_mean": float(np.mean(obs)),
            "observed": [float(o) for o in obs],
            "fires": fires, "n_draws": len(obs),
            "fire_rate": fires / len(obs),
            "effect": fires > len(obs) // 2}


# ══════════════════════════════════════════════════════════
# BASKET-SIZE-CONTROLLED TEST
# ══════════════════════════════════════════════════════════
def _match_plan(panel, idx_a, idx_b):
    """k_s baskets per size-bin from BOTH sides.

    k_s = min(count_a//2, count_b//2). The //2 on BOTH sides is
    required: the null alternates which side supplies it and must draw
    two DISJOINT same-size groups from whichever side it uses. Same
    'budget = half the smallest group' rule as
    validate_demand_geometry.py; keeps null and observed at the SAME
    item count."""
    plan = {}
    ba, bb = panel.bins[idx_a], panel.bins[idx_b]
    for s in np.unique(ba):
        k = min(int((ba == s).sum()) // 2, int((bb == s).sum()) // 2)
        if k > 0:
            plan[int(s)] = k
    return plan


def controlled_measure(panel, idx_a, idx_b, K, rng, n_null, n_test):
    plan = _match_plan(panel, idx_a, idx_b)
    if not plan:
        return None
    ba, bb = panel.bins[idx_a], panel.bins[idx_b]
    pools_a = {s: idx_a[ba == s] for s in plan}
    pools_b = {s: idx_b[bb == s] for s in plan}

    def draw(pools):
        picks = [pools[s][rng.choice(len(pools[s]), k, replace=False)]
                 for s, k in plan.items()]
        return panel.items_of(np.concatenate(picks))

    def draw_null_pair(pools):
        g1, g2 = [], []
        for s, k in plan.items():
            p = pools[s]
            sel = p[rng.choice(len(p), 2 * k, replace=False)]
            g1.append(sel[:k]); g2.append(sel[k:])
        return (panel.items_of(np.concatenate(g1)),
                panel.items_of(np.concatenate(g2)))

    null = []
    for i in range(n_null):
        try:
            a, b = draw_null_pair(pools_b if i % 2 == 0 else pools_a)
        except ValueError:
            continue
        null.append(tvd(a, b, K))
    if len(null) < n_null // 2:
        return None
    bar = float(np.percentile(null, PCTL))

    obs, resid, sz = [], [], []
    for _ in range(n_test):
        a, b = draw(pools_a), draw(pools_b)
        obs.append(tvd(a, b, K))
        resid.append(abs(len(a) - len(b)))
    fires = int(sum(o > bar for o in obs))
    return {"n_baskets_per_side": int(sum(plan.values())),
            "bar": bar, "null_mean": float(np.mean(null)),
            "null_n": len(null),
            "observed_mean": float(np.mean(obs)),
            "observed": [float(o) for o in obs],
            "mean_item_count_residual": float(np.mean(resid)),
            "fires": fires, "n_draws": n_test,
            "fire_rate": fires / n_test,
            "effect": fires > n_test // 2}


# ══════════════════════════════════════════════════════════
# FIXTURES - run on REAL data before the real test
# ══════════════════════════════════════════════════════════
def selftest(panel, K, verbose=True):
    if verbose:
        print("\n" + "=" * 68)
        print("SELFTEST on REAL data - verify before use")
        print("=" * 68)
    rng = np.random.RandomState(SEED)
    ok = True
    idx = rng.permutation(panel.n_baskets)
    half = len(idx) // 2
    ia, ib = idx[:half], idx[half:]

    ra = raw_measure(panel, ia, ib, K, rng, N_NULL_DRAWS, N_TEST_DRAWS)
    ca = controlled_measure(panel, ia, ib, K, rng, N_NULL_DRAWS,
                            N_TEST_DRAWS)
    if verbose:
        print("\n  CASE A (two halves of the SAME data - must be quiet)")
        for nm, r in (("raw", ra), ("size-controlled", ca)):
            print(f"    {nm:<16} observed {r['observed_mean']:.5f}  "
                  f"bar {r['bar']:.5f}  fires "
                  f"{r['fires']}/{r['n_draws']}")
    if ra["effect"] or ca["effect"]:
        ok = False
        if verbose:
            print("    FAIL - reported an effect on identical data")
    elif verbose:
        print("    PASS - quiet on flat data")

    # CASE B: plant a category shift in one half's items
    top = int(np.bincount(panel.codes, minlength=K).argmax())
    codes2 = panel.codes.copy()
    mask = np.zeros(len(codes2), dtype=bool)
    for i in ib:
        mask[panel.starts[i]:panel.ends[i]] = True
    pos = np.flatnonzero(mask)
    hit = rng.choice(pos, int(0.10 * len(pos)), replace=False)
    codes2[hit] = top
    p2 = panel.copy_with_codes(codes2)

    rb = raw_measure(p2, ia, ib, K, rng, N_NULL_DRAWS, N_TEST_DRAWS)
    cb = controlled_measure(p2, ia, ib, K, rng, N_NULL_DRAWS,
                            N_TEST_DRAWS)
    if verbose:
        print("\n  CASE B (10% of one half's items forced to the most")
        print("          common category - must DETECT)")
        for nm, r in (("raw", rb), ("size-controlled", cb)):
            print(f"    {nm:<16} observed {r['observed_mean']:.5f}  "
                  f"bar {r['bar']:.5f}  fires "
                  f"{r['fires']}/{r['n_draws']}")
    if not (rb["effect"] and cb["effect"]):
        ok = False
        if verbose:
            print("    FAIL - missed a planted shift")
    elif verbose:
        print("    PASS - planted shift detected")

    if verbose:
        print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok, {"caseA_raw": ra, "caseA_ctl": ca,
                "caseB_raw": rb, "caseB_ctl": cb}


# ══════════════════════════════════════════════════════════
def load_panels():
    df = pd.read_csv(os.path.join(DATA_DIR, "dj_items.csv"))
    cats = pd.Categorical(df["category"])
    df["code"] = cats.codes
    K = len(cats.categories)
    print(f"\n  {len(df):,} items   {df['basket_id'].nunique():,} "
          f"baskets   {K} categories")

    common = sorted(set(df[df["year"] == 1]["week_of_year"]) &
                    set(df[df["year"] == 2]["week_of_year"]))

    # ---- PANEL RAMP-UP, measured ---------------------------------
    # Only the LEADING ramp is dropped. Requiring the ratio to hold
    # for every later week would let one ordinary mid-year dip discard
    # 40 of 50 weeks (it did, in the first version).
    hh = (df.groupby(["year", "week_of_year"])["household_key"]
          .nunique().unstack(0))
    ratio = (hh[1] / hh[2]).reindex(common)
    okr = (ratio >= 0.90)
    cutoff, run = common[0], 3
    for i, w in enumerate(common):
        nxt = common[i:i + run]
        if len(nxt) == run and all(bool(okr.loc[x]) for x in nxt):
            cutoff = w
            break
    dropped = [w for w in common if w < cutoff]
    common = [w for w in common if w >= cutoff]
    later_dips = [int(w) for w in common if not bool(okr.loc[w])]
    print(f"\n  PANEL RAMP-UP CUTOFF (measured): week {cutoff}")
    print(f"    year1/year2 household ratio first holds >=0.90 for "
          f"{run} consecutive weeks at week {cutoff}")
    if dropped:
        print(f"    dropped week_of_year {min(dropped)}..{max(dropped)}"
              f" ({len(dropped)} weeks) from BOTH years")
    print(f"    later weeks below 0.90 (kept, ordinary variation): "
          f"{later_dips if later_dips else 'none'}")

    df = df[df["week_of_year"].isin(common)]
    print(f"  usable week_of_year {min(common)}..{max(common)} "
          f"({len(common)} weeks) -> {len(df):,} items")
    p1 = Panel(df[df["year"] == 1], K)
    p2 = Panel(df[df["year"] == 2], K)
    print(f"  year 1: {p1.n_baskets:,} baskets   "
          f"year 2: {p2.n_baskets:,} baskets")
    return df, p1, p2, K, common, cutoff, dropped, later_dips


def main():
    print("=" * 68)
    print("DUNNHUMBY SEASONAL SIGNAL SEARCH - THE GATE")
    print("=" * 68)
    df, p1, p2, K, common, cutoff, dropped, dips = load_panels()

    ok, fixtures = selftest(p1, K)
    if not ok:
        print("\nABORT: detector failed its fixtures. No result is "
              "trustworthy until it passes.")
        sys.exit(1)

    out = {"n_null_draws": N_NULL_DRAWS, "n_test_draws": N_TEST_DRAWS,
           "percentile": PCTL, "seed": SEED, "n_categories": K,
           "rule": "effect claimed only if it fires on a majority",
           "usable_weeks": [int(min(common)), int(max(common))],
           "n_usable_weeks": len(common),
           "panel_ramp_cutoff_week": int(cutoff),
           "panel_ramp_weeks_dropped": [int(w) for w in dropped],
           "later_weeks_below_ratio_0.90": dips,
           "selftest": {
               "caseA_raw_fires": fixtures["caseA_raw"]["fires"],
               "caseA_ctl_fires": fixtures["caseA_ctl"]["fires"],
               "caseB_raw_fires": fixtures["caseB_raw"]["fires"],
               "caseB_ctl_fires": fixtures["caseB_ctl"]["fires"],
               "PASS": True}}

    # ── DISCOVERY: year 1 only ────────────────────────────
    print("\n" + "=" * 68)
    print("DISCOVERY - year 1 ONLY (year 2 untouched here)")
    print("=" * 68)
    rng = np.random.RandomState(SEED)
    profile = []
    for L in WINDOW_LENGTHS:
        for start in range(min(common), max(common) - L + 2):
            win = set(range(start, start + L))
            if not win <= set(common):
                continue
            ia = p1.select_weeks(win)
            ib = p1.select_weeks(set(common) - win)
            if len(ia) < 200 or len(ib) < 200:
                continue
            N = int(BUDGET_FRAC * min(p1.sizes[ia].sum(),
                                      p1.sizes[ib].sum()))
            a = _items_capped(p1, ia, N, rng)
            b = _items_capped(p1, ib, N, rng)
            if a is None or b is None:
                continue
            profile.append({"length": L, "start": start,
                            "end": start + L - 1,
                            "tvd_year1": tvd(a, b, K),
                            "n_items_window": int(p1.sizes[ia].sum())})
    prof = pd.DataFrame(profile).sort_values("tvd_year1",
                                             ascending=False)
    prof.to_csv(os.path.join(RESULTS_DIR, "window_profile.csv"),
                index=False)
    print(f"\n  scanned {len(prof)} windows (lengths "
          f"{list(WINDOW_LENGTHS)}); top 5 by year-1 TVD:")
    print(prof.head(5).to_string(index=False))
    best = prof.iloc[0]
    WIN = set(range(int(best["start"]), int(best["end"]) + 1))
    print(f"\n  FROZEN WINDOW (year 1): weeks {int(best['start'])}.."
          f"{int(best['end'])}  (year-1 TVD "
          f"{best['tvd_year1']:.5f})")
    out["discovery"] = {
        "frozen_window": [int(best["start"]), int(best["end"])],
        "length": int(best["length"]),
        "year1_tvd_point_estimate": float(best["tvd_year1"]),
        "n_windows_scanned": int(len(prof)),
        "NOTE": "chosen on YEAR 1 ONLY; year 2 is out-of-sample"}

    # ── CONFIRMATION ──────────────────────────────────────
    print("\n" + "=" * 68)
    print("CONFIRMATION - frozen window, tested in EACH year")
    print("=" * 68)
    results = {}
    for label, panel in (("year1_IN_SAMPLE", p1),
                         ("year2_HELD_OUT", p2)):
        ia = panel.select_weeks(WIN)
        ib = panel.select_weeks(set(common) - WIN)
        rng = np.random.RandomState(SEED + 1)
        raw = raw_measure(panel, ia, ib, K, rng, N_NULL_DRAWS,
                          N_TEST_DRAWS)
        rng = np.random.RandomState(SEED + 2)
        ctl = controlled_measure(panel, ia, ib, K, rng, N_NULL_DRAWS,
                                 N_TEST_DRAWS)
        results[label] = {"raw": raw, "size_controlled": ctl,
                          "mean_basket_size_window":
                              float(panel.sizes[ia].mean()),
                          "mean_basket_size_baseline":
                              float(panel.sizes[ib].mean())}
        print(f"\n  [{label}]  weeks {min(WIN)}..{max(WIN)}   "
              f"mean basket size {panel.sizes[ia].mean():.2f} vs "
              f"{panel.sizes[ib].mean():.2f}")
        for nm, r in (("raw", raw), ("size-controlled", ctl)):
            if r is None:
                print(f"    {nm:<16} INSUFFICIENT DATA")
                continue
            print(f"    {nm:<16} observed {r['observed_mean']:.5f}  "
                  f"null {r['null_mean']:.5f}  bar {r['bar']:.5f}  "
                  f"fires {r['fires']}/{r['n_draws']} "
                  f"({100*r['fire_rate']:.0f}%)  -> "
                  f"{'EFFECT' if r['effect'] else 'no effect'}")
    out["confirmation"] = results

    # ── WEEK-BLOCK NULL - the strictest test ──────────────
    # The basket-clustered null above splits baskets randomly WITHIN a
    # period. But the observed test compares different WEEKS, and weeks
    # differ for non-seasonal reasons too (promotions, store mix, local
    # events). That variance is absent from a random basket split, so
    # the basket null is anti-conservative - visible in the selftest,
    # where Case A raw fired 3/10 against a 95th-pct bar that should
    # fire ~0.5/10.
    #
    # The honest null for a week-based hypothesis is a WEEK-based null:
    # compute the TVD of EVERY contiguous window of the same length
    # against its own complement. A genuinely seasonal window must
    # stand out from that distribution, not merely from basket noise.
    #
    # In year 1 this is circular - the window was CHOSEN as the year-1
    # maximum, so it is the max by construction. In year 2 the window
    # is frozen and untouched, so its rank there is a real test.
    print("\n" + "=" * 68)
    print("WEEK-BLOCK NULL - strictest test (year 2 is the real one)")
    print("=" * 68)
    L = int(best["length"])
    wb = {}
    for label, panel in (("year1_IN_SAMPLE", p1),
                         ("year2_HELD_OUT", p2)):
        rngw = np.random.RandomState(SEED + 3)
        vals = {}
        for s in range(min(common), max(common) - L + 2):
            w = set(range(s, s + L))
            if not w <= set(common):
                continue
            ia = panel.select_weeks(w)
            ib = panel.select_weeks(set(common) - w)
            if len(ia) < 200 or len(ib) < 200:
                continue
            N = int(BUDGET_FRAC * min(panel.sizes[ia].sum(),
                                      panel.sizes[ib].sum()))
            ts = []
            for _ in range(3):
                a = _items_capped(panel, ia, N, rngw)
                b = _items_capped(panel, ib, N, rngw)
                if a is not None and b is not None:
                    ts.append(tvd(a, b, K))
            if ts:
                vals[s] = float(np.mean(ts))
        obs_v = vals[int(best["start"])]
        others = [v for s, v in vals.items() if s != int(best["start"])]
        bar_w = float(np.percentile(others, PCTL))
        rank = int(sum(v >= obs_v for v in vals.values()))
        pct = 100.0 * (1 - (rank - 1) / len(vals))
        wb[label] = {"window_tvd": obs_v, "bar_95pct_other_windows":
                     bar_w, "n_windows": len(vals),
                     "rank_of_frozen_window": rank,
                     "percentile": pct,
                     "exceeds_bar": bool(obs_v > bar_w),
                     "all_window_tvds": {int(k): v
                                         for k, v in vals.items()}}
        print(f"\n  [{label}]  {len(vals)} contiguous {L}-week windows")
        print(f"    frozen window TVD {obs_v:.5f}   "
              f"95th pct of the other {len(others)} windows "
              f"{bar_w:.5f}")
        print(f"    rank {rank}/{len(vals)} "
              f"({pct:.0f}th percentile)  -> "
              f"{'EXCEEDS' if obs_v > bar_w else 'does NOT exceed'} "
              f"the week-block bar")
    out["week_block_null"] = wb

    # ── THE GATE ──────────────────────────────────────────
    y1, y2 = results["year1_IN_SAMPLE"], results["year2_HELD_OUT"]
    raw_both = bool(y1["raw"] and y2["raw"] and y1["raw"]["effect"]
                    and y2["raw"]["effect"])
    ctl_both = bool(y1["size_controlled"] and y2["size_controlled"]
                    and y1["size_controlled"]["effect"]
                    and y2["size_controlled"]["effect"])
    wb2 = out["week_block_null"]["year2_HELD_OUT"]
    week_ok = bool(wb2["exceeds_bar"])
    gate = raw_both and ctl_both and week_ok
    print("\n" + "=" * 68)
    print("THE GATE")
    print("=" * 68)
    print(f"  raw effect in BOTH years:              {raw_both}")
    print(f"  survives basket-size control BOTH yrs: {ctl_both}")
    print(f"  clears the WEEK-BLOCK bar in year 2:    {week_ok}"
          f"   (frozen window {wb2['percentile']:.0f}th pct of "
          f"{wb2['n_windows']} windows)")
    print(f"\n  GATE: {'PASS - real seasonal signal' if gate else 'FAIL - STOP, do not train'}")
    if not gate:
        msg = (("Raw seasonal shift exists but DIES under the "
                "basket-size control: the apparent seasonal signal is "
                "the basket-size effect, replicating the Instacart "
                "finding on a second retailer.")
               if raw_both else
               ("Seasonal period does not shift category mix above a "
                "measured noise floor in both years: calendar carries "
                "no signal, replicating the Instacart result on a "
                "second dataset."))
        print(f"  -> {msg}")
        out["gate_interpretation"] = msg
    out["GATE_PASS"] = gate
    out["raw_effect_both_years"] = raw_both
    out["survives_size_control_both_years"] = ctl_both
    out["clears_week_block_bar_year2"] = week_ok

    with open(os.path.join(RESULTS_DIR, "signal_search.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {os.path.join(RESULTS_DIR, 'signal_search.json')}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _, p1, _, K, *_ = load_panels()
        ok, _ = selftest(p1, K)
        sys.exit(0 if ok else 1)
    main()
