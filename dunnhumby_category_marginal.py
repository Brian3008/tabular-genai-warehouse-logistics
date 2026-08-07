"""Category-marginal error for a Dunnhumby synthetic table, against a MEASURED
real-vs-real floor.

WHY THIS EXISTS
---------------
The Jul 31 finding that reframed the conditional verdict -- synthetic-vs-real
category TVD 0.14053 against a measured floor of 0.01578, i.e. 9x, and larger
than the entire seasonal effect the model was asked to reproduce -- was
computed AD HOC during that session and lives in no script. It therefore could
not be re-measured for a retrained model on the same basis. This makes it
reproducible and takes the synthetic CSV as an argument, so v1 and v2 are
scored identically.

BASIS (fixed here, quote it with any number this prints)
  * population : the FITTED TRAINING table
                 (tabsyn_repo/data/<dataname>/train.csv), not the full corpus
  * floor      : 95th percentile of TVD between two disjoint halves of that
                 same real table, 20 splits, at n = len(train)//2 per side
  * observed   : real vs synthetic, SIZE-MATCHED to a common n over 10 draws,
                 because TVD is size-dependent (a mismatched n produced a
                 false pass once already in this project)
  * support    : the union of categories in either table; nothing dropped

READS  : tabsyn_repo/data/<dataname>/train.csv, the --synth CSV
WRITES : the --out JSON only

Usage:
  python dunnhumby_category_marginal.py \
      --synth data/dunnhumby/synthetic_season.csv \
      --out results/dunnhumby/category_marginal_v1.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ap = argparse.ArgumentParser()
ap.add_argument("--synth", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dataname", default="dunnhumby_season")
ap.add_argument("--n-floor-splits", type=int, default=20)
ap.add_argument("--n-draws", type=int, default=10)
ap.add_argument("--seed", type=int, default=20260731)
args = ap.parse_args()

TRAIN = Path("tabsyn_repo/data") / args.dataname / "train.csv"
real = pd.read_csv(TRAIN, usecols=["category"])["category"]
syn = pd.read_csv(args.synth, usecols=["category"])["category"]

cats = sorted(set(real.unique()) | set(syn.unique()))
K = len(cats)
print("=" * 70)
print("DUNNHUMBY CATEGORY-MARGINAL ERROR vs a MEASURED floor")
print("=" * 70)
print(f"  real (fitted training table) : {len(real):,} items   {TRAIN}")
print(f"  synthetic                    : {len(syn):,} items   {args.synth}")
print(f"  support (union)              : {K} categories")


def tvd(a, b):
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return float(0.5 * (pa - pb).abs().sum())


rng = np.random.RandomState(args.seed)

# ── FLOOR: two disjoint halves of the real table ────────────────────
half = len(real) // 2
floor = []
for _ in range(args.n_floor_splits):
    perm = rng.permutation(len(real))
    floor.append(tvd(real.iloc[perm[:half]], real.iloc[perm[half:2 * half]]))
floor = np.array(floor)
bar = float(np.percentile(floor, 95))
print(f"\n  FLOOR  {args.n_floor_splits} disjoint half-splits at "
      f"n={half:,}/side")
print(f"    mean {floor.mean():.5f}   sd {floor.std(ddof=1):.5f}   "
      f"95th-pct BAR {bar:.5f}")

# ── OBSERVED: size-matched real vs synthetic ────────────────────────
n_match = min(len(real), len(syn))
obs = []
for _ in range(args.n_draws):
    a = real.sample(n=n_match, random_state=int(rng.randint(1e9)))
    b = syn.sample(n=n_match, random_state=int(rng.randint(1e9)))
    obs.append(tvd(a, b))
obs = np.array(obs)
fires = int(np.sum(obs > bar))
ratio = float(obs.mean() / bar)
print(f"\n  OBSERVED  real vs synthetic, size-matched at n={n_match:,}/side, "
      f"{args.n_draws} draws")
print(f"    mean {obs.mean():.5f}   sd {obs.std(ddof=1):.5f}")
print(f"    fires {fires}/{args.n_draws}   =  {ratio:.1f}x the floor")

# ── structure: coverage collapse or systematic shift? ───────────────
p_r = real.value_counts(normalize=True).reindex(cats, fill_value=0.0)
p_s = syn.value_counts(normalize=True).reindex(cats, fill_value=0.0)
contrib = 0.5 * (p_r - p_s).abs()
tot = float(contrib.sum())
order = p_r.sort_values(ascending=False)
missing = int((p_s == 0).sum())
near0 = int(((p_s > 0) & (p_s < 0.1 * p_r) & (p_r > 0)).sum())
top20_common = float(contrib[order.index[:20]].sum() / tot * 100)
rare100 = float(contrib[order.index[-100:]].sum() / tot * 100)
structure = ("coverage_collapse" if (missing + near0) > 0.05 * K
             else "systematic_shift")
print(f"\n  STRUCTURE")
print(f"    categories absent from synthetic : {missing}")
print(f"    synthetic < 10% of real share    : {near0}")
print(f"    20 most COMMON cats share of TVD : {top20_common:.1f}%")
print(f"    rarest-100 share of total TVD    : {rare100:.1f}%")
print(f"    entropy real {-(p_r[p_r > 0] * np.log(p_r[p_r > 0])).sum():.4f}  "
      f"synth {-(p_s[p_s > 0] * np.log(p_s[p_s > 0])).sum():.4f}")
print(f"    -> {structure.replace('_', ' ').upper()}")

top = (pd.DataFrame({"real_pct": p_r * 100, "synth_pct": p_s * 100,
                     "diff_pp": (p_s - p_r) * 100, "tvd_contrib": contrib})
       .sort_values("tvd_contrib", ascending=False).head(20))
print(f"\n  TOP 10 CONTRIBUTORS")
for c, r in top.head(10).iterrows():
    print(f"    {str(c)[:34]:<36}{r.real_pct:>7.3f}{r.synth_pct:>8.3f}"
          f"{r.diff_pp:>+9.3f}")

out = {
    "synth_csv": args.synth, "dataname": args.dataname,
    "BASIS": "fitted training table; floor = 95th pct of 20 disjoint "
             "half-splits; observed size-matched over 10 draws; TVD is "
             "size-dependent so these are NOT interchangeable with any "
             "figure measured at a different n",
    "n_real": int(len(real)), "n_synth": int(len(syn)), "n_categories": K,
    "floor": {"draws": [float(x) for x in floor], "mean": float(floor.mean()),
              "sd": float(floor.std(ddof=1)), "bar_95pct": bar,
              "n_splits": args.n_floor_splits, "n_per_side": half},
    "observed": {"draws": [float(x) for x in obs], "mean": float(obs.mean()),
                 "sd": float(obs.std(ddof=1)), "n_per_side": n_match,
                 "fires": fires, "n_draws": args.n_draws,
                 "fire_rate": fires / args.n_draws},
    "ratio_to_floor": ratio,
    "structure": {
        "verdict": structure,
        "categories_absent": missing, "categories_near_zero": near0,
        "top20_MOST_COMMON_share_of_tvd_pct": top20_common,
        "rarest100_share_of_tvd_pct": rare100,
        "entropy_real": float(-(p_r[p_r > 0] * np.log(p_r[p_r > 0])).sum()),
        "entropy_synth": float(-(p_s[p_s > 0] * np.log(p_s[p_s > 0])).sum()),
        "top20_contributors": [
            {"category": str(c), "real_pct": float(r.real_pct),
             "synth_pct": float(r.synth_pct), "diff_pp": float(r.diff_pp),
             "tvd_contrib": float(r.tvd_contrib)}
            for c, r in top.iterrows()],
    },
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
print(f"\n  -> {args.out}")
