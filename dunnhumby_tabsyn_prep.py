"""
dunnhumby_tabsyn_prep.py - build the TabSyn training tables for the
seasonal-conditioning test.

WRITES INSIDE tabsyn_repo/ (approved, additive only):
    tabsyn_repo/data/dunnhumby_season/{X_num,X_cat,y}_{train,test}.npy
                                      train.csv info.json prep_manifest.json
Nothing existing in tabsyn_repo/ is modified; the `warehouse` and
`warehouse_smoke` datanames and their checkpoints are untouched.
Artifacts of record stay in data/dunnhumby/ and results/dunnhumby/.

SCHEMA - four columns, none a deterministic function of another
---------------------------------------------------------------
  category         305 categories   the axis under test
  season_period    binary           conditioning axis + binclass target
  basket_size_grp  3 groups         lets the size-controlled test run
                                    on synthetic too
  quantity         numeric          independent

week_of_year is DELIBERATELY EXCLUDED. season_period is a
deterministic function of it, so including both would let the model
recover the condition trivially and make the conditional test
circular - the exact defect that killed v2's aisle_popularity
seasonal conditioning.

TabSyn has NO conditional API (see project notes): it is trained
unconditionally and season_period is learned as an ordinary column,
exactly as order_size_grp was on Instacart. Generated rows are
partitioned by that column afterwards.

Reads:  data/dunnhumby/dj_items.csv
        results/dunnhumby/signal_search.json   (for the frozen window)
Writes: tabsyn_repo/data/dunnhumby_season/*
        results/dunnhumby/tabsyn_prep_report.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = Path("data/dunnhumby")
RESULTS_DIR = Path("results/dunnhumby")
DATANAME = "dunnhumby_season"
OUT_DIR = Path("tabsyn_repo/data") / DATANAME

CAT_COLS = ["category", "basket_size_grp"]
NUM_COLS = ["quantity"]
TARGET = "season_period"
ALL_COLS = ["category", "basket_size_grp", "quantity", TARGET]

TEST_FRAC = 0.10
SEED = 20260729
MAX_TRAIN_ROWS = 600_000   # measured cap, see [4]

print("=" * 68)
print("DUNNHUMBY -> TABSYN PREP (dataname: %s)" % DATANAME)
print("=" * 68)

# ══ [1] the frozen window, from the gate ═════════════════
gate = json.load(open(RESULTS_DIR / "signal_search.json",
                      encoding="utf-8"))
assert gate["GATE_PASS"], "FATAL: gate did not pass; nothing to train"
w0, w1 = gate["discovery"]["frozen_window"]
lo, hi = gate["usable_weeks"]
WIN = set(range(w0, w1 + 1))
print(f"\n[1] frozen window weeks {w0}..{w1}   usable weeks {lo}..{hi}")
print(f"    (window frozen from YEAR 1 in the gate; not re-chosen here)")

# ══ [2] load + restrict, exactly as the gate did ═════════
df = pd.read_csv(DATA_DIR / "dj_items.csv")
n_all = len(df)
df = df[(df["week_of_year"] >= lo) & (df["week_of_year"] <= hi)].copy()
print(f"\n[2] items {n_all:,} -> {len(df):,} after the gate's usable-week")
print(f"    restriction (panel ramp-up weeks excluded from BOTH years)")

df["season_period"] = np.where(df["week_of_year"].isin(WIN),
                               "window", "baseline")
bs = df.groupby("basket_id")["category"].transform("size")
# measured tertiles, not guessed
q33, q67 = bs.quantile([1 / 3, 2 / 3])
df["basket_size_grp"] = np.where(bs <= q33, "small",
                                 np.where(bs >= q67, "large", "mid"))
print(f"    basket_size_grp tertiles (measured): small<={q33:.0f} "
      f"large>={q67:.0f}")
print(f"    season_period: "
      f"{df['season_period'].value_counts().to_dict()}")

# ══ [3] circularity + integrity gates ════════════════════
print("\n[3] GATES")
X = df[ALL_COLS].copy()
assert not X.isna().any().any(), "FATAL: NaNs in training columns"
print("    no NaNs                                   PASS")

# no column may be a deterministic function of another
bad = []
for a in ALL_COLS:
    for b in ALL_COLS:
        if a == b:
            continue
        g = df.groupby(a)[b].nunique()
        if int(g.max()) == 1 and df[a].nunique() > 1:
            bad.append((a, b))
assert not bad, f"FATAL: deterministic redundancy {bad}"
print("    no deterministic redundancy between columns PASS")
assert "week_of_year" not in ALL_COLS
print("    week_of_year EXCLUDED (anti-circularity)   PASS")

# ══ [4] size cap, measured against the prior bench ═══════
print("\n[4] TRAIN SIZE")
print(f"    available {len(X):,} rows "
      f"(Instacart TabSyn trained on 272,820)")
if len(X) > MAX_TRAIN_ROWS:
    # stratified on season_period so the conditional axis keeps its
    # real proportions - the whole point of the model
    rng = np.random.RandomState(SEED)
    keep = (X.groupby(TARGET, group_keys=False)
            .apply(lambda g: g.sample(
                n=int(round(MAX_TRAIN_ROWS * len(g) / len(X))),
                random_state=SEED)))
    print(f"    capped to {len(keep):,} rows, STRATIFIED on "
          f"{TARGET} to preserve its proportions")
    print(f"    proportions before "
          f"{ (X[TARGET].value_counts(normalize=True)).round(4).to_dict() }")
    print(f"    proportions after  "
          f"{ (keep[TARGET].value_counts(normalize=True)).round(4).to_dict() }")
    X = keep
else:
    print("    no cap needed")

# ══ [5] split + write ════════════════════════════════════
# CATEGORY-STRATIFIED SPLIT. A plain random split put exactly one
# category (FROZEN PACKAGE MEAT) in test but not train; TabSyn sizes
# its embedding table from TRAIN, so that single unseen value produced
# a device-side assert (`t >= 0 && t < n_classes`) and killed the run
# at the first validation pass. Every category must therefore appear in
# train. Rare categories are NOT dropped - the 12 categories with <10
# rows cover 0.008% of items, but dropping them would move the TVD
# support away from the basis the gate measured.
X = X.reset_index(drop=True)
rng = np.random.RandomState(SEED)
te_parts = []
for _, grp in X.groupby("category", sort=True):
    k = len(grp)
    n_te = min(int(np.floor(k * TEST_FRAC)), k - 1)   # keep >=1 in train
    if n_te > 0:
        te_parts.append(rng.choice(grp.index.to_numpy(), n_te,
                                   replace=False))
te_idx = (np.concatenate(te_parts) if te_parts
          else np.array([], dtype=int))
Xte = X.loc[te_idx]
Xtr = X.drop(index=te_idx)
print(f"\n[5] split (category-stratified): train {len(Xtr):,}  "
      f"test {len(Xte):,}")

# GATE: the defect that crashed the first run is now asserted
unseen = set(Xte["category"]) - set(Xtr["category"])
assert not unseen, f"FATAL: categories in test but not train: {unseen}"
missing = set(X["category"]) - set(Xtr["category"])
assert not missing, f"FATAL: categories absent from train: {missing}"
print(f"    every category present in train              PASS")
print(f"    test categories subset of train              PASS")
print(f"    season_period proportions train "
      f"{Xtr[TARGET].value_counts(normalize=True).round(4).to_dict()}")
print(f"                              test  "
      f"{Xte[TARGET].value_counts(normalize=True).round(4).to_dict()}")

cat_codes = {c: {v: i for i, v in
                 enumerate(sorted(X[c].astype(str).unique()))}
             for c in CAT_COLS}
tgt_codes = {v: i for i, v in
             enumerate(sorted(X[TARGET].astype(str).unique()))}
print(f"    category cardinality {len(cat_codes['category'])}   "
      f"target classes {tgt_codes}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, part in (("train", Xtr), ("test", Xte)):
    np.save(OUT_DIR / f"X_num_{name}.npy",
            part[NUM_COLS].to_numpy(dtype=np.float32))
    np.save(OUT_DIR / f"X_cat_{name}.npy",
            part[CAT_COLS].astype(str).to_numpy())
    np.save(OUT_DIR / f"y_{name}.npy",
            part[TARGET].astype(str).map(tgt_codes)
            .to_numpy(dtype=np.int64))
Xtr.to_csv(OUT_DIR / "train.csv", index=False)

column_names = NUM_COLS + CAT_COLS + [TARGET]
info = {
    "name": DATANAME,
    "task_type": "binclass",
    "n_classes": len(tgt_codes),
    "header": "infer",
    "column_names": column_names,
    "num_col_idx": [0],
    "cat_col_idx": [1, 2],
    "target_col_idx": [3],
    "file_type": "csv",
    "data_path": f"data/{DATANAME}/train.csv",
    "test_path": None,
    "train_num": int(len(Xtr)),
    "test_num": int(len(Xte)),
    "idx_mapping": {str(i): i for i in range(len(column_names))},
    "inverse_idx_mapping": {str(i): i for i in range(len(column_names))},
    "idx_name_mapping": {str(i): c for i, c in enumerate(column_names)},
}
json.dump(info, open(OUT_DIR / "info.json", "w"), indent=4)

manifest = {
    "dataname": DATANAME,
    "frozen_window": [w0, w1],
    "usable_weeks": [lo, hi],
    "columns": ALL_COLS,
    "excluded_week_of_year": True,
    "basket_size_tertiles": [float(q33), float(q67)],
    "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    "n_categories": len(cat_codes["category"]),
    "target_classes": tgt_codes,
    "season_period_proportions":
        X[TARGET].value_counts(normalize=True).round(6).to_dict(),
    "NOTE_conditional": (
        "TabSyn has no conditional API; season_period is learned as an "
        "ordinary column and generated rows are partitioned by it, "
        "exactly as order_size_grp was on the Instacart bench."),
}
json.dump(manifest, open(OUT_DIR / "prep_manifest.json", "w"), indent=2)
json.dump(manifest,
          open(RESULTS_DIR / "tabsyn_prep_report.json", "w"), indent=2)
print(f"\n-> {OUT_DIR}")
print(f"-> {RESULTS_DIR / 'tabsyn_prep_report.json'}")
