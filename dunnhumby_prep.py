"""
dunnhumby_prep.py - prepare the Dunnhumby "Complete Journey" data
for the seasonal-conditioning test.

COMPLETELY SEPARATE from the Instacart work. Reads only Dunnhumby
files; writes only data/dunnhumby/ and results/dunnhumby/. No
Instacart artifact is read or written anywhere in this file.

WHAT IT BUILDS
--------------
  basket_id  = one shopping trip          (the "order")
  category   = product's commodity        (the "aisle_id" analogue)
  week_of_year, year                      (the seasonal axis)

THE CALENDAR ANCHOR
-------------------
Dunnhumby's WEEK_NO runs 1..102 relative to the study start, NOT the
calendar, and the Kaggle CSVs carry no real dates. So a window cannot
be *labelled* "Christmas". It can still be TESTED: the question is
whether the SAME week-of-year offset shows the same category shift in
both years, which is annual periodicity and is anchor-free. Any claim
made from this data is "annual periodicity at offset X", never a
named holiday.

  week_of_year = ((week - 1) % 52) + 1
  year         = 1 + (week - 1) // 52
    -> year 1 covers week_of_year 1..52
    -> year 2 covers week_of_year 1..50   (data stops at week 102)
  Only week_of_year 1..50 exists in BOTH years, so only that range is
  usable for the replication test. Asserted below.

Reads:  data/dunnhumby/transaction_data.csv
        data/dunnhumby/product.csv
Writes: data/dunnhumby/dj_items.csv
        results/dunnhumby/prep_report.json
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

CATEGORY_FIELD = "commodity_desc"   # confirmed in [3] below

rep = {}
print("=" * 68)
print("DUNNHUMBY PREP - seasonal conditioning")
print("=" * 68)


def find_file(*names):
    """Locate a file case-insensitively - Kaggle mirrors vary."""
    listing = {f.lower(): f for f in os.listdir(DATA_DIR)}
    for n in names:
        if n.lower() in listing:
            return os.path.join(DATA_DIR, listing[n.lower()])
    raise FileNotFoundError(
        f"none of {names} found in {DATA_DIR}. Present: "
        f"{sorted(listing.values())}")


# ══ [1] LOAD ═════════════════════════════════════════════
print("\n[1] LOAD")
tx_path = find_file("transaction_data.csv", "transactions.csv")
pr_path = find_file("product.csv", "products.csv")
tx = pd.read_csv(tx_path)
pr = pd.read_csv(pr_path)
tx.columns = [c.strip().lower() for c in tx.columns]
pr.columns = [c.strip().lower() for c in pr.columns]
print(f"    transactions {len(tx):,} rows x {len(tx.columns)} cols")
print(f"    products     {len(pr):,} rows x {len(pr.columns)} cols")
print(f"    tx cols: {list(tx.columns)}")
print(f"    pr cols: {list(pr.columns)}")

need_tx = ["basket_id", "product_id", "week_no", "household_key",
           "quantity", "store_id"]
missing = [c for c in need_tx if c not in tx.columns]
assert not missing, f"FATAL: transaction_data missing {missing}"
assert "product_id" in pr.columns, "FATAL: product.csv has no product_id"
rep["source_files"] = {"transactions": os.path.basename(tx_path),
                       "products": os.path.basename(pr_path)}
rep["n_transaction_rows"] = int(len(tx))
rep["n_product_rows"] = int(len(pr))

# ══ [2] CATEGORY CARDINALITY - choose, don't assume ══════
print("\n[2] CATEGORY FIELD (cardinality decides; the Instacart")
print("    analogue is 134 aisles)")
cand = [c for c in ("department", "commodity_desc", "sub_commodity_desc")
        if c in pr.columns]
card = {}
for c in cand:
    vals = pr[c].astype(str).str.strip().str.upper()
    card[c] = int(vals.nunique())
    print(f"    {c:<20} {card[c]:>6} distinct")
rep["category_cardinality"] = card
assert CATEGORY_FIELD in cand, \
    f"FATAL: {CATEGORY_FIELD} not in product.csv ({cand})"

# ══ [3] JOIN ═════════════════════════════════════════════
print("\n[3] JOIN transactions -> category")
pr_small = pr[["product_id", CATEGORY_FIELD]].copy()
pr_small[CATEGORY_FIELD] = (pr_small[CATEGORY_FIELD].astype(str)
                            .str.strip().str.upper())
pr_small = pr_small.drop_duplicates("product_id")
n_before = len(tx)
df = tx.merge(pr_small, on="product_id", how="left")
unmapped = int(df[CATEGORY_FIELD].isna().sum())
print(f"    rows {len(df):,} (join preserved rows: "
      f"{len(df) == n_before})")
print(f"    unmapped products: {unmapped:,} "
      f"({100*unmapped/len(df):.3f}%)")
assert len(df) == n_before, "FATAL: join changed row count"
df = df[df[CATEGORY_FIELD].notna()].copy()
# drop non-informative buckets that are not real categories
drop_cats = {"", "NAN", "UNKNOWN", "NO COMMODITY DESCRIPTION"}
n_pre = len(df)
df = df[~df[CATEGORY_FIELD].isin(drop_cats)].copy()
print(f"    dropped {n_pre - len(df):,} rows in "
      f"non-informative categories {sorted(drop_cats)}")
rep["unmapped_rows"] = unmapped
rep["rows_after_join"] = int(len(df))
rep["n_categories_used"] = int(df[CATEGORY_FIELD].nunique())
print(f"    categories in use: {df[CATEGORY_FIELD].nunique()}")

# ══ [4] SEASONAL AXIS ════════════════════════════════════
print("\n[4] SEASONAL AXIS (anchor-free)")
wmin, wmax = int(df["week_no"].min()), int(df["week_no"].max())
print(f"    week_no range {wmin}..{wmax}")
df["week_of_year"] = ((df["week_no"] - 1) % 52) + 1
df["year"] = 1 + (df["week_no"] - 1) // 52
yr_counts = df["year"].value_counts().sort_index()
print(f"    rows per year: {yr_counts.to_dict()}")

woy = df.groupby("year")["week_of_year"].agg(["min", "max", "nunique"])
print(f"    week_of_year coverage by year:\n{woy}")
common = sorted(set(df[df["year"] == 1]["week_of_year"]) &
                set(df[df["year"] == 2]["week_of_year"]))
print(f"    week_of_year present in BOTH years: {len(common)} "
      f"({min(common)}..{max(common)})")
assert len(common) >= 40, (
    f"FATAL: only {len(common)} overlapping weeks - too few for a "
    f"replication test")
rep["week_no_range"] = [wmin, wmax]
rep["rows_per_year"] = {int(k): int(v) for k, v in yr_counts.items()}
rep["common_weeks_of_year"] = [int(min(common)), int(max(common))]
rep["n_common_weeks"] = len(common)

# ══ [5] BASKETS ══════════════════════════════════════════
print("\n[5] BASKETS")
bs = df.groupby("basket_id").size()
print(f"    baskets {len(bs):,}   items {len(df):,}")
print(f"    basket size: mean {bs.mean():.2f}  median "
      f"{int(bs.median())}  p95 {int(bs.quantile(.95))}  "
      f"max {int(bs.max())}")
print(f"    households {df['household_key'].nunique():,}   "
      f"stores {df['store_id'].nunique():,}")
rep["n_baskets"] = int(len(bs))
rep["n_items"] = int(len(df))
rep["basket_size"] = {
    "mean": float(bs.mean()), "median": float(bs.median()),
    "p95": float(bs.quantile(.95)), "max": int(bs.max())}
rep["n_households"] = int(df["household_key"].nunique())
rep["n_stores"] = int(df["store_id"].nunique())

# a basket must belong to exactly one week/year - else the seasonal
# label is ambiguous and the whole axis is unsound
chk = df.groupby("basket_id")[["week_no"]].nunique()
multi = int((chk["week_no"] > 1).sum())
print(f"    baskets spanning >1 week: {multi} "
      f"({'PASS' if multi == 0 else 'FAIL'})")
rep["baskets_spanning_multiple_weeks"] = multi

# ══ [6] PER-WEEK VOLUME (confound visibility) ════════════
print("\n[6] PER-WEEK VOLUME (panel churn / store mix are")
print("    confounds the signal search must control for)")
per_wk = df.groupby(["year", "week_of_year"]).agg(
    rows=("basket_id", "size"),
    baskets=("basket_id", "nunique"),
    households=("household_key", "nunique"),
    stores=("store_id", "nunique")).reset_index()
print(f"    rows/week      min {per_wk['rows'].min():,} "
      f"median {int(per_wk['rows'].median()):,} "
      f"max {per_wk['rows'].max():,}")
print(f"    baskets/week   min {per_wk['baskets'].min():,} "
      f"median {int(per_wk['baskets'].median()):,} "
      f"max {per_wk['baskets'].max():,}")
print(f"    households/wk  min {per_wk['households'].min():,} "
      f"median {int(per_wk['households'].median()):,} "
      f"max {per_wk['households'].max():,}")
rep["per_week"] = {
    "rows": [int(per_wk["rows"].min()), int(per_wk["rows"].median()),
             int(per_wk["rows"].max())],
    "baskets": [int(per_wk["baskets"].min()),
                int(per_wk["baskets"].median()),
                int(per_wk["baskets"].max())],
    "households": [int(per_wk["households"].min()),
                   int(per_wk["households"].median()),
                   int(per_wk["households"].max())],
}
hh_both = (df.groupby("household_key")["year"].nunique() == 2).sum()
print(f"    households present in BOTH years: {hh_both:,} of "
      f"{df['household_key'].nunique():,} "
      f"({100*hh_both/df['household_key'].nunique():.1f}%)")
rep["households_in_both_years"] = int(hh_both)

# ══ [7] WRITE ════════════════════════════════════════════
out = df[["basket_id", "household_key", "store_id", "week_no",
          "week_of_year", "year", "product_id", "quantity",
          CATEGORY_FIELD]].rename(columns={CATEGORY_FIELD: "category"})
out_path = os.path.join(DATA_DIR, "dj_items.csv")
out.to_csv(out_path, index=False)
print(f"\n[7] WROTE {out_path}  ({len(out):,} rows)")
rep["output_rows"] = int(len(out))
rep["category_field"] = CATEGORY_FIELD
rep["NOTE_anchor"] = (
    "WEEK_NO is relative to study start, not the calendar; Kaggle CSVs "
    "carry no real dates. Claims are 'annual periodicity at offset X', "
    "never a named holiday.")

per_wk.to_csv(os.path.join(RESULTS_DIR, "per_week_volume.csv"),
              index=False)
with open(os.path.join(RESULTS_DIR, "prep_report.json"), "w",
          encoding="utf-8") as f:
    json.dump(rep, f, indent=2)
print(f"    -> {os.path.join(RESULTS_DIR, 'prep_report.json')}")
print(f"    -> {os.path.join(RESULTS_DIR, 'per_week_volume.csv')}")
