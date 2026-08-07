# TabSyn Bench Contract (approved by Brian, Jul 19 2026)

Adding TabSyn as a second generator to the validated v3 test bench. Protection
rules in the project notes apply: all existing files READ-ONLY; new work only in
`tabsyn_*.py`, `tabsyn_repo\`, `data\tabsyn\`, `results\tabsyn\`, `.venv_tabsyn\`.

## Time-box (hard)
- TabSyn not training successfully within 4 days of effort → STOP, fall back to TabDDPM.
- TabDDPM fails within 3 more days → fall back to TVAE (SDV, own cloned env).
- Do not let environment fights eat the schedule.

## Approved environment plan (Jul 19)
- Clone `amazon-science/tabsyn` → `tabsyn_repo\` (repo pins torch 2.0.1+cu117 —
  CANNOT run on RTX 5080/sm_120; first stable torch with sm_120 is 2.7.0+cu128).
- New env `.venv_tabsyn` from Python 3.10; `torch==2.11.0` cu128 wheel — the exact
  build proven working in `.venv` on this GPU.
- Minimal deps only (train+sample path). Skip transformers/peft/sdmetrics/kaggle/
  synthcity/DGL/PyG — those serve their baselines & their eval, which we don't use.
- If numpy-2 breakage: pin `numpy<2` in `.venv_tabsyn` only.

## Training contract
- Train on `data\v3_train.csv`, same 6 columns, PLUS `order_size_grp` as a 7th
  categorical column (approved), RECOMPUTED with screened buckets small ≤10 /
  large ≥14 — never the stored 33/67 column (project-notes caveat 4).
- TabSyn has no conditional generation → train unconditional; write-up notes
  "conditional (CTGAN) vs learned-as-column (TabSyn)".

## Scoring contract — five standing-rule additions (Brian, Jul 19)
1. **Known-answer verification first.** The `tabsyn_*` scoring copies must first
   reproduce CTGAN's recorded numbers on the EXISTING `data\synthetic_v3.csv`
   (quality 0.9205, marginal errors, demand_geometry.json verdicts) before they
   are trusted on TabSyn. A copy that can't reproduce the known result is broken.
2. **Matched sample sizes everywhere.** Same synthetic row count as the CTGAN
   scoring file; same held-out real files (`v3_eval.csv` / `v3_compare.csv` as
   each original script used); demand geometry uses the SAME recorded noise bars
   from `demand_geometry.json` with the same size-matched N — never re-derived.
3. **Report generated `order_size_grp` proportions vs training proportions**
   (under-generation of small/large ⇒ travel-gap comparison loses power).
4. **Match the stability protocol.** ML efficacy via the identical 5-seed
   `test_marginal_impact.py` protocol (CTGAN: 89.2% ± 0.7) — mean-vs-mean.
   Confirmed by Brian (Jul 19): 5 INDEPENDENT TabSyn samples, each scored with
   the same ML-efficacy procedure as test_marginal_impact.py (matched training
   sizes, same classifier settings), so the comparison is mean-vs-mean under
   the identical protocol.
5. **Artifact of record:** `data\tabsyn\synthetic_tabsyn.csv` — the saved CSV is
   citable, not re-sampling (sampling not seed-reproducible, per model audit).

## Scoring scripts
Verbatim copies with only I/O paths changed (and WITHOUT evaluate_v3.py's
hard-coded `'quality': 0.9205`), diffed against originals:
`tabsyn_evaluate.py`, `tabsyn_audit_model.py` (adapted — TabSyn artifact differs),
`tabsyn_association_audit.py`, `tabsyn_demand_geometry.py`, `tabsyn_marginal_impact.py`.
Outputs → `results\tabsyn\`.

## Deliverable
Comparison table CTGAN v3 vs TabSyn: quality, ML efficacy (5-seed mean±spread),
DCR, marginal errors, association inflation (weak vs strong pairs), demand-geometry
verdicts incl. small-vs-large travel gap. Headline question: does TabSyn ALSO
fabricate the travel effect (CTGAN: 2.36-aisle gap vs 0.19 real), or not?
