# Legacy v1 / v2 — withdrawn work, retained as evidence

**Nothing in this folder is part of the project's results. Every number these
scripts produced has been formally withdrawn.**

They are kept deliberately. The dissertation's methodology chapter documents an
audit that found five defects in this pipeline, and these files are the evidence
behind that account. Deleting them would remove the ability to verify the claim.

Do not build on anything here. The current pipeline is in the repository root.

---

## What was withdrawn

The v2 pipeline reported quality 93.47%, ML efficacy 100.3%, correlation 97.6%
and a privacy ratio of 1.16. An audit found all four to be invalid.

| # | Defect | Where to see it |
|---|---|---|
| 1 | Train and evaluation sets overlapped | `create_eval_set.py` |
| 2 | Synthetic `order_hour_of_day` was overwritten with **real** hours before scoring, and three further columns were recomputed from them | `train_ctgan_v2.py`, `evaluate_synthetic.py` |
| 3 | The "random shuffle" baseline never shuffled | `shuffling_baseline.py:20-23` |
| 4 | Seasonal conditioning was circular — it conditioned on `aisle_popularity`, a direct lookup of the target `aisle_id`, giving a fake 98.7% separability | `conditional_ctgan.py`, `generate_seasonal.py` |
| 5 | 7 of 13 training columns were deterministic functions of the other 6 | `prepare_data_v2.py` |

### Defect 3 is the clearest one to inspect

`shuffling_baseline.py:20-23` reads:

```python
# Shuffle each column independently
for col in ['aisle_id', 'department_id', 'order_dow', 'order_hour_of_day']:
    baseline[col] = sample[col].sample(frac=1, random_state=42).values
```

The comment says each column is shuffled independently. It is not. `random_state=42`
is fixed, so **every column receives the identical permutation**. Rows stay intact,
merely reordered. The "baseline" was therefore the real data compared against
itself, which is why it scored 98.1% and appeared to beat the generator.

The corrected baseline is in `model_comparison_v3.py` in the repository root. With
a genuine per-column shuffle it scores 0.780 on ML efficacy against CTGAN's 0.880 —
the opposite of the original conclusion.

---

## Also here, and why

- **`smart_simulator.py`** — a warehouse simulator in which the zoned strategy wins
  by construction. Abandoned as rigged; replaced by direct demand-geometry
  measurement and later by the RWARE bridge.
- **`warehouse_congestion.py`, `debug_congestion.py`** — an A\* congestion simulator
  that deadlocked and was abandoned. The same failure mode (idle agents parking as
  walls) was later reproduced and fixed properly in the RWARE harness.
- **`*_10percent*.py`** — small-scale exploratory runs. `privacy_10percent.py` is
  an empty file.
- **`view_*.py`, `explore_data.py`, `check_*.py`** — ad-hoc inspection scripts.
- **`results/`** — the 23 withdrawn figures. They are the *old* numbers and must not
  be quoted.

---

## A note on running these

Paths inside these scripts assume the repository root as the working directory,
which is where they originally sat. They are preserved for reading, not for
re-running, and several depend on data files that are not distributed.
