# Tabular Generative AI for Warehouse Logistics

**Do synthetic orders support real fleet decisions?**

MSc dissertation project, EEEM004, University of Surrey.
Industry context: Locus Robotics.

---

## The question

Warehouse operators need order data to plan a robot fleet, but order data is
commercially sensitive and rarely shared. Synthetic data is the proposed way
around that: train a generative model on real orders, share generated orders
instead.

That only works if the synthetic data supports the **same operational decisions**
as the real data would. The metrics normally used to judge synthetic data measure
whether it *looks* like the real thing, not whether it *behaves* like it.

This project tests the assumption directly.

## The headline result

**A generator can pass every standard check and still be unfit for its purpose.**

CTGAN reaches 0.9205 distributional quality, 89.2% machine-learning utility and a
privacy ratio of 1.08 — and its synthetic data reports that large baskets cost
**1.906 aisles more robot travel** than small ones. In the real data that effect
does not exist.

| Source | Mean travel gap | Draws clearing the measured bar |
|---|---|---|
| Real data | 0.238 aisles | **1 / 10** |
| **CTGAN** | **1.906 aisles** | **10 / 10** |
| TabSyn | 0.381 aisles | **1 / 10** |

No standard metric detects this, and the work proves that a pairwise association
audit is *mathematically incapable* of detecting it — the statistic it rests on is
exactly invariant under the aisle relabellings that move travel by 4.6× the bar.

A second generator, TabSyn, was added to separate model-specific failures from
general ones. It beats CTGAN on nearly every standard metric and does **not**
invent the travel effect — but it reproduces the aisle mix just as poorly. One
failure belongs to the model; the other belongs to the approach.

## The finding nobody set out to look for

Across 596 fleet-simulator episodes, the demand-map error did **not** propagate to
fleet throughput, at a measured sensitivity of ±1.8 steps per delivery. What did
move was order latency — and not because of the items.

Real baskets repeat locations far more than independently assembled ones (**28.7%**
vs **18.3%**), and that structure moved order completion time by **1,041 steps on
26 of 30 draws** — the only comparison in the experiment to fire on a majority. It
replicated on a second dataset at **+43%**, sign-consistent in all 16 comparisons.

**Neither generator models basket membership at all.** Their output has no order
ID. The property that dominates fleet latency is the one they discard.

---

## How results are reported

Three rules, applied everywhere:

1. **Every threshold is measured, never chosen.** Real data is split into matched
   halves and compared against itself, 40 times; the 95th percentile of that null
   distribution is the bar.
2. **Every claim is a fire rate over 10 independent draws**, never a single point
   estimate — a single draw overstated a result twice in this project, and both
   claims were withdrawn.
3. **Every detector is validated against a planted fault** before it is trusted.

An earlier version of this pipeline reported far better numbers and was found by
its own audit to be invalid. All of it was withdrawn and rebuilt. See
[`legacy_v1_v2/`](legacy_v1_v2/) — the defective code is retained deliberately as
the evidence behind that account.

---

## Repository layout

| Path | Contents |
|---|---|
| `prepare_data_v3.py`, `create_eval_set_v3.py`, `train_v3.py`, `evaluate_v3.py` | The v3 CTGAN pipeline — clean schema, provably disjoint split |
| `audit_model_artifact.py`, `model_comparison_v3.py`, `association_audit.py` | Model audit and baselines |
| `validate_demand_geometry.py`, `test_real_fleet_effect.py` | Demand geometry, and the real-data-only anchor |
| `tabsyn_*.py` | TabSyn replication, scored by copies of the same scoring code |
| `rtf_*.py` | REaLTabFormer — assessed and rejected (memorisation) |
| `rware_*.py`, `rwstyle_*.py`, `rwseason_*.py` | RWARE fleet-simulator bridge and experiments |
| `dunnhumby_*.py` | Second dataset — seasonal and location conditioning |
| `audit_all_results.py` | Re-reads every result file and asserts the headline claims |
| `results/` | Result artifacts and per-experiment findings write-ups |
| `data/*.json` | The measured numbers behind every figure quoted above |
| `legacy_v1_v2/` | **Withdrawn** v1/v2 work, kept as audit evidence |

Detailed write-ups live alongside their experiments:
[`results/rtf/rtf_findings.md`](results/rtf/rtf_findings.md),
[`results/rware/rware_findings.md`](results/rware/rware_findings.md),
[`results/rwstyle/rwstyle_findings.md`](results/rwstyle/rwstyle_findings.md),
[`results/rwseason/rwseason_findings.md`](results/rwseason/rwseason_findings.md),
[`results/dunnhumby/dunnhumby_findings.md`](results/dunnhumby/dunnhumby_findings.md).

## Results summary

| Metric | CTGAN | TabSyn |
|---|---|---|
| Distributional quality | 0.9205 | **0.9448** |
| ML efficacy (matched 10k) | 89.2% | 89.7% |
| Privacy — DCR ratio | 1.124 | 1.025 |
| Problem-column marginals | baseline | **3–4× more accurate** |
| Invents weak associations | +0.0585 | −0.0006 (does not) |
| **Aisle-mix error** | **fails** | **fails** |
| **Travel fabrication** | **10/10 draws** | 1/10 (matches reality) |

Correlation preservation is reported in the dissertation but is close to
meaningless on this data: the six columns are nearly uncorrelated in reality
(strongest pair ≈ 0.105), so a generator emitting zero correlation everywhere
scores ~97%. ML efficacy above the measured do-nothing floor (~86%) is the
discriminating metric.

---

## Reproducing this

**Data is not distributed.** Both datasets are licensed and must be downloaded
from Kaggle:

- [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) → `data/`
- [Dunnhumby — The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) → `data/dunnhumby/`

Each generator needs its **own environment**, because their dependencies conflict:

| Environment | For | Notes |
|---|---|---|
| `.venv` | v3 CTGAN pipeline, analysis | Python 3.10, SDV |
| `.venv_tabsyn` | TabSyn | torch 2.11 + CUDA 12.8; TabSyn cloned separately into `tabsyn_repo/` |
| `.venv_rtf` | REaLTabFormer | transformers pinned to 4.57.6 — RTF 0.2.4 breaks on 5.x |
| `.venv_rware` | Fleet simulator | rware 2.0.0 + gymnasium 1.3.0, **CPU only** |

Then, in order:

```bash
python prepare_data_v3.py        # 6 clean columns, no derived redundancy
python create_eval_set_v3.py     # provably disjoint split
python train_v3.py               # 300 epochs, conditioned on basket size
python evaluate_v3.py            # standard metrics
python validate_demand_geometry.py
python test_real_fleet_effect.py --selftest   # fixture check, then run
python audit_all_results.py      # asserts every headline claim
```

Note that CTGAN sampling only reproduces byte-for-byte if the model is reloaded
fresh **and** `torch.manual_seed` is set — re-seeding `random` and `numpy` alone is
not enough, because sampling advances RNG state held inside the loaded model. The
seed behind the recorded synthetic CSV was never captured, so that file remains the
artifact of record.

---

## Author

Nyi Nyi Myo Zin · MSc, University of Surrey
Supervisor: Dr Simon Hadfield
