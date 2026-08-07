# Seasonally-conditioned demand through the fleet simulator — findings

**Run of record:** `results/rwseason/comparison.json` — 96 episodes
(2 mappings × 8 draws × 3 sources × 2 periods), **0 deadlocks**, 1.50 h, 2026-07-31.
All-metric analysis: `results/rwseason/analysis_all_metrics.json`.
Power fixture: `results/rwseason/fixtures.json`. Raw records: `runs.jsonl`.

This is the brief's clause *"these differently conditioned data patterns will be fed to
the simulator to test the performance of the fleet"* for the **date** conditioning axis
(Dunnhumby weeks 35–40 vs baseline).

---

## The two results

1. **Neither real nor synthetic demand produces a detectable fleet-level seasonal
   effect** — at a measured sensitivity of **±2.45 steps/delivery**. The seasonal
   category shift is real (TVD 0.086, replicated out-of-sample) but **washes out** at
   the fleet.
2. **The basket-structure finding replicates on a second dataset**: real baskets take
   **+948 steps longer per order** than i.i.d.-reassembled ones — **+43.3%**,
   sign-consistent **16/16 in both periods**.

---

## Why the real anchor ran first

Design deliberately mirrors `test_real_fleet_effect.py`: establish whether the effect
exists **in reality** before asking whether a generator reproduces it. If reality shows
no fleet-level seasonal effect, then a generator's failure to reproduce the seasonal
shift has **no operational consequence** — and that is itself the result, independent of
how good the generator is.

**That is exactly what happened, and it defuses the Dunnhumby conditional verdict
operationally.** The conditional TabSyn under-states the seasonal shift (Part 2b of
`dunnhumby_findings.md`), but the shift does not move fleet throughput in the first
place.

---

## Setup

Dunnhumby has **300 categories** and the medium RWARE layout only 144 storage cells, so
this uses `shelf_columns=7, column_height=8, shelf_rows=3` → **320 shelves**, permitting
a genuine **bijection** (asserted). `rware_bridge.py` layout constants are **rebound at
runtime, never edited** — `rware_*` stays read-only.

| stream | what it is |
|---|---|
| **A_real** | real Dunnhumby baskets, intact — ground truth |
| **B_pool** | the same real items, re-assembled i.i.d. — **assembly control** |
| **C_syn** | conditional-TabSyn items, same rule |

Each run twice: filled from **window** items and from **baseline** items. The contrast
is (window − baseline) **within** each source, off one shared basket-size schedule, so
basket size cannot differ between the conditions being compared. Item counts identical
across sources — asserted every draw.

`synthetic_season.csv` has no `basket_id` (TabSyn emits independent item rows and does
not model basket membership), so an assembly rule is unavoidable — which is precisely
why B exists.

**Power fixture:** planted geometry fault (all items routed to the 60 farthest aisles)
shifts **+2.4531 steps/delivery, fires 4/4** against a measured bar of **0.2384** — a
~10× margin.

---

## Result 1 — the seasonal shift does not reach the fleet

steps/delivery, window − baseline, n = 16 (8 draws × 2 mappings):

| source | mean | sd | fires vs 0.2384 |
|---|---|---|---|
| A real baskets | +0.0924 | 0.2064 | 7/16 |
| B real, reassembled | +0.0456 | 0.1515 | **3/16** |
| C synthetic (TabSyn) | +0.0173 | 0.1771 | **3/16** |

**Nothing fires on a majority.** Synthetic fires at *exactly* the same rate as its real
control (3/16) with a mean within 0.03 steps/delivery — **no fabrication and no miss**,
because there is nothing at fleet level to reproduce.

> State it as **"no detectable fleet-level seasonal effect at a measured sensitivity of
> ±2.45 steps/delivery"**, never as "no effect". The generators produce ±0.02–0.09,
> i.e. 26–140× smaller than what the harness demonstrably resolves.

This is the same shape as the Jul 29 Instacart verdict (demand-geometry mismatch washes
out at ±1.8 sensitivity) — now on a second dataset and a different conditioning axis.

---

## Result 2 — basket structure moves latency; item content does not

`mean_order_completion`, A_real − B_pool:

| period | mean | same sign | % of B |
|---|---|---|---|
| window | **+943.2** | **16/16** | +42.4% |
| baseline | **+953.6** | **16/16** | +44.2% |

**Mechanism, measured on this dataset:** real baskets repeat a category far more than
i.i.d. draws — **24.7% vs 10.1%** within-basket duplicate rate. Under one-category-
one-shelf a repeat cannot be requested twice concurrently, so duplicates **serialise**.

### Cross-dataset replication

| | duplicate rate (real vs i.i.d.) | latency effect A−B |
|---|---|---|
| **Dunnhumby** — 300 categories, 320 shelves | **24.7% vs 10.1%** (2.4×) | **+948 (+43.3%)**, 16/16 |
| Instacart — 134 aisles, 144 shelves | 28.7% vs 18.3% (1.6×) | +1041 (+14.6%), 26/30 |

The effect now replicates across **two datasets, two category spaces and two warehouse
layouts**, with the same mechanism and a *larger* relative magnitude on Dunnhumby.
**Neither CTGAN nor TabSyn models basket membership at all** — the property that
dominates fleet latency is precisely the one they omit. This is direct quantitative
support for the whole-basket sequence-model future work.

> **⚠ EVIDENTIAL STATUS DIFFERS FROM RESULT 1.** rwseason has **no independent-redraw
> stream**, so there is **no measured bar for the assembly contrast**. It is reported as
> an effect size with sign consistency, **never as a fire rate**. The Jul 29 Instacart
> run is the barred version of the same comparison (26/30 at 1.44× its bar). Treat this
> as replication of a previously-barred result, not as a new barred result.

**Caveat carried over:** a real warehouse picks two same-category items in one visit
while this harness serialises them, so A−B is an **upper bound** on the cost of losing
basket coherence, not an estimate.

---

## Stated limitations

1. **The category→shelf map is a modeling abstraction**, not a real store layout.
   Validity rests on it being **identical** across real and synthetic — hence two seeded
   mappings, since travel geometry is relabelling-sensitive.
2. **No measured bar for the assembly contrast** (see above).
3. **Baskets capped at 40 items**, identically in every condition, because the 8-slot
   queue turns very long baskets into pure serialisation.
4. **Scale:** 150 orders/run, 8 draws, 2 mappings. Bars are scale-dependent — the
   comparison is gated to refuse a fixture measured at a different scale. **Do not pool
   these numbers with the rwstyle or Jul 29 RWARE runs.**
5. `season_period` is TabSyn's binclass **target**, returned as integer codes; it is
   mapped back to labels via the prep report's own `target_classes` with assertions,
   the same trap that would otherwise silently empty every synthetic pool.
