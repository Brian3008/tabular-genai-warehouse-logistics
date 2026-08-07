# RWARE fleet bridge — findings

**Run of record:** `results/rware/comparison.json` (30 records = 10 draws × 3
mappings; 180 episodes; 0 deadlocks; 12,100 s). Harness gates:
`results/rware/fixtures.json`, all five PASS, certified against the final
bridge code. Per-episode records: `results/rware/runs.jsonl`.

---

## The question

Prior work in this project measured a **demand-geometry mismatch** between real
and synthetic orders: synthetic misrepresents aisle mix (TVD fails for both
generators), CTGAN under-concentrates demand (Gini) and robustly fabricates a
travel-cost effect that reality does not have (fires 100% of draws vs a 0.64
noise bar, where reality fires 10%); TabSyn does not robustly fabricate travel.

**Does that mismatch propagate to fleet-level fulfillment, or wash out?**

---

## Verdict

**It washes out.** At fleet level, replacing real item content with synthetic
item content produces **no effect this harness can detect**, for either
generator, on either throughput metric — while the *same harness* resolves
planted geometry faults on every single draw.

The failure mode that matters for operational fleet planning is therefore
**not** visible in fleet throughput. That is a genuine negative result, and it
*bounds* the project's primary contribution rather than contradicting it.

---

## 1. Headline — throughput metrics

Four streams off one schedule (identical orders, sizes, item counts).
Comparison is **B vs C** and **B vs D**; **A** anchors ground truth.

| Stream | | steps/delivery | signed Δ vs B | fires (pooled bar) | fires (per-mapping bars) |
|---|---|---|---|---|---|
| **A** | real baskets, intact | 6.8635 | −0.0648 | 3/30 (10%) | 10/30 (33%) |
| **B** | real items, i.i.d. — *control* | 6.9283 | — | — | — |
| **C** | **TabSyn** | 6.9445 | **+0.0163** | **0/30 (0%)** | **1/30 (3%)** |
| **D** | **CTGAN** | 6.9545 | **+0.0262** | **0/30 (0%)** | **5/30 (17%)** |

Pooled bar **0.4364** (95th pct of |B − same-pool redraw|, n=30, null mean
0.1577). Per-mapping bars are tighter for two of three mappings (0.1562 /
0.4892 / 0.2108), so the right-hand column is the **stricter** test.

**Throughput per 1k steps** agrees: C fires 0/30, D fires 1/30, A fires 4/30
against a bar of 7.2619.

Under the repeated-draw majority rule, **nothing fires on a majority of draws**.
Even against the tightest per-mapping bar (map 11, bar 0.1562), TabSyn fires
1/10 and CTGAN 2/10.

> **Two bars, stated explicitly to avoid mixing them.** The pooled bar (0.4364)
> is inflated by mapping 22, whose null is noisier than the other two. The
> per-mapping bars treat each mapping as its own experiment. Both are reported
> above; neither produces a majority fire rate for either generator. Do not
> quote a pooled-bar number alongside a per-mapping number as if they were the
> same analysis.

---

## 2. The power evidence — this is a *measured* negative, not a null harness

A null result is only meaningful with the sensitivity attached. From
`fixtures.json` (F4, 120 orders, bar 0.6147, baseline 6.9341):

| Planted fault | Loaded travel cost | signed Δ | fires |
|---|---|---|---|
| **far40** — demand on the 40 aisles farthest from the goal | 24.6 steps | **+1.7380** | **8/8 (100%)** |
| **near40** — demand on the 40 nearest | 10.7 steps | **−1.8705** | **8/8 (100%)** |
| same-distribution redraw (false-positive check) | — | — | **0/8 (0%)** |

The harness resolves a **±1.8 steps/delivery** geometry effect on *every* draw,
in the correct direction, with a 0% false-positive rate. The generators produce
**+0.016 (TabSyn)** and **+0.026 (CTGAN)** — roughly **70–110× smaller** than
the planted effects, and **4–6% of the bar**.

**Correct statement of the result:** *no detectable fleet-throughput effect at a
measured sensitivity of ±1.8 steps/delivery (100% detection) — the generators'
effects are ~2% of that magnitude.* It is **not** correct to say "no effect".

---

## 3. The finding that was NOT expected — basket structure moves latency, item content does not

Stream B exists to price the assembly rule. It earned its place:

| Stream | mean order completion (steps) | signed Δ vs B | fires (bar 721.4) |
|---|---|---|---|
| **A** real baskets, intact | 8152.80 | **+1041.32** | **26/30 (87%)** |
| **B** real items, i.i.d. | 7111.49 | — | — |
| **C** TabSyn | 7123.73 | +12.25 | 0/30 (0%) |
| **D** CTGAN | 6754.70 | −356.79 | 5/30 (17%) |

**A vs B fires 26/30 (87%) at +1041 steps — 14.6% of B's mean, 1.44× the bar,
and in the same direction on every draw.** This is the one comparison in the
whole experiment that fires on a majority.

So: **swapping real item content for synthetic item content changes nothing
detectable; destroying real *basket structure* changes order latency a lot.**

**Mechanism, measured.** Real baskets repeat aisles far more than i.i.d. draws
do — duplicate-item rate **A 28.7%** vs **B 18.3%** (C 18.3%, D 14.5%). Under
the one-aisle-one-shelf abstraction a repeated aisle cannot be requested twice
concurrently, so duplicates are deferred and re-issued sequentially, stretching
completion time. Within-basket correlation is the driver.

**Why this matters for the thesis.** CTGAN and TabSyn emit independent item rows
and **do not model basket membership at all** — the property that turns out to
dominate fleet latency is the one neither generator represents. This is direct
quantitative support for the flagged future-work item (a sequence model
generating a whole basket as a unit) and it explains why that direction is worth
pursuing even though REaLTabFormer failed on memorisation.

**Caveat, load-bearing:** part of the A−B latency gap is an artifact of the
abstraction itself. A real warehouse would pick two items from one aisle in a
single visit; our harness serialises them. The gap is therefore an **upper
bound** on the true cost of losing basket coherence, not an estimate of it.
State it that way.

---

## 4. Mapping robustness — not an artifact of one layout

Travel geometry is relabelling-sensitive (`tabsyn_conditional_geometry.py`
measured the travel gap ranging 2.94 aisles = 4.6× bar across aisle
relabellings), so a single aisle→shelf map could manufacture or erase an effect.
Three independent seeded bijections were used, fingerprints
`8023f8a186410296` / `2ad8c3a7443c9194` / `5aab2c4593e16e5e` — the same maps the
gates were certified on.

| Mapping | bar | A fires | C fires | D fires |
|---|---|---|---|---|
| 11 | 0.1562 | 5/10 | 1/10 | 2/10 |
| 22 | 0.4892 | 2/10 | 0/10 | 0/10 |
| 33 | 0.2108 | 3/10 | 0/10 | 3/10 |

Neither generator reaches a majority under **any** mapping. The conclusion holds
across all three, so it is not a mapping artifact. Note A's deviation from B
exceeds both generators' under every mapping — consistent with §3.

---

## 5. Stated limitations

1. **The aisle→shelf map is a modeling abstraction, not a real warehouse
   layout.** Validity rests on the map being IDENTICAL across real and synthetic
   runs, not on physical realism — the same standing as the travel-distance
   proxy. Mitigated, not removed, by the three-mapping sweep.
2. **Basket structure for B/C/D is injected, not generated.** The generators
   have no `order_id`; baskets are assembled i.i.d. against a real size
   schedule. **Basket-size realism is not tested and cannot be** with these
   generators. §3 measures the cost of that injection rather than hiding it.
3. **One-aisle-one-shelf serialises duplicate picks**, inflating the A−B latency
   gap relative to a real warehouse (see §3 caveat).
4. **One fleet configuration** — medium layout, 8 agents, queue 8, scripted
   nearest-assignment policy. No fleet-size sweep and no RL. A different
   fleet size or policy could sit at a different sensitivity.
5. **Sensitivity is ±1.8 steps/delivery at 300 orders/run.** Effects smaller
   than that are invisible to this experiment. The generators' effects are ~2%
   of it, so a much larger experiment would be needed to resolve them — and it
   is not obvious that an effect that small is operationally meaningful.
6. **Fleet-styles comparison (random vs nearest assignment across small-heavy vs
   large-heavy streams) was NOT run** — time-boxed out. Remains future work.

---

## 6. One-line summary for the write-up

> Synthetic order data does **not** mislead fleet-throughput conclusions for
> either generator: the measured demand-geometry mismatch — including CTGAN's
> robustly fabricated travel effect — **washes out** at fleet level, with no
> detectable difference at a sensitivity that resolves planted ±1.8
> steps/delivery geometry faults on 100% of draws. What *does* move fleet
> performance is **basket structure**: real baskets differ from i.i.d.
> re-assembly of the very same items by +1041 steps of order latency (fires
> 87%), and basket membership is precisely what these generators do not model.
