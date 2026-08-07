# Fleet management styles per demand pattern — findings

**Run of record:** `results/rwstyle/comparison.json` — 320 episodes
(2 mappings × 8 draws × 2 demand mixes × 2 policies × 5 streams), **0 deadlocks**,
140.8 min, 2026-08-01. Power fixture: `results/rwstyle/fixtures.json`.
Raw per-episode metrics: `results/rwstyle/runs.jsonl` (flushed every episode).

Closes two open items at once:
* **Dr. Hadfield's standing suggestion** — "explore fleet management styles per demand
  pattern" — which was time-boxed out of the Jul 29 RWARE run (`rware_findings.md`
  item 6: *"NOT run … remains future work"*).
* **The brief's clause** *"these differently conditioned data patterns will be fed to
  the simulator to test the performance of the fleet."* Basket size is the axis CTGAN
  was actually conditioned on (`order_size_grp`), so a small-heavy vs large-heavy
  stream **is** a conditioned pattern.

---

## The three results

1. **`nearest` beats `random` by ~0.7 steps/delivery (−8 to −9%), on every draw of
   every stream — 32/32 sign-consistent.** Fleet policy matters.
2. **Both generators reproduce that policy effect to within 0.06 steps/delivery.**
   This is the first operational question in this project where synthetic data
   **succeeds** — a decision made on synthetic data is the same decision.
3. **Whether the best policy depends on demand mix cannot be answered at this scale.**
   The interaction is underpowered by ~2.4×; this is a measured limitation, not a null
   result.

---

## Design

| axis | levels |
|---|---|
| policy | `nearest`, `random` — both already implemented at `rware_bridge.py:357-374` and **never exercised** (Jul 29 was `nearest` only) |
| demand mix | small-heavy (80/20), large-heavy (20/80) |
| stream | A real baskets · B real items reassembled i.i.d. (**assembly control**) · C TabSyn · D CTGAN · NULL (independent real redraw) |

One schedule per (mapping, draw, mix) cell, reused verbatim by all five streams, so
order count, per-order sizes and total item count are identical — asserted every draw.
`rware_bridge.py` is imported unmodified; `rware_*` and `results/rware/` remain
read-only.

---

## POWER — measured before any verdict

A null is uninterpretable without sensitivity; `rware_findings.md` is citable only
because its F4 established ±1.8 steps/delivery. This harness refuses to run the
comparison until its own fixture passes **at the same orders/run scale** (enforced —
bars are scale-dependent).

**All planted faults are CONCURRENCY-PRESERVING map relabellings.** Demand is untouched;
only which aisle sits on which shelf moves. Faults were **chosen from a closed-form
power calculation**, not guessed — expected loaded travel per pick,
Σ_a freq(a)·cost(map(a)):

| map | E[cost]/pick | large − small | shift |
|---|---|---|---|
| baseline (seeded) | 18.031 | −0.083 | — |
| **adversarial** (most-demanded → farthest) | 24.744 | +0.299 | **+6.71** |
| **benign** (most-demanded → nearest) | 10.563 | −0.300 | **−7.47** |
| size_skew (large-skewed → farthest) | 20.068 | **+1.384** | +2.04 |

Measured result:

| fault | Δ steps/delivery | fires |
|---|---|---|
| adversarial | **+1.7841** | 4/4 |
| benign | **−2.2580** | 4/4 |
| size_skew | +1.0103 | 4/4 |

against a bar of **0.2199** — an 8–10× margin, correct directions.
**Geometry sensitivity ±2.02 steps/delivery**, comparable to the Jul 29 F4's ±1.8.

### Two fixture designs were rejected on evidence before this one

* **v1** routed large orders onto the 40 farthest aisles. That moves travel distance
  **and concurrency** at once (48 large baskets contending over 40 shelves behind an
  8-slot queue) — the *same defect* that archived the original RWARE F4 — and it made
  episodes crawl.
* **v2** was a size-skew-only relabelling: it moved expected travel just +2.04
  steps/pick and shifted steps/delivery by only +0.24 against a 0.31 bar, i.e.
  **underpowered — caught by the closed-form calculation, not after hours of episodes.**

### A pass rule that was mathematically unsatisfiable

The gate originally also required **zero** null exceedances of a bar defined as the
95th percentile **of that same null sample**. For n ≤ 20, numpy's interpolated 95th
percentile lies strictly below the maximum, so **≥1 exceedance is guaranteed**. The
rule was circular and produced a spurious FAIL on 2026-07-31 22:54 that skipped the
whole comparison. The measurements were never in question; the rule was corrected and
PASS recomputed from the saved data. **A genuine false-positive control needs an
INDEPENDENT redraw set** (as the Jul 29 F4 had, 0/8) — this harness does not have one,
and that is a stated limitation.

---

## Result 1 — policy matters, and it is unambiguous

Policy effect = `nearest − random` (negative is better), per draw, n = 32:

| stream | mean | sd | same sign | % of `random` |
|---|---|---|---|---|
| A real baskets | **−0.7045** | 0.2323 | **32/32** | −9.1% |
| B real, reassembled | **−0.6984** | 0.1727 | **32/32** | −8.8% |
| C TabSyn | **−0.6394** | 0.2099 | **32/32** | −8.1% |
| D CTGAN | **−0.6845** | 0.1760 | **32/32** | −8.7% |
| NULL (real redraw) | −0.6506 | 0.2028 | **32/32** | −8.1% |

Nearest-assignment saves ~0.7 steps per delivery, about 9% of the random-policy cost,
and it does so on **every single draw of every stream and both demand mixes**.

---

## Result 2 — synthetic data supports the correct fleet-policy decision

The generators agree with the real streams to **within 0.06 steps/delivery** — under
9% of the effect they are estimating. Per-cell fidelity against the B control
(fires / 16):

| cell | A − B | C − B (TabSyn) | D − B (CTGAN) |
|---|---|---|---|
| small-heavy, nearest | −0.2197 (3/16) | +0.0349 (2/16) | −0.0011 (0/16) |
| small-heavy, random | −0.1508 (2/16) | −0.0429 (0/16) | +0.0029 (0/16) |
| large-heavy, nearest | −0.1182 (1/16) | +0.0445 (0/16) | −0.1620 (1/16) |
| large-heavy, random | −0.1750 (3/16) | +0.0042 (1/16) | −0.1940 (0/16) |

**Nothing fires on a majority in any cell, under either policy or either demand mix.**
No fabrication, no miss.

**This is the project's first positive operational result for synthetic data.** It
should be stated alongside the CTGAN travel-fabrication finding, not instead of it: the
same generator that invents a travel-cost effect (fires 100% vs a 0.64 bar where
reality fires 10%) nonetheless supports the correct **policy** decision. Which
operational question you ask determines whether the synthetic data is fit for purpose.

**Two bars, never mix them.** Pooled |B − independent redraw| = **0.5568**; per-cell
bars are 0.6916 / 0.5290 / 0.4861 / 0.5193, so the pooled bar is *looser* than two of
the four. Fire rates above are per-cell.

---

## Result 3 — the interaction is UNDERPOWERED, and that is the honest finding

*Does the best policy depend on the demand mix?* Interaction
I = (nearest − random | large-heavy) − (nearest − random | small-heavy):

| stream | I | fires vs 0.4187 |
|---|---|---|
| A real baskets | +0.1282 | 4/16 |
| B real, reassembled | +0.0026 | 0/16 |
| C TabSyn | −0.0349 | 1/16 |
| D CTGAN | +0.0386 | 2/16 |

Nothing fires on a majority — **but this must not be read as "no interaction".** The
fixture already showed the estimator cannot resolve one: the `size_skew` fault moves
the closed-form large-minus-small differential by **+1.47 steps/pick**, yet the
measured interaction shifted only **+0.178** against a **0.4187** bar, and the planted
interaction fired 0/4, 0/4, 1/4.

**Interaction sensitivity is ±0.178 against a 0.419 bar — a factor ~2.4 short.** A
difference of differences carries roughly 4× the variance of a main effect, so
resolving it needs on the order of **5× more episodes**. The correct statement is:

> **The policy × demand-mix interaction is not resolvable at this scale.** Main effects
> are well powered (±2.02); the interaction is not, and no claim either way is
> supported.

---

## Stated limitations

1. **The aisle→shelf map is a modeling abstraction, not a real warehouse layout.**
   Validity rests on it being **identical** across real and synthetic, not on physical
   realism. Two seeded mappings were used (fingerprints reproduce the Jul 29 certified
   `8023f8a186410296` for seed 11); travel geometry is relabelling-sensitive, so a
   single mapping would not be sufficient.
2. **No independent false-positive control**, per the pass-rule note above.
3. **Basket-size realism is not tested and cannot be** — the generators do not model
   basket membership, so the assembly rule is an injected assumption. That is exactly
   what stream B prices, and A−B is consistently the largest contrast in the table.
4. **Scale:** 100 orders/run, 8 draws, 2 mappings. The Jul 29 bench used 300 orders ×
   10 draws × 3 mappings; **do not pool the two runs' numbers** — different scale means
   different bars.
