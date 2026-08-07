# Dunnhumby seasonal conditioning — findings

**Requirement 9, seasonal/date part.** Second dataset, kept fully separate from the
Instacart work. Sources of record:
`results/dunnhumby/signal_search.json` (the gate),
`results/dunnhumby/conditional_test.json` (the verdict),
`results/dunnhumby/fixtures.json` (detector fixtures),
`results/dunnhumby/conditional_known_answer.json` (known-answer gate),
`results/dunnhumby/vae_train_state.json` (training state).
Data: Kaggle `frtgnn/dunnhumby-the-complete-journey`
(`transaction_data.csv` + `product.csv` only).

**Location conditioning was NOT attempted.** No public order dataset carries real
geography, and inventing one would repeat the kind of fabricated axis this project
already withdrew once. Documented as future work.

---

## The results

1. **A real annual seasonal signal exists** in Dunnhumby, replicates out-of-sample in
   a second year, and **survives a basket-size control**. It is also **not** store-mix
   drift: store composition induces only **2.7%** of it (Part 1c).
2. **Every 6-week window is distinguishable from every other**, and a shift profile
   **does not transfer** between windows (index 0.042) — so "distinguishable" is not
   itself evidence of seasonality, and conditioning must be validated per-period
   (Part 1b).
3. **Location conditioning is testable after all**, via `store_id`, and **carries ~2.2×
   the signal that season does** — it also does not transfer between sites (Part 1c).
4. **The conditional TabSyn under-states the shift** — verdict **MISSES** at full
   training budget, but it now gets the **direction largely right** (Spearman 0.6681 =
   84% of the real year-on-year ceiling, 7/10 top risers) and fails on **magnitude**
   (81% of real, 0/10 above the bar). Its 300-category marginal is **2.9×** the
   measured floor (Part 2b).

**Cite v2, not v1.** The under-trained v1 model gave 9× marginal error and near-chance
direction; those numbers are superseded.

---

## Part 1 — THE GATE: a real, replicated seasonal signal

Frozen window **weeks 35–40**, discovered on **year 1 only**; year 2 untouched during
selection (the project's disjointness discipline moved onto the seasonal axis).

| | observed TVD | null mean | bar (95th pct) | fires |
|---|---|---|---|---|
| year 1 raw *(in-sample)* | 0.09547 | 0.03383 | 0.03690 | **10/10** |
| year 1 size-controlled | 0.09562 | 0.03136 | 0.03472 | **10/10** |
| **year 2 raw** *(held out)* | **0.08598** | 0.03305 | 0.03587 | **10/10** |
| **year 2 size-controlled** *(held out)* | **0.08417** | 0.03099 | 0.03415 | **10/10** |

**It is not the basket-size effect in disguise.** The control changes the result
almost not at all (0.09547→0.09562; 0.08598→0.08417) because there was little size
confound to remove: window basket size 9.62 vs baseline 9.54 in year 2. This matters
because the project's Instacart finding is that *basket size, not calendar*, carries
category signal — so a raw seasonal shift had to be proven not to be that.

### The strict week-block null

The basket-clustered null above prices basket sampling noise, not week-to-week
variation (promotions, store mix). The stricter test computes the TVD of **every**
contiguous 6-week window against its own complement:

| | frozen window | 95th pct of the other 29 | rank |
|---|---|---|---|
| year 1 *(in-sample, circular)* | 0.09322 | 0.08154 | 1/30 |
| **year 2** *(held out)* | **0.08773** | **0.08276** | **1/30** |

**Cite the replication, not the magnitude.** The margin over ordinary week-to-week
variation is only ~6%. The strong evidence is that a window frozen from year 1 ranks
**first of 30** in the held-out year (~1/30 ≈ 3% by chance).

### Category evidence — unambiguous

Risers in weeks 35–40 (consistent across both years, `category_shift.csv`):
**BAKING NEEDS +0.82pp**, **CHRISTMAS SEASONAL +0.45pp**, REFRGRATD DOUGH +0.34,
FROZEN PIE/DESSERTS +0.34, SUGARS/SWEETNERS +0.31, SPICES & EXTRACTS +0.25.
Fallers: YOGURT −0.44, STONE FRUIT −0.31, BEEF −0.31, FRZN NOVELTIES/WATER ICE −0.24,
BERRIES −0.22.

Holiday baking replacing summer produce. **Year-1 vs year-2 shift agreement
Spearman 0.7930** — replication evidence independent of the TVD test.

### The anchor caveat

Dunnhumby's `WEEK_NO` is relative to study start, not the calendar, and the Kaggle
CSVs carry no dates. The statistical claim is therefore **"annual periodicity at
offset X"**, never a named holiday. That said, a category literally named
`CHRISTMAS SEASONAL` peaking in the window anchors it *informally* to Nov–Dec.

---

## Part 1b — IS THE CONDITIONING EFFECTIVE? Windows are mutually distinct, and profiles do NOT transfer

*(`dunnhumby_window_discrimination.py`, Jul 31. Read-only, nothing retrained. The
detector is **imported** from `dunnhumby_signal_search.py`, not copied, so it cannot
drift from the fixture-verified original.)*

Part 1 established that **one** window differs from its baseline. It did not ask the
two questions an examiner will ask: are *different* periods distinguishable **from each
other**, and does a model fitted to one period **work on another**? Five
non-overlapping 6-week windows inside the measured usable range, anchored so the frozen
window is one of them: `W17_22, W23_28, W29_34, W35_40 (frozen), W41_46`.

**Gates, all passed before any verdict was computed:** known-answer reproduces
`signal_search.json` on all 8 values **bit-exact (0.00e+00**, spec ≤1e-9); null fixture
(two halves of the *same* window) fires **0/10** raw and size-controlled; a planted 30%
category shift fires **10/10** both.

### Q1 — every window is distinguishable from every other

All **10/10** pairs fire **10/10 draws in both years**, size-controlled. Magnitudes
(year 2, size-controlled TVD):

| pair | TVD | | pair | TVD |
|---|---|---|---|---|
| **W17_22 vs W35_40** | **0.1213** | | W17_22 vs W41_46 | 0.0804 |
| W23_28 vs W35_40 | 0.0946 | | W35_40 vs W41_46 | 0.0775 |
| W17_22 vs W29_34 | 0.0887 | | W29_34 vs W35_40 | 0.0722 |
| W17_22 vs W23_28 | 0.0589 | | W23_28 vs W41_46 | 0.0562 |
| W23_28 vs W29_34 | 0.0549 | | **W29_34 vs W41_46** | **0.0525** |

**⚠ This is a caveat on the seasonal framing and must be stated.** Because *any* two
6-week blocks separate at this sample size, "the window is distinguishable" is **not**
by itself evidence of seasonality. The frozen window is the *extreme* of a continuous
through-year drift — the largest single separation is between the two temporal
extremes (W17_22 vs W35_40) — and the seasonal claim rests on **magnitude + the
out-of-sample rank 1/30 + cross-year replication**, not on distinguishability.

### Q2 — a profile learned on one window does NOT transfer to another

**Basis was the hard part, and two earlier ones were contaminated in opposite
directions.** Both are recorded in the script so the error is not repeated:
- profile = window − *"all other weeks"* → every window sits inside the others'
  baseline, which **anti-correlates** the profiles by construction (measured −0.15);
- profile = window − *"a fixed disjoint baseline"* → now all five share a
  "mid-year vs boundary-weeks" contrast, which **co-correlates** them. Its own chance
  floor measured **+0.65 — above the cross-year value**, proof that the basis rather
  than the signal was dominating.

Final basis is symmetric (each window centred on the mean of all five) with the floor
**measured by label permutation through the identical pipeline**:

| quantity | Spearman |
|---|---|
| FLOOR — permuted window labels (n=200 pairs) | **−0.2144** |
| CEILING — same window, two disjoint halves | **+0.6542** |
| replication — same window, year 1 vs year 2 | **+0.6253** |
| **transfer — different windows, same year** | **−0.1793** |

**TRANSFER INDEX = (transfer − floor)/(replication − floor) = 0.042.**

A window's profile replicates *itself* across years at **96% of its own reliability**,
and predicts a *different* window at **4%** — statistically indistinguishable from a
permuted label. The measured floor (−0.2144) matches the theoretical sum-to-zero bias
of −1/(5−1) = −0.25, which confirms the null is pricing the right artifact rather than
an accident of the data.

**Two things follow.**
1. **The conditioning is period-specific.** A generator conditioned on one window
   carries no useful information about another. Seasonal conditioning cannot be
   validated on one window and assumed to generalise — each period is its own target.
2. **It also rules out a panel-composition confound.** If the window's category shift
   were an artifact of *which households happened to shop those weeks*, it would not
   replicate across two years with different panels at 96% of its own reliability. The
   shift is a property of the calendar position, not of the shoppers present.

---

## Part 1c — LOCATION CONDITIONING, tested with the best available proxy

*(`dunnhumby_store_discrimination.py`, Jul 31. Read-only, nothing retrained.)*

Location conditioning was previously recorded as **NOT ATTEMPTED** on the grounds that
no public order dataset carries geography. That is true of *coordinates* — but
Dunnhumby carries **`store_id` (582 stores, 516 inside the usable weeks)**, and a store
is a place. If location carries demand signal at all, it must appear as between-store
differences in category mix.

**What this does and does not settle.** It does **not** recover geography: stores
cannot be placed on a map, ranked by distance, or grouped into regions, so nothing here
supports a claim about "parts of the world". What it *does* settle is the prior
question the brief depends on — **is there a location signal to condition on, and does
a profile fitted at one location transfer to another?** Same detector, same basis, and
the same two questions as the seasonal axis, so the two are directly comparable.

**Stores were chosen by VOLUME, before any measurement** (the five highest basket
counts), so there is no selection on the outcome. **Gates:** same-store halves fire
**0/10** raw and controlled; a planted 30% shift fires **10/10** both.

**The basket-size control is load-bearing here** — the five stores have mean basket
sizes from **7.85 to 11.01**, i.e. genuinely different store formats. A difference that
died under the control would be a format artifact, not location.

### Q1 — every store pair is distinguishable, and it survives the size control

**10/10 pairs fire 10/10 draws, raw AND size-controlled** (so none of it is format).
Size-controlled TVD ranges **0.1326 (343 vs 361) to 0.2086 (406 vs 381)**.

### Q2/Q3 — store-specific, and stable a year later

| | Spearman |
|---|---|
| FLOOR — permuted store labels (n=200 pairs) | −0.2153 |
| CEILING — same store, disjoint halves | +0.6866 |
| **cross-YEAR replication (same store, year1 vs year2)** | **+0.5199** |
| transfer BETWEEN stores | −0.1959 |

**TRANSFER INDEX = 0.021.** A store's profile reproduces itself a year later at **76%
of its own reliability** while remaining anti-correlated with other stores.

**The cross-year check is what makes this a location result rather than a household
one.** A sceptic can say stores differ only because different households happen to shop
at them; a profile that reproduces itself a year later, with a partly different panel,
is a stable property of the site.

### ⭐ The comparison that matters — location carries MORE signal than season

Both axes measured with the **identical detector, basket-clustered null, and
size control**, so the magnitudes are directly comparable:

| axis | size-controlled TVD (range) | mean | replicates? | transfers? |
|---|---|---|---|---|
| **location (store)** | **0.133 – 0.208** | **~0.170** | yes, +0.520 | no, index 0.021 |
| season (6-week window) | 0.053 – 0.121 | ~0.077 | yes, +0.625 | no, index 0.042 |

**Location carries roughly 2.2× the category-mix signal that season does.** For the
brief this reverses the expected priority: the axis dismissed as untestable is the
*stronger* one, and it is testable with the data in hand. Both axes share the same
operational consequence — **neither transfers**, so a generator validated on one
window, or at one store, says nothing about another. Conditioning must be validated
per-instance.

**Caveat to carry:** `store_id` is an identifier, not a coordinate. This supports
"location-specific demand exists, is stable, and is instance-specific". It does **not**
support any claim about regions, distance, or geography, and a real deployment would
need actual site metadata to go further.

### The confound this result created — and closed

*(`dunnhumby_seasonal_within_store.py`)*

The location finding immediately puts the **seasonal gate at risk**. If stores differ
in category mix by ~2.2× more than seasonal windows do, then any drift in *which stores
the panel visited* between window and baseline weeks would masquerade as a seasonal
shift. The panel-ramp cutoff fixed the household-count ramp; it never addressed store
mix. **And the drift is real: store-mix TVD between window and baseline weeks is
0.04898.**

But the raw store-mix TVD (over 516 stores) and the seasonal category TVD (over 302
categories) are on **different supports and are not comparable as numbers**. The
question is how much category shift the store drift actually *induces*. That is
directly measurable: take **baseline-period baskets only** (season held fixed) and
reweight the stores to the **window's** store distribution.

| quantity | value |
|---|---|
| category TVD induced by store-mix drift **alone** | **0.00235** |
| seasonal category TVD to be explained (gate, year 2) | 0.08598 |
| **share of the seasonal effect attributable to store mix** | **2.7%** |

Computed over the 139 stores with ≥200 baseline items, covering **99.0% / 99.1%** of
baseline / window baskets. **Store composition accounts for at most 2.7% of the
seasonal effect — the confound was live in principle and is negligible in magnitude.**

A weaker corroborator points the same way: holding the store fixed entirely, the
seasonal effect still fires in **4/6** individual stores size-controlled (10/10, 8/10,
9/10, 6/10, 5/10, 3/10). **Treat this as corroboration only, not evidence** — per-store
n is 1–2 orders of magnitude smaller than the pooled gate, so its bars are ~5× looser
(0.15–0.24 vs the gate's 0.034) and its TVD magnitudes are **not** interchangeable with
the gate's 0.086. The decomposition above is the load-bearing evidence.

---

## Part 2 — THE CONDITIONAL VERDICT: MISSES

> ### ⚠ READ PART 2b FIRST — THIS SECTION IS THE UNDER-TRAINED MODEL
>
> Everything below describes **v1**, whose VAE stopped at epoch 650 of a 4000-epoch
> cap. The full-protocol rerun (**Part 2b**) was completed 2026-08-01 and **materially
> changes two of the three numbers here**: the category marginal falls from 8.9× the
> floor to **2.9×**, and direction agreement rises from 0.2575 to **0.6681** (7/10 top
> risers instead of 2/10). The *verdict label* is unchanged — still MISSES — but the
> reason is different, and **the v1 direction and marginal figures must not be quoted
> as the project's result.** They are retained because the pre-registration and the
> v1-vs-v2 contrast are themselves the evidence that the training caveat was
> load-bearing.

TabSyn trained on dataname `dunnhumby_season` (600,000 rows; schema `category` (300),
`season_period` (binary, also the binclass target), `basket_size_grp` (3), `quantity`).
**`week_of_year` was deliberately excluded** — `season_period` is a deterministic
function of it, so including both would have made the test circular (the v2
`aisle_popularity` defect). Artifact of record: `data/dunnhumby/synthetic_season.csv`.

Criteria were **pre-registered before any synthetic output existed**.

### Magnitude — fires 0/10

All numbers on ONE common basis: item-level, matched on `basket_size_grp`, at an
identical per-group budget (small 16,562 / mid 14,461 / large 16,264 = **47,287 items
per side**) shared by every dataset and every week block.

| Stream | TVD | fires vs item-level bar **0.07908** |
|---|---|---|
| real_train *(what the model saw)* | 0.09037 ± 0.00239 | **10/10** |
| real_full | 0.09071 ± 0.00166 | **10/10** |
| **synthetic** | **0.07088 ± 0.00254** | **0/10** |

Synthetic reaches **78.4%** of real's magnitude but never clears the strict bar — it
is indistinguishable from ordinary week-to-week variation.

### Direction — 0.2575 against a bar of 0.7500

| | value |
|---|---|
| synth-vs-real shift agreement (Spearman) | **0.2575** |
| chance agreement (arbitrary week blocks), mean | 0.0096 |
| pre-registered bar (95th pct of arbitrary blocks) | **0.7500** |
| real year-1-vs-year-2 ceiling | 0.7930 |
| top-10 riser overlap | **2/10** |

- **Real top risers:** BAKING NEEDS · CHRISTMAS SEASONAL · REFRGRATD DOUGH ·
  SUGARS/SWEETNERS · FROZEN PIE/DESSERTS
- **Synthetic top risers:** BAG SNACKS · EGGS · FLOUR & MEALS · VEGETABLES-ALL OTHERS ·
  CITRUS

The model learned *something* real — 0.2575 sits far above chance (0.0096) and
`FLOUR & MEALS` is baking-adjacent — but roughly a third of the way to faithful. The
coherent holiday-baking signature is absent.

**Bar honesty:** the 0.75 bar is inflated by week blocks that *overlap* the frozen
window and therefore share its signal (the null's mean is 0.0096, so the distribution
is heavily skewed). The bar was **not** moved after seeing the result; the informative
reading is synthetic's position between chance 0.01 and ceiling 0.79.

---

## Part 2b — THE FULL-PROTOCOL RERUN (2026-08-01): the caveat was load-bearing

The v1 verdict carried a training caveat that could not be argued away: its VAE stopped
at **epoch 650 of an unconditional 4000-epoch cap**, while the Instacart bench
completed all 4000. Two facts made the asymmetry undeniable — `tabsyn/vae/main.py` has
**no `break`**, so the only exit from its loop is finishing 4000 epochs, and the
`warehouse` run's post-loop artifacts exist (proving it did); and `warehouse`'s last
validation improvement came at **epoch ~1,550 of 4000 (39% of the run)** whereas
`dunnhumby_season`'s came at **epoch 650 of 650 — still improving when killed**.

The rerun used isolated dataname **`dunnhumby_season_v2`**: a bit-identical copy of the
prepared tables (every file sha256-verified; `info.json` differing only in `name` and
`data_path`), **repo-default hyperparameters, unchanged**, and the original epoch-650
checkpoint hash-asserted intact. Because every TabSyn path is keyed on dataname,
nothing under `dunnhumby_season` could be touched. The scorer
(`dunnhumby_conditional_test_v2.py`) was **generated from the pre-registered original by
verified substitution** — exactly three path lines differ, asserted programmatically —
so v1 and v2 are judged by identical criteria. Its known-answer gate reproduces
**bit-exact (0.00e+00)**.

**Cost, measured:** 4000 epochs in **17,865.7 s = 4.96 h at 4.47 s/epoch**. *(This
supersedes the ~14.8 h figure derived from the Jul-29 run, which was evidently
contended. Treat 13.35 s/epoch as a contended upper bound, not the cost.)*

### What changed

| | v1 (epoch 650) | **v2 (epoch 4000)** | reference |
|---|---|---|---|
| category-marginal TVD | 0.14047 = **8.9×** floor | **0.04645 = 2.9× floor** | floor 0.01587 |
| direction, Spearman | 0.2575 | **0.6681** | bar 0.7420, ceiling 0.7930 |
| top-10 risers correct | 2/10 | **7/10** | — |
| magnitude TVD | 0.07088, fires **0/10** | 0.07331, fires **0/10** | real 0.09037, bar 0.07908 |
| **verdict** | MISSES | **MISSES** | — |

### The verdict label is unchanged; the finding underneath it is not

v1 missed on **both** axes — its direction agreement (0.2575) sat near the chance floor
(0.0096), i.e. it had barely learned *which* categories move. v2 gets the direction
**largely right**: 0.6681 is **84% of the real year-on-year ceiling** (0.7930), with
7 of the 10 top risers correct. What remains is a **magnitude** failure, and a
systematic one — synthetic reaches **81%** of real's seasonal shift (0.0733 vs 0.0904)
and still never clears the week-block bar.

**The defensible claim is therefore narrower and sharper than v1's:**

> At full training budget, TabSyn learns **which** categories shift seasonally but
> **under-states how much** they shift. Its 300-way category marginal remains 2.9× the
> measured noise floor.

**Do NOT quote the v1 figures** (9× marginal, 0.26 direction, 2/10 risers) as the
project's result — they measure an under-trained model, and the v1-vs-v2 contrast is
now itself the evidence for why training budget matters on a high-cardinality
conditional target.

**What did NOT change:** the magnitude verdict (0/10 both times) and the category
marginal's *structure* — still a **systematic shift, not coverage collapse** (0
categories absent, 0 near-zero, entropy 4.855 real vs 4.829 synthetic). The model gets
the distribution's shape right and misallocates which category holds the mass.

---

## ⚠ The finding that reframed the v1 verdict — the category marginal was also wrong
*(v1 only; reduced from 8.9× to 2.9× by the full-protocol rerun — see Part 2b)*

| | value |
|---|---|
| synthetic-vs-real **category** TVD | **0.14053** |
| measured real-vs-real floor (95th pct, 20 half-splits at n=270,061/side) | **0.01578** |
| ratio | **9×** |

The model's unconditional category error is **larger than the entire seasonal effect
(0.090) it was asked to reproduce.** `season_period` (0.8301/0.1699 vs real
0.8249/0.1751) and `basket_size_grp` (within 0.002 on every level) are reproduced
well; the 300-way high-cardinality `category` column is not.

**So this is NOT "reproduces the marginals but misses the conditional."** The category
distribution itself is substantially off, and the conditional miss sits on top of that.
Any write-up must say it that way.

*(A sanity-check one-liner printed "marginals reproduced well" during the run. That was
a hardcoded string contradicted by its own number and is withdrawn.)*

---

## ⚠ Training caveat — RESOLVED 2026-08-01, and it was justified

> **STATUS: CLOSED.** The full-protocol rerun described in **Part 2b** completed all
> 4000 epochs on 2026-08-01. The caveat below is retained as the record of why the
> rerun was necessary — and it was: retraining cut the category-marginal error from
> 8.9× to 2.9× the floor and lifted direction agreement from 0.2575 to 0.6681. Anyone
> citing the Dunnhumby conditional result should cite **v2**.

The **VAE was stopped at epoch 650 of an unconditional 4000-epoch cap**.
`tabsyn_repo/tabsyn/vae/main.py` has **no early stopping** — there is no `break`, and
the `patience==10` branch only decays beta — so the cap would have taken **~14.8 h
(measured)**.

> **CORRECTION (Jul 31) — cost measured, not estimated.** This document previously
> said `~15.5 h`, an estimate at 4.3 epochs/min. Measured from the run's own logs:
> wall clock 20:56:55 → 23:21:49 on Jul 29 = **8,694 s over 651 completed epochs
> = 13.35 s/epoch → 14.84 h** for 4000. The tqdm bars alone give **7.71 s/epoch
> → 8.57 h**, but that is the *training loop only*: the validation forward pass at
> `vae/main.py:163-171` runs every epoch over the full test tensor (59,877 rows,
> unbatched) and drives both the LR scheduler and the checkpoint save, so it belongs
> in the cost. **The original estimate was very nearly right; a train-loop-only
> reading of ~8.3 s/epoch → ~9.2 h understates the rerun by ~40% and must not be
> used.** (Same defect class as the DCR / marginal-error basis mixing corrected
> earlier in this project.)
>
> **The VAE also cannot resume.** `vae/main.py` has **no `torch.load`, no
> `load_state_dict`** on the training path and no `--resume` argument; it builds
> `Model_VAE` fresh at line 105 and trains from random init on every invocation.
> (`pre_encoder.load_weights(model)` at line 202 is an in-memory copy *after*
> training, not a checkpoint restore.) The epoch-650 checkpoint cannot be continued —
> a full-protocol run pays the entire ~14.8 h from scratch.

The checkpoint **is converged**: val ACC **1.000000**, val MSE/CE 5e-06, beta decayed
0.01 → 8e-06 (the plateau branch fired repeatedly). sha256[:16] `5094d9ec664fc803`,
best-val-loss state (`vae_train_state.json`).

Two things qualify the caveat **in opposite directions**:

- **Weakening it** — the **diffusion** stage, which actually learns the joint and
  conditional structure, ran to its *own* early-stopping criterion (epoch 510,
  patience 500, 1566.8 s) and is complete by the protocol's definition. The VAE
  already reconstructs at ~100%.
- **Strengthening it** — the 9×-floor category-marginal error is exactly what a
  better-refined latent space might reduce, and it is plausibly upstream of the
  conditional miss.

**How to cite this — SUPERSEDED, see Part 2b.** The v1 wording was: *"this model, at
this training budget, misses the seasonal shift, while also getting the 300-category
marginal wrong by 9× the noise floor."* **That is no longer the project's claim.** The
full-protocol rerun completed 2026-08-01 and the marginal error is **2.9×**, not 9×,
with direction agreement 0.6681 rather than 0.2575. The current claim is:

> **At full training budget, TabSyn learns *which* categories shift seasonally but
> under-states *how much* they shift; its 300-category marginal remains 2.9× the
> measured noise floor.**

It remains true that NOT *"TabSyn cannot reproduce seasonal conditioning"* — that
overclaims in the other direction. **The gate result (Part 1) is unaffected and stands
on its own.**

---

## Method notes — five defects caught before they reached a result

1. **An item-level null is invalid here.** The first raw null split *items*, but items
   within a shopping trip are correlated, so it understated variance — its own Case-A
   fixture **fired 10/10 on two halves of the same year**. Now basket-clustered
   throughout. Without the real-data selftest, every gate number would have been
   inflated.
2. **Panel ramp-up.** Dunnhumby onboarded households progressively: year 1 week 1 has
   86 households / 65 stores vs year 2's 1,288 / 138. Weeks 1–15 measure panel
   composition, not season, and the discovery step would have selected them. Cutoff
   **measured** (first week the year1/year2 household ratio holds ≥0.90 for three
   consecutive weeks) → usable weeks **16–50**, dropped from both years. A first
   cutoff rule ("ratio holds for every later week") was also wrong — one week-40 dip
   discarded 40 of 50 weeks.
3. **One unseen category killed the first training run.** `FROZEN PACKAGE MEAT` landed
   in test but not train; TabSyn sizes its embedding table from train, so that single
   value triggered a device-side assert (`t >= 0 && t < n_classes`) at the first
   validation pass. Fixed by a category-stratified split, asserted as a gate. Rare
   categories were **not** dropped — the 12 with <10 rows are 0.008% of items, but
   dropping them would have moved the TVD support off the gate's basis.
4. **The binclass target comes back as an integer code.** `season_period` is TabSyn's
   target, so synthetic returns `0/1` while real data uses `'baseline'`/`'window'`.
   Left unmapped, every synthetic pool would have come back **empty** and the verdict
   would have been computed on nothing. Now mapped via the prep report's own
   `target_classes`, with assertions on both sides.
5. **VAE artifacts had to be reconstructed.** A completed run writes `model.pt` +
   `encoder.pt` + `decoder.pt` + `train_z.npy`, but the last three are written only
   *after* the training loop (`vae/main.py:195-212`); the paused run had only
   `model.pt`, while diffusion loads `train_z.npy` and sampling needs `decoder.pt`.
   `dunnhumby_vae_finalize.py` replicates the repo's post-loop block verbatim against
   the saved weights (constants imported from the repo module so they cannot drift) —
   one forward pass, **no retraining**.

### Detector fixtures (`dunnhumby_fixtures.py`, synthetic data, 3/3 PASS)

| Fixture | Planted | Raw | Size-controlled |
|---|---|---|---|
| **F1 flat** | nothing | 0/10 quiet ✓ | 0/10 quiet ✓ |
| **F2 real seasonal** | mix shifts, sizes held identical | 10/10 fires ✓ | 10/10 fires ✓ |
| **F3 size-driven** | mix depends *only* on basket size; window just has bigger baskets | **10/10 fires** (TVD 0.322, ≈9× bar) | **2/10 quiet** ✓ |

**F3 is what earns the basket-size control its place** — it plants exactly the confound
this project already hit on Instacart and produces an overwhelming false positive when
measured raw. F2 is the necessary counterweight: the control does not destroy genuine
signal.

### Known-answer gate — bit-exact

Before touching synthetic data, the scorer reproduced the gate's own year-2 numbers.
All seven checks matched to **0.00e+00** (spec ≤1e-9), including
`weekblock_frozen_tvd` 0.087730061, `weekblock_bar` 0.082764067 and
`category_shift_spearman` 0.792962778.

---

## Stated limitations

1. **Anchor-free.** `WEEK_NO` is relative to study start; no calendar dates. Claims are
   "annual periodicity at offset X", never a named holiday.
2. **Two week-block bars exist and are NOT interchangeable** — basket-clustered
   **0.08276** (the gate) and item-level **0.07908** (the conditional test, because
   synthetic rows have no `basket_id`). Same defect class as the DCR and
   marginal-error basis mixing corrected earlier in this project.
3. **The thin margin.** The seasonal window exceeds ordinary week-to-week variation by
   only ~6%; the replication, not the magnitude, is the evidence.
4. **The training budget** (above) — load-bearing for the negative verdict.
5. **One generator, one budget.** No CTGAN comparison and no hyper-parameter search
   were run on this dataset.

## Framing against the Instacart result — must be stated precisely

This does **not** contradict the Instacart "calendar carries no signal" finding.
Instacart has no seasonal field at all (only `order_dow`, `order_hour_of_day`,
`days_since_prior_order`), so that result was about **intra-week** timing. Dunnhumby
tests **annual** seasonality, which Instacart could not. The two are complementary, and
the write-up must say so explicitly rather than presenting them as a reversal.

## One-line summary

> A genuine annual seasonal signal exists in Dunnhumby — it replicates out-of-sample
> (rank 1/30 in a held-out year), survives a basket-size control that strips a planted
> size-driven confound from 10/10 to 2/10, and is not store-mix drift (store
> composition induces 2.7% of it). Trained to its full protocol, a conditional TabSyn
> **learns which categories move but under-states how much**: direction agreement
> 0.6681 against a real year-on-year ceiling of 0.7930 with 7/10 top risers correct,
> yet only 81% of real's magnitude and 0/10 draws above the week-block bar. Its
> 300-category marginal remains 2.9× the measured noise floor. The under-trained model
> that preceded it looked far worse on both counts (9× marginal, near-chance
> direction) — **the training budget was load-bearing, and the v1 figures are
> superseded**. Separately, the axis originally written off as untestable — **location,
> via `store_id` — carries ~2.2× the signal season does**, and neither axis transfers
> between instances, so conditioning must be validated per-period and per-site.
