# CTGAN v3 vs TabSyn — bench comparison (Jul 20, 2026; travel REVISED Jul 23; marginals REVISED Jul 27)

> **Revision (Jul 27, 2026) — full bench re-audit.** All six known-answer gates
> were re-run and re-passed (association bit-exact; the fleet gate reproduces
> `real_fleet_effect.json` to 0.0e+00; quality/correlation/DCR/exact-matches
> exact). DCR basis, matched sample sizes, 10/14 bucket handling and the
> fire-rate framing all verified clean. Two corrections came out of it, both
> below: the **marginal-error row** was on two different bases and is now on
> one measured basis (and "TabSyn passes both marginals" is withdrawn), and
> the **"5-draw" ML-efficacy protocol** differs between the generators. Also
> recorded: on the exact-match basis, real held-out data gives **221/5000**
> matches against training vs TabSyn's 196 and CTGAN's 154 — both generators
> sit *below* the real-data baseline, so 196 is not a memorisation signal.

> **Revision (Jul 23, 2026).** The original headline — "TabSyn ALSO fabricates the
> travel effect (~0.98 aisles, inverted)" — rested on a **single-draw** point
> estimate and has been WITHDRAWN. Put through the same repeated-draw
> majority-vote rule the real number went through
> (`tabsyn_fleet_effect_draws.py`), TabSyn's travel gap fires only **10% of
> draws** (= reality); CTGAN's fires **100%**. The travel fabrication is robust
> for CTGAN only. See "Demand geometry" and "Headline answer" below, both
> rewritten. Aisle-mix/Gini findings are unchanged.

Deliverable of `tabsyn_bench_contract.md`. Both generators trained on the same
`data/v3_train.csv` (6 columns + `order_size_grp`, screened 10/14 buckets;
CTGAN conditions on it, TabSyn learns it as a 7th column). Both scored by the
SAME code: the `tabsyn_*` scoring copies, each of which first reproduced
CTGAN's recorded numbers on `data/synthetic_v3.csv` (known-answer gates, all
passed Jul 20 — see `results/tabsyn/known_answer/`). Artifacts of record:
`data/synthetic_v3.csv` (CTGAN) and `data/tabsyn/synthetic_tabsyn.csv`
(TabSyn), 50,000 rows each (sampling is not seed-reproducible for either).

## Standard metrics

| Metric | CTGAN v3 | TabSyn | Winner |
|---|---|---|---|
| Quality (held-out, all-categorical) | 0.9205 | **0.9448** | TabSyn |
| ML efficacy, 5-draw mean (matched 10k protocol) | 89.2% ± 0.7% | **89.7%** (draws 89.5/89.7/90.0/90.5/88.9, spread 1.6%) | tied (Δ0.5% < split noise 1.9%) |
| ML efficacy, single 50k run (unmatche<br/>d-size caveat applies to both) | 87.8% | 88.6% | — |
| Correlation similarity | 97.0% | 98.6% | near-meaningless on this data (project-notes caveat 1) — do not lead with either |
| DCR ratio — evaluate_v3 basis (see DCR-basis note below) | 1.124 | 1.025 | CTGAN (TabSyn sits closer to training data; both ≥ 1, no memorisation flag) |
| Exact matches / 5000 | 154 | 196 | CTGAN (raw count; ratio is the real signal) |
| Marginal errors: is_reorder / is_early_in_cart (single basis — see note) | 4.80pp / 3.01pp | **1.16pp / 0.96pp** | TabSyn (4.1× / 3.1× smaller error; both generators still exceed the measured bar) |
| Artifact audit (range/set/coverage/stability/scale) | sound | sound (5×50k draws stable, 5k-vs-50k scale OK) | tied |
| Generated small/large proportions vs training 39.8/60.2 | forced by conditioning | 40.0/60.0 (learned; no under-generation → gap comparison keeps power) | — |

### Marginal errors — basis note (corrected Jul 27, 2026)

The Jul 20 table quoted CTGAN 4.8%/3.0% against TabSyn 0.9%/1.5%. Those two
columns were on **different bases** (CTGAN vs the full fitted training
population; TabSyn vs an n=9,000 seeded subsample of `v3_eval`) and neither
was written to a file — they existed only in console output. Both columns are
now recomputed on **one declared basis** by `tabsyn_marginal_errors.py` →
`results/tabsyn/marginal_errors.json`.

**Basis:** absolute error in the marginal rate against the population both
generators were fitted to — `data/v3_train.csv` with `order_size_grp`
recomputed at the screened 10/14 buckets, mid dropped (272,820 rows).
Bar **measured**, not assumed: 95th percentile of |rate difference| over 40
disjoint matched half-splits of the real data at the same n (50,000 per side,
identical in null and observed). Fire rate = 10 independent draws, majority
vote. Gate: CTGAN's full-population errors reproduce the recorded 4.8/3.0 to
≤1e-4. Fixture-verified first (clean data fires 0%; a planted +5pp shift
measures 4.90pp and fires 100%).

| is_reorder / is_early_in_cart | CTGAN v3 | TabSyn | bar (measured) |
|---|---|---|---|
| Training basis, n=50,000/side | 4.80pp / 3.01pp — fires 100% / 100% | 1.16pp / 0.96pp — fires 100% / 100% | 0.49pp / 0.53pp |
| Held-out `v3_eval`, n=4,273/side | 5.95pp / 3.13pp — fires 100% / 80% | 2.32pp / 1.08pp — fires 70% / **40% (passes)** | 1.78pp / 1.64pp |
| CTGAN error ÷ TabSyn error | — | **4.1× / 3.1×** (training) · 2.6× / 2.9× (held-out) | — |

**One earlier claim is withdrawn:** "both of CTGAN's marginal fails PASS for
TabSyn" was stated against an unstated threshold. Against a *measured* bar at
n=50,000 TabSyn's errors are 2.0–2.4× the bar and fire 100% — smaller than
CTGAN's by 3–4×, but still distinguishable from real data. The honest
statement is the **ratio**, not a pass/fail: TabSyn's marginals are 3–4×
closer to real, and only at the smaller held-out sample size does
`is_early_in_cart` become indistinguishable from noise (fires 40%).

Note that pass/fail here is n-dependent by construction — at 50,000 rows a
0.5pp discrepancy is resolvable, so almost any generator "fails". That is an
argument for reporting the error magnitude and its ratio, which is what the
table above does.

## Association audit (21 pairs, same fixture-verified code)

| | CTGAN v3 | TabSyn |
|---|---|---|
| Weak-pair mean inflation | **+0.0585 (fabricates)** | −0.0006 (faithful) |
| Strong-pair mean inflation | +0.0304 | −0.0292 (slightly understates) |
| Verdict | CONFIRMED fabrication | NOT confirmed |

TabSyn does NOT share CTGAN's weak-association fabrication mechanism.

## Demand geometry (same recorded noise bars, floors asserted equal ≤1e-9, N=8,417)

| Axis | CTGAN v3 | TabSyn |
|---|---|---|
| Aisle-mix TVD small / large | 0.0998 / 0.1294 — FAIL both | 0.1021 / 0.0948 — FAIL both (bars ≈ 0.0615) |
| Gini small / large | FAIL both (under-concentrates, err ≈ 0.068/0.093) | **OK both** (err 0.0066/0.0059 vs bar ≈ 0.0129); gap ratio 95% = faithful |
| Travel small / large | OK / FAIL | OK / FAIL (large err 2.0% vs bar 1.2%) |
| Small-vs-large travel gap — **repeated-draw rule** (real fires 10% = no effect; bar 0.64) | **ROBUST fabrication: fires 100%** vs both bars, obs mean 1.906 aisles | **NOT robust: fires 10%** (20% vs own bar), obs mean 0.381 aisles; closed-form signed gap −0.093 = noise |
| all_ok | false | false |

### Travel gap — the repeated-draw reconciliation (Jul 23)

The Jul 20 single-draw point estimates (real 0.19 / CTGAN 2.36 / TabSyn 0.98) were
each ONE `|travel(large) − travel(small)|` on ONE 8,417-row subsample per group.
`tabsyn_fleet_effect_draws.py` reruns the IDENTICAL rule applied to the real data
(40 null half-splits → 95th-pct bar; 10 fresh observed draws; majority vote;
N forced to 8,417; gated by selftest + a known-answer gate that reproduces
`real_fleet_effect.json` to ≤1e-9). Locating each old estimate inside the new
10-draw distributions:

| | 10-draw distribution @ N=8,417 | old point est. | position |
|---|---|---|---|
| Reality | min 0.012 · max 0.655 · mean 0.238 · sd 0.221 | 0.186 | z = −0.24 (centre) |
| CTGAN | min 1.523 · max 2.269 · mean 1.906 · sd 0.220 | 2.364 | z = +2.09 (high tail of a genuinely large effect) |
| TabSyn | min 0.004 · max 0.731 · mean 0.381 · sd 0.202 | 0.983 | **z = +2.99 (3σ outlier of a noise-centred distribution)** |

**Cause = ordinary draw variance**, not N, estimator, or RNG-sharing:
- **N** identical (8,417) — gate-verified to ≤1e-9. Ruled out.
- **Closed-form vs permutation estimator** agree to 0.008 aisles (selftest
  fixture 3) — two orders of magnitude below the 0.4–0.9-aisle movements. Ruled out.
- **RNG-sharing** in the original (one `rs` stream shared across real+synth
  `choice()`, synth pool size differs per generator) is real but only corrupts
  the *real* reference — it is why the original printed the real gap as 0.186 in
  the CTGAN run vs 0.100 in the TabSyn run. It does not drive the synthetic
  magnitudes.
- **Draw variance** is the driver: reality's and TabSyn's true signed gaps ≈ 0,
  so a single `|Δ|` draw lands anywhere in a 0–0.7 band; TabSyn's happened to be
  a +3σ excursion, which is the entire basis of the withdrawn claim. CTGAN's true
  gap is genuinely large (whole distribution ≫ the 0.64 bar), so its single draw
  was merely ~2σ high and the conclusion is unchanged.

## Headline answer (the contract's question)

**Robustly, only CTGAN fabricates the travel effect.** Reality has no
small-vs-large travel effect (`test_real_fleet_effect.py`: fires 10% = noise).
Under the same repeated-draw rule at the same N, **CTGAN fires 100%** of draws
vs both the real noise bar and its own bar (observed mean 1.906 aisles) — a
robust fabrication. **TabSyn fires only 10%** (= reality; observed mean 0.381,
closed-form signed gap −0.093) — its Jul 20 "0.98-aisle, inverted" figure was a
+3σ single-draw outlier and is withdrawn.

The primary contribution stands, restated honestly:
1. The operational-fabrication failure mode is demonstrated **robustly for
   CTGAN**: a generator passes quality (92.0%), ML efficacy (89.2%), and privacy
   (DCR 1.08) yet invents a travel-cost effect (fires 100% vs a 0.64 bar) that
   does not exist in reality.
2. **Pairwise association audits do not predict it** — proven, not asserted, by
   `tabsyn_conditional_geometry.py`: excess Cramér's V is EXACTLY invariant under
   aisle relabelling (worst drift 2.78e-17) while the travel gap ranges 2.94
   aisles = 4.6× the bar over the same relabellings. A contingency-table metric
   is mathematically blind to travel geometry.
3. **TabSyn is the contrast that makes the point sharper, not weaker:** a modern
   latent-diffusion generator that beats CTGAN on every standard metric (quality
   94.5 vs 92.0, marginals ~5× better, faithful Gini, no weak-association
   fabrication) STILL gets aisle-mix TVD wrong (both fail that axis) — so
   standard metrics miss a real geometric error in BOTH — yet TabSyn does NOT
   robustly fabricate travel. The travel fabrication is generator-specific, not
   universal; the aisle-mix miss is shared.
4. TabSyn's better standard scores come with a thinner privacy margin (DCR 1.025
   vs 1.124) — the quality/privacy trade-off moved, as expected.

**Standing rule (added to the project notes Jul 23): cite repeated-draw fire rates as the
primary evidence, never single point estimates — a single draw has now overstated
a result twice.**

## Honest caveats
- **All single-draw travel-gap point estimates are superseded** by the
  repeated-draw section above — cite fire rates, not points. The csv-mode
  real-side point estimates (travel gap "0.10" printed for the TabSyn run vs
  "0.19" for the CTGAN run) differ because the verbatim original shares one RNG
  between synthetic and real subsampling and the synthetic pool sizes differ
  between modes; both are draws of the same no-effect null (mean 0.26, bar 0.64).
  This RNG-sharing was investigated Jul 23 and confirmed to affect only the
  *real* reference draw, not the synthetic magnitudes. Floors/bars were asserted
  identical to the recorded JSON; verdicts are unaffected.
- TabSyn trained unconditional (no conditional API); CTGAN conditioned.
  "Forced 50/50" style probes are not possible for TabSyn; its gap is measured
  from generated groups (proportions verified ≈ training).
- ML-efficacy single-run figures (87.8/88.6) carry the known unmatched-size
  caveat; the 5-draw matched protocol (89.2 vs 89.7) is the clean comparison.
- **"5-draw" is not the same protocol for the two generators.** CTGAN's 89.2%
  is 5 *seeds* on one CSV (its sampling is not seed-reproducible, so 5 fresh
  generations were not available); TabSyn's 89.7% is 5 *independent 50k
  generations*, each itself a 5-seed mean. TabSyn's protocol absorbs
  generation variance as well as split variance, so it is strictly the more
  conservative of the two — which makes the "tied within noise" verdict safe
  in the direction that matters.
- **Marginal errors are on one declared basis** (training population, 10/14,
  mid dropped) with a measured bar — see the basis note above. The Jul 20
  mixed-basis figures (0.9%/1.5% vs 4.8%/3.0%) are superseded, as is the
  "TabSyn passes both marginals" claim; the correct statement is a 4.1×/3.1×
  smaller error. Source: `results/tabsyn/marginal_errors.json`.
- Correlation numbers are near-meaningless on this data (caveat 1) for BOTH
  generators.
- **DCR basis — do not mix in the write-up.** Both DCR numbers in the table
  (CTGAN 1.124, TabSyn 1.025) are the `evaluate_v3.py` procedure:
  median(dist synth→`v3_train`) ÷ median(dist `v3_compare`→`v3_train`),
  min-max normalised, n=5000 seeded subsample. Same-basis is *verified*, not
  assumed — the known-answer gate ran this exact scoring code on CTGAN's own
  `data/synthetic_v3.csv` and reproduced CTGAN's recorded DCR to 7 dp
  (1.1243757 = `evaluation_v3.json`; see
  `results/tabsyn/known_answer/evaluation.json`). CTGAN has a SECOND, different
  recorded DCR — **1.077** (`model_comparison_v3.json`; same formula but n=4000
  subsample; README / project notes cite 1.08 as CTGAN's standalone "honest" figure).
  TabSyn was never scored by that n=4000 procedure, so **1.025 must be compared
  only to 1.124, never to 1.077/1.08.**
