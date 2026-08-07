# REaLTabFormer — Documented Negative Result

**Status: concluded 27 Jul 2026, day 2 of a 6-day time-box, ~5 h GPU compute.**
REaLTabFormer is **not** a peer generator in the comparison table. No checkpoint
was selected, because none is usable. The final generator set for the
dissertation's comparison table remains **CTGAN v3 + TabSyn**. RTF appears in the
write-up as the **memorisation finding** below.

---

## 1. Headline

A parent/child (relational) transformer generator was trained on the same 27,664
orders / 272,820 items as CTGAN v3 and TabSyn. Sweeping the 12 retained child
checkpoints (epochs 25–300) shows that its **memorisation-free window and its
fidelity-adequate window do not overlap**:

- The only checkpoint clean on all three memorisation measures is **epoch 25**,
  and it **fails fidelity** (2/4 checks).
- The only checkpoint that passes all four fidelity checks is **epoch 300**, and
  it reproduces **47.95 % of its generated baskets verbatim** from the training
  set.

There is no operating point that is both faithful and non-copying, so no
checkpoint was chosen.

---

## 2. The disjoint windows

Memorisation and fidelity measured on the same generated draws (400 baskets per
draw, 10 independent draws per checkpoint, bars = 95th percentile of a null
measured from real data, majority-of-draws decision rule).

| epoch | M1 self-copy | M2 any-copy (ordered) | M3 multiset | fidelity |
|---|---|---|---|---|
| **25**  | 0.0000 · 0/10 ✅ | 0.1020 · 1/10 ✅ | 0.1598 · 5/10 ✅ | **2/4** ❌ |
| 50  | 0.0003 · 0/10 ✅ | 0.0998 · 5/10 ✅ | 0.1716 · **9/10** ❌ | 3/4 ❌ |
| 75  | 0.0000 · 0/10 ✅ | 0.1333 · **10/10** ❌ | 0.2218 · **10/10** ❌ | 2/4 ❌ |
| 100 | 0.0010 · 0/10 ✅ | 0.2341 · **10/10** ❌ | 0.3471 · **10/10** ❌ | 1/4 ❌ |
| 125 | 0.0022 · 1/10 ✅ | 0.3035 · **10/10** ❌ | 0.4225 · **10/10** ❌ | – |
| 150 | 0.0022 · 0/10 ✅ | 0.3945 · **10/10** ❌ | 0.4965 · **10/10** ❌ | – |
| 175 | 0.0053 · 4/10 ✅ | 0.4311 · **10/10** ❌ | 0.5269 · **10/10** ❌ | – |
| 200 | 0.0085 · **7/10** ❌ | 0.4525 · **10/10** ❌ | 0.5520 · **10/10** ❌ | 3/4 ❌ |
| 225 | 0.0095 · **7/10** ❌ | 0.4740 · **10/10** ❌ | 0.5672 · **10/10** ❌ | – |
| 250 | 0.0135 · **9/10** ❌ | 0.4837 · **10/10** ❌ | 0.5835 · **10/10** ❌ | – |
| 275 | 0.0160 · **9/10** ❌ | 0.4818 · **10/10** ❌ | 0.5673 · **10/10** ❌ | – |
| **300** | 0.0208 · **10/10** ❌ | **0.4795** · **10/10** ❌ | 0.5687 · **10/10** ❌ | **4/4** ✅ |

✅ = at the measured baseline / matches real · ❌ = fires above the bar

Fidelity detail (4 checks, measured bars, fires-below-bar out of 10):

| epoch | aisle TVD | basket-size TVD | is_reorder | is_early_in_cart | pass |
|---|---|---|---|---|---|
| 25  | 0.1372 vs 0.0927 · 0/10 ❌ | 0.1570 vs 0.1750 · 6/10 ✅ | 0.0850 vs 0.0223 · 0/10 ❌ | 0.0125 vs 0.0197 · 8/10 ✅ | 2/4 |
| 50  | 0.1178 vs 0.0906 · 0/10 ❌ | 0.1393 · 10/10 ✅ | 0.0126 · 10/10 ✅ | 0.0133 · 8/10 ✅ | 3/4 |
| 75  | 0.1148 vs 0.0941 · 0/10 ❌ | 0.1460 · 9/10 ✅ | 0.0315 · 3/10 ❌ | 0.0122 · 8/10 ✅ | 2/4 |
| 100 | 0.1033 vs 0.0890 · 0/10 ❌ | 0.1550 · 7/10 ✅ | 0.0202 · 5/10 ❌ | 0.0207 · 5/10 ❌ | 1/4 |
| 200 | 0.0940 vs 0.0930 · 4/10 ❌ | 0.1583 · 8/10 ✅ | 0.0232 · 6/10 ✅ | 0.0176 · 7/10 ✅ | 3/4 |
| 300 | 0.0898 vs 0.0931 · 7/10 ✅ | 0.1610 · 7/10 ✅ | 0.0125 · 9/10 ✅ | 0.0166 · 6/10 ✅ | **4/4** |

---

## 3. It achieves fidelity *by* copying

Aisle-mix TVD improves monotonically across the sweep — 0.1372 → 0.1178 →
0.1148 → 0.1033 → 0.0940 → **0.0898** — and crosses its bar only at epoch 300,
which is precisely the checkpoint where **48 % of generated baskets are literal
training baskets**. Half the output *is* real data, so of course the aisle mix
matches. At that point the fidelity metrics are not measuring generation
quality; they are measuring how much of the training set was reproduced.

**All four standard fidelity checks PASS at epoch 300.** Marginals, basket-size
distribution and aisle mix are all clean. A conventional synthetic-data
evaluation would sign this model off while it regurgitates half its training
records.

The only thing that detects it is the **basket-level verbatim-copy rate against a
held-out-real baseline**, which no standard synthetic-data bench computes. This
is the same failure pattern as the project's primary contribution, in a new and
sharper form: previously *standard metrics miss a fabricated operational
effect*; here *standard metrics miss wholesale memorisation*.

Scale of the excess: generated baskets copy at **47.95 %** against a measured
held-out-real rate of **8.85 %** — a **5.4×** excess, firing 10/10 draws.

---

## 4. Mechanism: corpus memorisation, not conditional memorisation

The three measures separate cleanly and tell a specific story.

- **M1 (self-copy)** — given training parent *i*, does the model emit parent
  *i*'s **own** basket? Stays at or near the 0.005 chance baseline until epoch
  200, reaching only 2.08 % at epoch 300.
- **M2 (any-copy)** — does the generated basket appear **anywhere** in the 27,664
  training baskets? Explodes to 47.95 %.

The model is **not** learning the mapping "this parent → that basket". It is
memorising the **basket corpus** and emitting members of it.

This is explicable from the design: the parent row carries only
`order_dow`, `order_hour_of_day`, `days_since_prior_order` and
`order_size_grp`. That is far too weak a signal to identify a specific order
among 27,664, so there is no conditional mapping available to memorise — but the
decoder can and does memorise the set of baskets itself.

**Conditional generation with a weak conditioning signal degrades into corpus
reproduction.** The conditioning is real (verified: `_fit_relational` drops only
the join key and encodes every remaining parent column, so children are
conditioned on `order_size_grp` by construction) — it is simply not
*informative* enough to anchor the decoder.

---

## 5. Architectural finding: the child has no overfitting detector

`realtabformer.py:461-501` routes the two model types differently:

```python
if self.model_type == ModelType.tabular:
    trainer = self._train_with_sensitivity(df, ..., n_critic, num_bootstrap, qt_interval, quantile, ...)
elif self.model_type == ModelType.relational:
    trainer = self._fit_relational(df, in_df, join_on=join_on, device=device)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)   # plain training
```

RTF's Q_δ statistic, 500-sample bootstrap and `n_critic` early stop are reachable
**only** through the tabular branch. So:

- The **parent** model gets full overfitting detection. Ours behaved perfectly:
  30 epochs, loss 1.6301 → 1.1932, **PLATEAUED** (+0.31 %), **0 Q_δ breaches**,
  and 4/4 order-level marginals indistinguishable from real at a measured bar.
- The **child** model — the one that actually memorises — trains with **no
  detector at all**.

Target masking (`mask_rate`) *is* wired for the child (`data_utils.py:740`) and
was enabled at 0.10. It did not prevent memorisation.

Two aggravating details:

1. **The loss curve gives no warning.** Child training loss descended smoothly
   from 8.33 to 0.4274 across 300 epochs with no inflection at the memorisation
   onset (epoch ~50–75). Training-loss convergence and memorisation onset are
   **decoupled**, and the standard convergence criterion is blind to the failure.
2. **RTF's default is `epochs=1000`.** Following the library default would land
   far deeper into the memorised regime, with a loss curve that looks healthier
   the whole way.

A related observation on the parent, which had a detector: its loss plateaued by
epoch ~5 but Q_δ `val_sensitivity` climbed monotonically 0.4588 → 0.4773 through
epoch 30, consuming ~96 % of the margin to the 0.4782–0.4792 threshold while the
loss showed nothing. Same decoupling, caught only because a detector existed.
Note also `freeze_parent_model=True` is the default, so whichever parent
checkpoint is chosen is **baked permanently into the child's encoder**.

Retrospective signal (used to motivate the sweep, not as evidence): the child's
final loss of 0.4274 nats/token against a 1.1708 unigram-equivalent implies
roughly **4.1 effective aisle choices out of 134** on its own training data.

---

## 6. Caveats — both material

**(a) The bars in this document are not cross-comparable with the
demand-geometry bars.** The aisle-TVD bar here (≈0.093) is measured at ~4,000
items per side. The CTGAN/TabSyn aisle-mix bar (≈0.0615) comes from the
demand-geometry procedure at a different N with different conditioning. TVD is
sample-size dependent. **No claim is made that RTF's 0.0898 beats CTGAN's 0.0998
or TabSyn's 0.1021** — those numbers come from different rulers and must not be
placed in the same column.

**(b) This is a result about this dataset size, not about REaLTabFormer in
general.** A ~110 M-parameter encoder-decoder over 27,664 basket sequences is
heavily over-parameterised; a corpus that small is memorisable. On a larger
basket corpus the two windows might well separate, and RTF could be viable. The
narrower, firmer, generalisable finding is the architectural one in §5: RTF
ships overfitting detection but does not wire it to the relational path, so the
component that memorises trains unmonitored — and that is true regardless of
dataset size.

---

## 7. Method

Every threshold measured, every detector verified against known answers before
use, every claim as a fire rate over repeated draws.

**Nulls.** All bars are the 95th percentile of a null distribution measured from
real data at the **same sample size** as the observed comparison. A metric fires
only on a **majority** of 10 independent draws.

- M1 null: permuted same-size pairing (chance agreement).
- M2/M3 null: **size-matched held-out real baskets** — genuinely disjoint
  v3_eval + v3_compare orders, disjointness asserted. Size matching is essential
  because small baskets collide by chance: 1,871 of the 27,664 training baskets
  are duplicates of one another (25,793 distinct ordered keys).
- Fidelity nulls: disjoint half-splits of the real data at matched n.

**Population match (a bug found and fixed).** The held-out pool was initially
built from raw data *without* the mid-drop that the training tables have (sizes
11–13 excluded by the screened 10/14 buckets). The fidelity selftest correctly
**failed on clean held-out real data** (`f2_size_tvd` 0.2228 vs bar 0.1775,
0/10) — the detector was reporting a difference the null construction had
created. After matching the population (5,953 → 5,174 held-out baskets) the
clean selftest passes 4/4. The memorisation sweep was re-run under the corrected
null; only the corrected numbers appear in this document.

**Selftests (both pass).**

| fixture | expected | result |
|---|---|---|
| memorisation, planted (training baskets fed as generated) | fire | M1/M2/M3 all 1.0000, 10/10 |
| memorisation, clean (held-out real baskets) | not fire | M2 0.0885 vs 0.1100 · 0/10; M3 0.1393 vs 0.1600 · 0/10 |
| fidelity, clean (held-out real baskets) | pass | 4/4 MATCHES REAL |
| fidelity, planted (uniform aisles, flipped reorder) | fail on corrupted cols | aisle 0.4964 · 0/10; is_reorder 0.1708 · 0/10 |

The fidelity planted fixture correctly did **not** flag basket-size or
is_early_in_cart, which the plant did not corrupt — sensitivity and specificity
both demonstrated.

**Training configuration.** Parent 30 epochs (Q_δ on, `n_critic=5`); child 300
epochs, `mask_rate=0.10`, `train_size=1` (all 27,664 baskets — matching CTGAN and
TabSyn, which also trained on all rows; no validation holdout, deliberately, to
preserve comparability). batch 32, `gradient_accumulation_steps=4` (RTF default)
→ 217 steps/epoch. Parent 13.9 min; child 572.2 min (9.54 h). Checkpoints
retained every 25 epochs.

**Data.** Parent 27,664 rows (`order_dow`, `order_hour_of_day`,
`days_since_prior_order`, `order_size_grp`); child 272,820 rows (`aisle_id`,
`reordered`). Verified before training: raw rebuild equals `v3_train` row count
exactly; train ∩ eval = train ∩ compare = 0; `add_to_cart_order` == basket
position on all 27,664 baskets; 0 baskets dropped by `output_max_length`
(longest label 438 tokens vs 512); and the flatten mapping reproduces
`v3_train` **bit-for-bit** on 272,820 × 7.

---

## 8. Reproduction

Scripts (all new, `.venv_rtf`, nothing pre-existing modified):

| script | role |
|---|---|
| `rtf_prepare_data.py` | parent/child tables + 5 gates incl. bit-for-bit flatten check |
| `rtf_smoke.py` | 9 environment/API gates incl. the planted-fault generation-order fixture |
| `rtf_probe.py` | convergence + cost probe |
| `rtf_train.py` | production run (parent 30 / child 300) |
| `rtf_parent_marginals.py` | parent order-level marginals vs measured bars |
| `rtf_memorisation.py` | M1/M2/M3 checkpoint sweep |
| `rtf_fidelity.py` | F1–F4 checkpoint sweep |

Outputs: `results/rtf/{smoke_report.json, probe/, train/, memorisation/}`,
`data/rtf/`. Key files: `memorisation/sweep.json`, `memorisation/fidelity.json`,
`memorisation/selftest.json`, `memorisation/fidelity_selftest.json`,
`train/train_report.json`, `probe/parent_marginals.json`.

Four REaLTabFormer 0.2.4 defects were worked around **without modifying the
library** (documented in the scripts): `fit()` crashing on default `gen_kwargs`
(issue #103); `save()` failing to stringify `full_save_dir`; the undocumented
`gradient_accumulation_steps=4` default that makes steps/epoch `n/(batch×4)`;
and a sensitivity-bootstrap OOM at 6 workers (636 MiB × 6 vs available RAM),
capped to 3 workers with no statistic changed.

---

## 9. What was deliberately not done

- **No checkpoint was selected.** None is both faithful and non-copying.
- **RTF is not added to the comparison table** as a fourth generator.
- The automated selection rule in `rtf_memorisation.py` returns epoch 50 because
  it tests only M1 and M2; epoch 50 fires 9/10 on M3. The rule is too permissive
  and its output is **not** used.
- No attempt was made to rescue the model (larger `mask_rate`, smaller
  architecture, more data). That is future work, and the time-box was closed
  early rather than spent on it.
