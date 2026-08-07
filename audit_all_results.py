"""End-to-end audit: every result JSON loads, every gate passed, and every
number the write-up will quote is read back FROM THE ARTIFACT rather than from
memory or from prose.

This exists because the project has been burned twice by numbers that drifted
between a JSON, a doc and a sentence. It re-reads the artifacts and flags:
  * missing or unparseable result files
  * any gate/selftest that did not pass
  * any invalid episode or deadlock in a simulator run
  * the headline figures, so they can be diffed against the docs by eye

READS  : results/**.json only.  WRITES: nothing.
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROBLEMS = []


def load(p):
    q = Path(p)
    if not q.is_file():
        PROBLEMS.append(f"MISSING: {p}")
        return None
    try:
        return json.loads(q.read_text(encoding="utf-8"))
    except Exception as e:
        PROBLEMS.append(f"UNPARSEABLE: {p} ({e})")
        return None


def gate(cond, label):
    ok = bool(cond)
    print(f"    {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        PROBLEMS.append(f"GATE FAILED: {label}")
    return ok


def hr(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t)
        print("=" * 74)


# ══ INSTACART CORE ════════════════════════════════════════════
hr("INSTACART v3 (CTGAN) — the numbers the write-up leads with")
ev = load("results/../data/evaluation_v3.json") or load("data/evaluation_v3.json")
mc = load("data/model_comparison_v3.json")
mi = load("data/marginal_impact.json")
if ev:
    print(f"  quality {ev.get('quality')}  correlation {ev.get('correlation')}  "
          f"DCR {ev.get('dcr_ratio', ev.get('dcr'))}")
if mc:
    # model_comparison_v3.json is a LIST of model dicts, not a dict keyed by model
    # name.  The original lookup assumed a dict, matched nothing and printed nothing.
    rows = {r["model"]: r for r in mc} if isinstance(mc, list) else mc
    for k, v in rows.items():
        print(f"  {k:<20} quality {v.get('quality'):.4f}  "
              f"ml {v.get('ml_efficacy'):.4f}  "
              f"dcr {v.get('privacy', v.get('dcr_ratio')):.4f}")
    # THE baseline story: the shuffle wins on marginal quality (it preserves marginals
    # by construction) but CTGAN wins decisively on ML efficacy, the metric that
    # reflects learned structure.  If that ordering ever flips, the chapter is wrong.
    if "CTGAN v3" in rows and "Shuffling baseline" in rows:
        c, s = rows["CTGAN v3"], rows["Shuffling baseline"]
        gate(s["quality"] > c["quality"],
             "baseline story: shuffle beats CTGAN on marginal quality")
        gate(c["ml_efficacy"] > s["ml_efficacy"],
             "baseline story: CTGAN beats shuffle on ML efficacy")
if mi:
    print(f"  ML efficacy 5-seed mean {mi.get('mean_efficacy', mi.get('mean'))}  "
          f"stable={mi.get('stable')}  eic_matters={mi.get('eic_matters')}")

# ══ REAL FLEET EFFECT ═════════════════════════════════════════
hr("REAL FLEET EFFECT (the primary contribution's anchor)")
rf = load("data/real_fleet_effect.json")
# The JSON keys metrics INSIDE each block (observed_mean/bar/fire_rate/effect are each
# dicts over {travel,gini,tvd}), NOT one block per metric.  An earlier version of this
# loop assumed the opposite shape, matched nothing, and printed nothing -- while the
# audit still reported "no failed gates".  A verification script that silently skips
# its most important check is the exact defect class this project guards against, so
# the expected verdict is now ASSERTED rather than merely printed.
if rf:
    EXPECTED = {"travel": False, "gini": True, "tvd": True}  # travel = the fabrication
    n_draws = rf.get("design", {}).get("n_test_draws", 10)
    for m in ("tvd", "gini", "travel"):
        obs = rf["observed_mean"][m]
        bar = rf["bar"][m]
        fr = rf["fire_rate"][m]
        eff = rf["effect"][m]
        print(f"    {m:<8} obs {obs:.4f}  bar {bar:.4f}  "
              f"fires {round(fr * n_draws)}/{n_draws}  effect={eff}")
        gate(eff == EXPECTED[m],
             f"real fleet effect: {m} effect is {EXPECTED[m]} "
             f"(travel must be False -- reality has NO travel effect)")
    gate(rf["fire_rate"]["travel"] <= 0.5,
         f"real travel fires {rf['fire_rate']['travel']:.0%} <= 50% "
         f"(the anchor for 'CTGAN fabricates travel')")

# ══ CROSS-GENERATOR (TabSyn) ══════════════════════════════════
# Neither TabSyn's scores nor THE headline finding (CTGAN fabricates travel, TabSyn
# does not) was covered by this audit before Aug 2.  They are the two claims most
# likely to be quoted externally, so they are gated here.
hr("CROSS-GENERATOR — TabSyn scores + the travel-fabrication finding")
ts = load("results/tabsyn/evaluation.json")
if ts and ev:
    print(f"  TabSyn  quality {ts['quality']:.4f}  corr {ts['correlation']:.4f}  "
          f"DCR {ts['dcr_ratio']:.4f}   (CTGAN same basis: "
          f"quality {ev['quality']:.4f}  DCR {ev['dcr_ratio']:.4f})")
    gate(ts["quality"] > ev["quality"],
         "TabSyn beats CTGAN on quality (same scorer)")
    gate(ts["dcr_ratio"] >= 1.0 and ev["dcr_ratio"] >= 1.0,
         "both generators DCR >= 1.0 (no memorisation)")

draws = {g: load(f"results/tabsyn/fleet_draws/{g}/fleet_effect_draws.json")
         for g in ("ctgan", "tabsyn")}
if rf and all(draws.values()):
    real_bar = rf["bar"]["travel"]
    print(f"  travel gap vs the REAL noise bar {real_bar:.4f}  "
          f"(reality itself fires {rf['fire_rate']['travel']:.0%})")
    fires = {}
    for g, d in draws.items():
        t = d["observed_draws"]["travel"]
        fires[g] = sum(1 for x in t if x > real_bar)
        print(f"    {g:<7} mean {d['observed_mean']['travel']:.4f} aisles  "
              f"fires {fires[g]}/{len(t)} vs real bar  "
              f"({sum(1 for x in t if x > d['bar']['travel'])}/{len(t)} vs own bar)")
    gate(fires["ctgan"] > len(draws["ctgan"]["observed_draws"]["travel"]) // 2,
         "CTGAN travel fabrication fires on a MAJORITY of draws (the contribution)")
    gate(fires["tabsyn"] <= len(draws["tabsyn"]["observed_draws"]["travel"]) // 2,
         "TabSyn travel does NOT fire on a majority (fabrication is CTGAN-specific)")
    # aisle-mix is the failure BOTH share -- if one ever passes, the story changes
    for g, d in draws.items():
        gate(d["observed_mean"]["tvd"] > d["bar"]["tvd"],
             f"{g} aisle-mix TVD exceeds its bar (the shared geometric failure)")

# ══ DUNNHUMBY ═════════════════════════════════════════════════
hr("DUNNHUMBY — gate, effectiveness, location, conditional v1 vs v2")
g = load("results/dunnhumby/signal_search.json")
if g:
    c = g["confirmation"]["year2_HELD_OUT"]
    gate(g.get("GATE_PASS"), "seasonal gate PASS")
    print(f"    year2 raw {c['raw']['observed_mean']:.5f} vs bar "
          f"{c['raw']['bar']:.5f}  fires {c['raw']['fires']}/10")
    print(f"    year2 size-controlled {c['size_controlled']['observed_mean']:.5f} "
          f"vs bar {c['size_controlled']['bar']:.5f}  "
          f"fires {c['size_controlled']['fires']}/10")

wd = load("results/dunnhumby/window_discrimination.json")
if wd:
    gate(wd["gates"]["known_answer"]["PASS"], "window: known-answer bit-exact")
    gate(wd["gates"]["known_answer"]["worst_abs_diff"] == 0.0,
         "window: worst_abs_diff == 0.00e+00")
    gate(wd["gates"]["fixtures"]["PASS"], "window: null + planted fixtures")
    v = wd["verdict"]
    print(f"    pairs distinguishable both years "
          f"{v['pairs_distinguishable_both_years']}/{v['n_pairs']}")
    print(f"    TRANSFER INDEX {v['TRANSFER_INDEX']:.3f}  "
          f"(floor {v['floor_label_permutation']:+.4f}, "
          f"ceiling {v['ceiling_self_reliability']:+.4f}, "
          f"replication {v['replication_cross_year']:+.4f}, "
          f"transfer {v['transfer_cross_window']:+.4f})")

sd = load("results/dunnhumby/store_discrimination.json")
if sd:
    gate(sd["gates"]["PASS"], "store: null + planted fixtures")
    q1, q2 = sd["Q1_summary"], sd["Q2_transfer"]
    print(f"    store pairs fire raw {q1['fire_raw']}/{q1['pairs']}  "
          f"size-controlled {q1['fire_size_controlled']}/{q1['pairs']}")
    print(f"    TRANSFER INDEX {q2['TRANSFER_INDEX']:.3f}  "
          f"cross-year {sd['Q3_cross_year_replication']['mean']:+.4f}")
    tv = [v["size_controlled"]["observed_mean"] for v in sd["Q1_pairwise"].values()]
    print(f"    store size-controlled TVD range {min(tv):.4f}–{max(tv):.4f}  "
          f"mean {sum(tv)/len(tv):.4f}")
if wd:
    tv2 = [v["size_controlled"]["observed_mean"]
           for v in wd["Q1_pairwise"]["year2"].values()]
    print(f"    window size-controlled TVD range {min(tv2):.4f}–{max(tv2):.4f}  "
          f"mean {sum(tv2)/len(tv2):.4f}")
    if sd:
        print(f"    -> location / season signal ratio "
              f"{(sum(tv)/len(tv)) / (sum(tv2)/len(tv2)):.2f}x")

ws = load("results/dunnhumby/seasonal_within_store.json")
if ws:
    print(f"    store-mix drift TVD {ws['store_mix_drift_tvd']:.5f}; induces "
          f"{ws['category_tvd_induced_by_store_mix']:.5f} = "
          f"{ws['share_of_seasonal_explained_by_store_mix']*100:.1f}% of the "
          f"seasonal effect")

c1 = load("results/dunnhumby/conditional_test.json")
c2 = load("results/dunnhumby/v2/conditional_test.json")
m1 = load("results/dunnhumby/category_marginal_v1.json")
m2 = load("results/dunnhumby/v2/category_marginal_v2.json")
if c2:
    gate(c2["known_answer_gate"]["PASS"], "conditional v2: known-answer gate")
    gate(c2["known_answer_gate"]["worst_abs_diff"] == 0.0,
         "conditional v2: bit-exact 0.00e+00")
if c1 and c2:
    print(f"    magnitude  v1 {c1['tvd']['synth']['tvd_mean']:.5f} "
          f"({c1['tvd']['synth']['fires_vs_week_block_bar']}/10)   "
          f"v2 {c2['tvd']['synth']['tvd_mean']:.5f} "
          f"({c2['tvd']['synth']['fires_vs_week_block_bar']}/10)   "
          f"real {c2['tvd']['real_train']['tvd_mean']:.5f}")
    print(f"    direction  v1 {c1['direction']['spearman_synth_vs_real']:.4f} "
          f"({c1['direction']['top10_overlap']}/10)   "
          f"v2 {c2['direction']['spearman_synth_vs_real']:.4f} "
          f"({c2['direction']['top10_overlap']}/10)   "
          f"ceiling {c2['direction']['real_year1_vs_year2_ceiling']:.4f}")
    print(f"    verdict    v1 {c1['verdict']['VERDICT']}   "
          f"v2 {c2['verdict']['VERDICT']}")
if m1 and m2:
    print(f"    marginal   v1 {m1['ratio_to_floor']:.1f}x   "
          f"v2 {m2['ratio_to_floor']:.1f}x   "
          f"(structure v2: {m2['structure']['verdict']}, "
          f"absent {m2['structure']['categories_absent']})")

# ══ SIMULATOR RUNS ════════════════════════════════════════════
for name, path, fxpath in (
        ("RWARE (Jul 29)", "results/rware/comparison.json",
         "results/rware/fixtures.json"),
        ("RWSEASON", "results/rwseason/comparison.json",
         "results/rwseason/fixtures.json"),
        ("RWSTYLE", "results/rwstyle/comparison.json",
         "results/rwstyle/fixtures.json")):
    hr(f"{name}")
    d = load(path)
    fx = load(fxpath)
    if fx:
        # two schemas in use: rwstyle/rwseason expose top-level PASS; the Jul 29
        # RWARE fixture exposes ALL_PASS over five named sub-gates
        passed = fx.get("PASS", fx.get("ALL_PASS"))
        subs = {k: v.get("PASS") for k, v in fx.items()
                if isinstance(v, dict) and "PASS" in v}
        gate(passed, f"{name}: power fixture PASS"
                     + (f"  [{len(subs)} sub-gates: "
                        f"{sum(1 for x in subs.values() if x)}/{len(subs)}]"
                        if subs else ""))
    if not d:
        continue
    cfg = d.get("config", {})
    dl = d.get("deadlocks", [])
    gate(len(dl) == 0, f"{name}: 0 deadlocks (got {len(dl)})")
    n_ep = cfg.get('n_episodes')
    if n_ep is None:   # Jul 29 RWARE records 30 rows x 6 streams = 180 episodes
        n_ep = f"{len(d.get('records', [])) or d.get('n_records','?')} records"
    elapsed = cfg.get('elapsed_s') or d.get('elapsed_sec') or 0
    print(f"    episodes {n_ep}  elapsed {elapsed/60:.1f} min")
    for metric in ("steps_per_delivery", "throughput_per_1k", "mean_order_completion"):
        if metric not in d:
            continue                      # only the Jul 29 RWARE run has these
        blk = d[metric]
        print(f"    {metric}  pooled bar {blk['bar_95pct']:.4f}")
        for s in ("A_real_true", "B_real_pool", "C_tabsyn", "D_ctgan"):
            if s in blk:
                v = blk[s]
                print(f"      {s:<13} signed vs B {v['signed_diff_vs_B_mean']:+9.4f}  "
                      f"fires {v['fires_vs_B']}/{v['n_draws']}")
        gate(all(blk[s]["fires_vs_B"] <= blk[s]["n_draws"] // 2
                 for s in blk if isinstance(blk[s], dict) and "fires_vs_B" in blk[s]
                 and s != "A_real_true"),
             f"{name} {metric}: no generator stream fires on a majority vs B")
    if "results" in d:                       # rwseason
        for k, v in d["results"].items():
            print(f"    {k:<8} w-b {v['mean_window_minus_baseline']:+.4f}  "
                  f"fires {v['fires']}/{v['n']}")
        print(f"    VERDICT: {d['VERDICT'][:96]}")
    if "Q_style" in d:                       # rwstyle
        pf = d["power_fixture"]
        print(f"    sensitivity: geometry {pf['sensitivity_overall']:+.4f} "
              f"(bar {pf['bar_overall']:.4f});  interaction "
              f"{pf['sensitivity_interaction']:+.4f} (bar "
              f"{pf['bar_interaction']:.4f})")
        for k, v in d["Q_style"].items():
            m = v["policy_effect_by_mix"]
            print(f"    {k:<14} small {m['small_heavy']:+.4f}  large "
                  f"{m['large_heavy']:+.4f}  interaction "
                  f"{v['interaction_mean']:+.4f} ({v['fires_vs_fixture_bar']}"
                  f"/{v['n_draws']})")

# ══ SUMMARY ═══════════════════════════════════════════════════
hr("AUDIT SUMMARY")
if PROBLEMS:
    print(f"  {len(PROBLEMS)} PROBLEM(S):")
    for p in PROBLEMS:
        print(f"    - {p}")
    sys.exit(1)
print("  no missing files, no failed gates, no deadlocks.")
