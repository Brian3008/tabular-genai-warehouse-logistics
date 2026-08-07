"""Generate `dunnhumby_conditional_test_v2.py` from the pre-registered
original by VERIFIED SUBSTITUTION.

The original is read-only and its criteria were pre-registered before any
synthetic output existed -- that is the whole value of the verdict. Hand-
editing a copy risks silently changing a criterion, so the copy is generated
here from exactly three path substitutions, each asserted to apply exactly
once. Any other difference is impossible by construction.

The gate references (`signal_search.json`, `tabsyn_prep_report.json`) and
TRAIN_CSV are deliberately NOT redirected: the v2 training table is a
bit-identical copy, so the known-answer gate must still reproduce the same
recorded numbers. Only the synthetic input and the two outputs move.
"""
import hashlib
from pathlib import Path

SRC = Path("dunnhumby_conditional_test.py")
DST = Path("dunnhumby_conditional_test_v2.py")

SUBS = [
    ('SYNTH_CSV = DATA / "synthetic_season.csv"',
     'SYNTH_CSV = DATA / "v2" / "synthetic_season_v2.csv"'),
    ('json.dump(out, open(RESULTS / "conditional_known_answer.json", "w"),',
     'json.dump(out, open(RESULTS / "v2" / "conditional_known_answer.json", "w"),'),
    ('json.dump(out, open(RESULTS / "conditional_test.json", "w"), indent=2)',
     'json.dump(out, open(RESULTS / "v2" / "conditional_test.json", "w"), indent=2)'),
]

src = SRC.read_text(encoding="utf-8")
print(f"source sha256[:16] = {hashlib.sha256(src.encode()).hexdigest()[:16]}")

out = src
for old, new in SUBS:
    n = out.count(old)
    assert n == 1, f"expected exactly 1 occurrence of:\n  {old}\ngot {n}"
    out = out.replace(old, new)
    print(f"  ok  {old[:58]}...")

header = ('"""AUTO-GENERATED from dunnhumby_conditional_test.py by\n'
          'make_conditional_test_v2.py -- DO NOT EDIT BY HAND.\n\n'
          'Identical to the pre-registered original except for three paths:\n'
          '  * reads   data/dunnhumby/v2/synthetic_season_v2.csv\n'
          '  * writes  results/dunnhumby/v2/conditional_known_answer.json\n'
          '  * writes  results/dunnhumby/v2/conditional_test.json\n'
          'Every criterion, bar and basis is unchanged, so the v2 verdict is\n'
          'directly comparable with the v1 verdict.\n'
          f'source sha256[:16] = {hashlib.sha256(src.encode()).hexdigest()[:16]}\n'
          '"""\n')
out = header + out

# prove nothing else moved
a = [l for l in src.splitlines()]
b = [l for l in out.splitlines()[len(header.splitlines()):]]
diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
assert len(a) == len(b), f"line count changed {len(a)} -> {len(b)}"
assert len(diff) == len(SUBS), f"expected {len(SUBS)} changed lines, got {len(diff)}"
print(f"\nexactly {len(diff)} lines differ (lines "
      f"{[i + 1 for i, _, _ in diff]}) -- nothing else changed")

Path("results/dunnhumby/v2").mkdir(parents=True, exist_ok=True)
Path("data/dunnhumby/v2").mkdir(parents=True, exist_ok=True)
DST.write_text(out, encoding="utf-8")
print(f"-> {DST}")
