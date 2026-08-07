"""Set up the FULL-PROTOCOL VAE rerun as an isolated TabSyn dataname.

WHY
---
The Jul 31 conditional verdict (MISSES) carries a load-bearing training caveat:
the VAE stopped at epoch 650 of an unconditional 4000-epoch cap, while the
Instacart bench (`warehouse`) COMPLETED all 4000 (proved by the existence of its
post-loop artifacts -- tabsyn/vae/main.py has no `break`, so the only exit from
the loop is finishing 4000 epochs). Worse, `warehouse`'s last val improvement was
at ~39% of its run while `dunnhumby_season`'s was at the very epoch it was killed,
i.e. it was still improving. The asymmetry is real, so the rerun is justified.

WHAT THIS DOES
--------------
Creates dataname `dunnhumby_season_v2` as a BIT-IDENTICAL copy of the prepared
`dunnhumby_season` tables. Training length is the ONLY variable under test, so the
data must not change by even one row -- hence copy-and-verify rather than re-prep.

Because every TabSyn path is keyed on `dataname`
(vae/main.py:56,66,72; latent_utils.py:13,18,19), the v2 run writes exclusively to
    tabsyn_repo/data/dunnhumby_season_v2/
    tabsyn_repo/tabsyn/vae/ckpt/dunnhumby_season_v2/
    tabsyn_repo/tabsyn/ckpt/dunnhumby_season_v2/
and CANNOT touch the artifacts behind the current verdict.

READS  : tabsyn_repo/data/dunnhumby_season/*            (never modified)
WRITES : tabsyn_repo/data/dunnhumby_season_v2/*         (new)
         results/dunnhumby/v2/rerun_setup.json          (new)

Run:  .venv_tabsyn/Scripts/python.exe dunnhumby_vae_rerun_setup.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_NAME = "dunnhumby_season"
DST_NAME = "dunnhumby_season_v2"
SRC = ROOT / "tabsyn_repo" / "data" / SRC_NAME
DST = ROOT / "tabsyn_repo" / "data" / DST_NAME
VAE_CKPT = ROOT / "tabsyn_repo" / "tabsyn" / "vae" / "ckpt" / DST_NAME
DIF_CKPT = ROOT / "tabsyn_repo" / "tabsyn" / "ckpt" / DST_NAME
OUT = ROOT / "results" / "dunnhumby" / "v2"

# Files that must be copied byte-for-byte. info.json is handled separately
# because exactly two fields legitimately change.
DATA_FILES = [
    "X_cat_train.npy", "X_cat_test.npy",
    "X_num_train.npy", "X_num_test.npy",
    "y_train.npy", "y_test.npy",
    "train.csv", "prep_manifest.json",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    print(f"\n  GATE FAILED: {msg}")
    sys.exit(1)


def main():
    print("=" * 72)
    print("FULL-PROTOCOL VAE RERUN -- SETUP")
    print("=" * 72)

    if not SRC.is_dir():
        fail(f"source dataset not found: {SRC}")

    # -- Gate 0: refuse to clobber a previous v2 attempt -------------------
    for p in (DST, VAE_CKPT, DIF_CKPT):
        if p.exists() and any(p.iterdir()):
            fail(f"{p} already exists and is non-empty. Refusing to overwrite a "
                 f"previous attempt. Move it aside first.")

    # -- Gate 1: every expected source file is present ---------------------
    missing = [f for f in DATA_FILES + ["info.json"] if not (SRC / f).is_file()]
    if missing:
        fail(f"source files missing: {missing}")

    DST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # -- Copy + verify bit-identity ---------------------------------------
    print(f"\ncopying {SRC_NAME} -> {DST_NAME}")
    digests = {}
    for f in DATA_FILES:
        shutil.copy2(SRC / f, DST / f)
        a, b = sha256(SRC / f), sha256(DST / f)
        if a != b:
            fail(f"copy of {f} is not byte-identical ({a[:16]} vs {b[:16]})")
        digests[f] = a
        print(f"  ok  {f:<24} sha256[:16]={a[:16]}  {(SRC / f).stat().st_size:>12,} B")

    # -- info.json: exactly two fields may change --------------------------
    info = json.loads((SRC / "info.json").read_text())
    src_info = json.loads((SRC / "info.json").read_text())
    info["name"] = DST_NAME
    info["data_path"] = f"data/{DST_NAME}/train.csv"
    (DST / "info.json").write_text(json.dumps(info, indent=4))

    changed = {k for k in set(src_info) | set(info)
               if src_info.get(k) != info.get(k)}
    if changed != {"name", "data_path"}:
        fail(f"info.json differs in unexpected fields: {sorted(changed)}")
    print(f"  ok  info.json               only 'name' and 'data_path' changed")

    # -- Gate 2: row counts still agree with the manifest ------------------
    man = json.loads((DST / "prep_manifest.json").read_text())
    if (info["train_num"], info["test_num"]) != (man["n_train"], man["n_test"]):
        fail("info.json train/test counts disagree with prep_manifest.json")
    print(f"  ok  row counts              train {info['train_num']:,} / "
          f"test {info['test_num']:,}")

    # -- Gate 3: the ORIGINAL run's artifacts are untouched -----------------
    orig_vae = ROOT / "tabsyn_repo" / "tabsyn" / "vae" / "ckpt" / SRC_NAME / "model.pt"
    orig_sha = sha256(orig_vae)[:16] if orig_vae.is_file() else None
    if orig_sha != "5094d9ec664fc803":
        fail(f"the ORIGINAL epoch-650 checkpoint changed! sha256[:16]={orig_sha}, "
             f"expected 5094d9ec664fc803")
    print(f"  ok  original ckpt intact    sha256[:16]={orig_sha}")

    report = {
        "purpose": "full-protocol (4000-epoch) VAE rerun, isolated dataname",
        "src_dataname": SRC_NAME,
        "dst_dataname": DST_NAME,
        "data_is_bit_identical": True,
        "only_variable_under_test": "VAE training length (650 -> 4000 epochs)",
        "hyperparameters": "repo defaults, unchanged (LR 1e-3, D_TOKEN 4, "
                           "max_beta 1e-2, min_beta 1e-5, lambd 0.7)",
        "sha256_16": {k: v[:16] for k, v in digests.items()},
        "original_ckpt_sha256_16_verified": orig_sha,
        "evidence_rerun_is_justified": {
            "warehouse_vae_completed_4000": True,
            "warehouse_reason": "post-loop artifacts (encoder/decoder/train_z) exist "
                                "and vae/main.py has no break -- only exit is 4000 epochs",
            "warehouse_last_val_improvement_frac_of_run": 0.39,
            "dunnhumby_last_val_improvement_frac_of_run": 1.0,
            "dunnhumby_interpretation": "still improving when killed",
        },
        "measured_cost": {
            "s_per_epoch_wall_clock": 13.35,
            "basis": "8694 s / 651 completed epochs, Jul 29 20:56:55->23:21:49",
            "projected_hours_4000_epochs": 14.84,
            "vae_can_resume": False,
            "resume_evidence": "no torch.load / load_state_dict on the training "
                               "path in tabsyn/vae/main.py; Model_VAE built fresh "
                               "at line 105",
        },
    }
    (OUT / "rerun_setup.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 72}\nALL GATES PASSED\n{'=' * 72}")
    print(f"  dataset  -> {DST}")
    print(f"  report   -> {OUT / 'rerun_setup.json'}")
    print(f"\nNext (detached, ~14.8 h):")
    print(f"  cd tabsyn_repo && python -m tabsyn.vae.main "
          f"--dataname {DST_NAME} --gpu 0")


if __name__ == "__main__":
    main()
