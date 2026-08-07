"""Diffusion stage for dataname `dunnhumby_season_v2`, without editing the
vendored tabsyn repo.

`tabsyn/main.py:54` passes `verbose=True` to `ReduceLROnPlateau`, which torch
2.11 REMOVED -- the same defect as the VAE stage. Identical runtime shim,
identical behaviour-neutrality assertions (the parameter is absent from the
signature AND unreferenced in `__init__`, i.e. it was print-only).

*** WINDOWS SPAWN GUARD -- LOAD-BEARING ***
`tabsyn/main.py:36-41` builds its DataLoader with `num_workers=4`. On Windows
those workers are SPAWNED, and each one re-imports `__main__`. The first
version of this file executed everything at module level, so every worker
re-entered `dif_main.main(ns)` on import and died, producing

    RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly

roughly 8 seconds into the run. Everything that actually does work therefore
lives inside `main()` behind the `if __name__ == "__main__"` guard. Only the
imports and the scheduler shim stay at module level, because a re-importing
worker legitimately needs those.

Unlike the VAE, `tabsyn/main.py` DOES call `main(args)` via the repo's
dispatcher, and its loop has real early stopping (`patience == 500`,
main.py:92) driven by the EPOCH-MEAN loss (accumulated main.py:74-75,
averaged main.py:83). The tqdm trace on stderr is the LAST-BATCH loss
(main.py:81) -- a different quantity, and the only one the log records.

Hyperparameters are repo defaults; the project's stopping rule forbids
hyperparameter search.

READS  : tabsyn_repo/tabsyn/vae/ckpt/dunnhumby_season_v2/train_z.npy
WRITES : tabsyn_repo/tabsyn/ckpt/dunnhumby_season_v2/model*.pt
"""
import argparse
import inspect
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "tabsyn_repo"
DATANAME = "dunnhumby_season_v2"

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Module level: the repo resolves paths relative to its own root, and a
# re-importing worker needs the same sys.path to import tabsyn.*
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import torch                                    # noqa: E402
from torch.optim import lr_scheduler            # noqa: E402

_REAL = lr_scheduler.ReduceLROnPlateau
assert "verbose" not in inspect.signature(_REAL.__init__).parameters, (
    "this torch still accepts `verbose`; the shim is unnecessary and would "
    "mask a real change")
assert "verbose" not in inspect.getsource(_REAL.__init__), (
    "`verbose` is referenced inside ReduceLROnPlateau.__init__; dropping it "
    "may NOT be behaviour-neutral")


class _Compat(_REAL):
    """Drops the removed print-only `verbose` kwarg. Nothing else changes."""

    def __init__(self, *a, **kw):
        kw.pop("verbose", None)
        super().__init__(*a, **kw)


lr_scheduler.ReduceLROnPlateau = _Compat

import tabsyn.main as dif_main                  # noqa: E402

assert dif_main.ReduceLROnPlateau is _Compat, "shim did not reach the module"


def main():
    emb = REPO / "tabsyn" / "vae" / "ckpt" / DATANAME / "train_z.npy"
    assert emb.is_file(), (
        f"{emb} missing -- the VAE stage must have COMPLETED (that file is "
        f"written only after the training loop, vae/main.py:195-212)")

    ns = argparse.Namespace(dataname=DATANAME, gpu=0)
    ns.device = f"cuda:{ns.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"stage=diffusion dataname={ns.dataname} device={ns.device}",
          flush=True)
    print(f"shim=ReduceLROnPlateau(verbose=) dropped [print-only, removed in "
          f"torch {torch.__version__}]", flush=True)
    print(f"train_z: {emb}  ({emb.stat().st_size:,} B)", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    dif_main.main(ns)
    print("=" * 70, flush=True)
    print(f"DIFFUSION COMPLETE  {time.time() - t0:.1f} s", flush=True)
    ck = REPO / "tabsyn" / "ckpt" / DATANAME / "model.pt"
    print(f"  {'ok ' if ck.is_file() else 'MISSING'} model.pt "
          f"{ck.stat().st_size if ck.is_file() else 0:,} B", flush=True)


if __name__ == "__main__":
    main()
