"""Launch the FULL-PROTOCOL (4000-epoch) TabSyn VAE on dataname
`dunnhumby_season_v2`, WITHOUT modifying the vendored tabsyn repo.

WHY A LAUNCHER IS NEEDED (two separate repo problems)
-----------------------------------------------------
1. `tabsyn/vae/main.py` has NO `main(args)` call in its `__main__` block --
   it parses args, sets `args.device`, and the file ends. Running
   `python -m tabsyn.vae.main` therefore exits 0 having done NOTHING, with
   empty stdout/stderr. The repo's own entry point is `main.py --method vae`,
   which routes through `utils.execute_function` (utils.py:4-27).

2. `tabsyn/vae/main.py:115` passes `verbose=True` to `ReduceLROnPlateau`.
   torch 2.11.0 REMOVED that parameter, so the repo entry point crashes with
   `TypeError: unexpected keyword argument 'verbose'`.

PROTECTION RULES say the vendored repo is not to be edited, so (2) is fixed
with a runtime shim rather than a source edit.

WHY THE SHIM IS BEHAVIOUR-NEUTRAL (asserted at runtime, not assumed)
--------------------------------------------------------------------
`verbose` was a PRINT-ONLY flag. In torch 2.11 it is not merely deprecated but
fully removed: `inspect.signature(ReduceLROnPlateau.__init__)` has no `verbose`
parameter and `__init__` contains no reference to it. Dropping the kwarg cannot
change the LR schedule, the loss, or the checkpoint. The launcher asserts both
facts before training and refuses to run if either stops holding.

HYPERPARAMETERS ARE REPO DEFAULTS, UNCHANGED
--------------------------------------------
max_beta 1e-2, min_beta 1e-5, lambd 0.7 -- read from the repo's own parser
defaults (utils.py:140-142), which match tabsyn/vae/main.py:219-221 and the
0.01 -> 8e-06 beta decay recorded for the original run. No tuning: the stopping
rule for this project forbids hyperparameter search.

READS  : tabsyn_repo/data/dunnhumby_season_v2/*
WRITES : tabsyn_repo/tabsyn/vae/ckpt/dunnhumby_season_v2/{model,encoder,decoder}.pt
         tabsyn_repo/tabsyn/vae/ckpt/dunnhumby_season_v2/train_z.npy
         (nothing under dataname `dunnhumby_season` is touched)

Run detached; ~14.8 h at the measured 13.35 s/epoch.
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

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# The repo resolves every path relative to its own root ('data/{dataname}/...').
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch.optim import lr_scheduler  # noqa: E402

# ── Shim, applied BEFORE importing the repo module ───────────────────────
# vae/main.py does `from torch.optim.lr_scheduler import ReduceLROnPlateau`
# at import time, so patching after the import would have no effect.
_REAL = lr_scheduler.ReduceLROnPlateau
_sig = inspect.signature(_REAL.__init__)

assert "verbose" not in _sig.parameters, (
    "This torch still accepts `verbose`; the shim is unnecessary and would "
    "mask a real change. Re-check before running."
)
assert "verbose" not in inspect.getsource(_REAL.__init__), (
    "`verbose` is referenced inside ReduceLROnPlateau.__init__; dropping it "
    "may NOT be behaviour-neutral. Stop and investigate."
)


class _CompatReduceLROnPlateau(_REAL):
    """Drops the removed print-only `verbose` kwarg. Nothing else changes."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("verbose", None)
        super().__init__(*args, **kwargs)


lr_scheduler.ReduceLROnPlateau = _CompatReduceLROnPlateau

import tabsyn.vae.main as vae_main  # noqa: E402

assert vae_main.ReduceLROnPlateau is _CompatReduceLROnPlateau, (
    "shim did not reach the repo module -- import order is wrong"
)


def build_args():
    """Repo-default hyperparameters, taken from the repo's own parser."""
    ns = argparse.Namespace(
        dataname=DATANAME,
        gpu=0,
        max_beta=1e-2,   # utils.py:140
        min_beta=1e-5,   # utils.py:141
        lambd=0.7,       # utils.py:142
    )
    ns.device = f"cuda:{ns.gpu}" if torch.cuda.is_available() else "cpu"
    return ns


def main():
    args = build_args()
    print(f"stage=vae dataname={args.dataname} device={args.device}", flush=True)
    print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}",
          flush=True)
    print(f"shim=ReduceLROnPlateau(verbose=) dropped [print-only, removed in torch 2.11]",
          flush=True)
    print(f"hyperparameters (REPO DEFAULTS): max_beta={args.max_beta} "
          f"min_beta={args.min_beta} lambd={args.lambd}", flush=True)
    print(f"epoch cap = 4000 (unconditional; vae/main.py has no early stopping)",
          flush=True)
    print(f"projected ~14.8 h at the measured 13.35 s/epoch", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    vae_main.main(args)
    dt = time.time() - t0
    print("=" * 70, flush=True)
    print(f"VAE COMPLETE  {dt:.1f} s  ({dt / 3600:.2f} h)  "
          f"{dt / 4000:.2f} s/epoch", flush=True)

    ck = REPO / "tabsyn" / "vae" / "ckpt" / DATANAME
    for f in ("model.pt", "encoder.pt", "decoder.pt", "train_z.npy"):
        p = ck / f
        print(f"  {'ok ' if p.is_file() else 'MISSING'} {f:<14} "
              f"{p.stat().st_size if p.is_file() else 0:>12,} B", flush=True)
    print("\nAll four artifacts present => the training loop ran to completion "
          "(they are written only after the loop, vae/main.py:195-212).",
          flush=True)


if __name__ == "__main__":
    main()
