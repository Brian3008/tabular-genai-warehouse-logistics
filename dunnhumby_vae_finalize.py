"""
dunnhumby_vae_finalize.py - write the VAE artifacts that the paused
training run never got to write.

WHY THIS EXISTS
---------------
TabSyn's VAE writes model.pt inside the training loop (on every
val-loss improvement) but writes encoder.pt, decoder.pt and
train_z.npy only AFTER the loop finishes
(tabsyn_repo/tabsyn/vae/main.py:195-212). Our run was stopped at epoch
650 of an unconditional 4000-epoch cap, so only model.pt exists. The
diffusion stage loads train_z.npy (latent_utils.py:19-20) and sampling
needs decoder.pt, so neither can run without these.

This script performs EXACTLY the repo's post-loop block, using their
own classes, against the saved model.pt. It does NOT train: it loads
the converged weights, copies them into the encoder/decoder wrappers,
saves those, and runs one forward pass to produce the latent
embeddings. Constants (NUM_LAYERS, D_TOKEN, N_HEAD, FACTOR) are read
from the repo module rather than retyped, so they cannot drift.

No repo file is modified. Writes only into
tabsyn_repo/tabsyn/vae/ckpt/dunnhumby_season/.

CAVEAT THAT TRAVELS WITH EVERY RESULT DOWNSTREAM
------------------------------------------------
The checkpoint is converged (val ACC 1.000000, val MSE/CE 5e-06, beta
decayed 0.01 -> 8e-06, i.e. the plateau branch fired repeatedly) but
the full 4000-epoch protocol used by the Instacart TabSyn bench was
NOT completed. Any verdict drawn from it must state that - especially
a negative one, where under-training would otherwise be an
unfalsifiable confound.

Usage:  python dunnhumby_vae_finalize.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
REPO = PROJECT_ROOT / "tabsyn_repo"
DATANAME = "dunnhumby_season"


def main():
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    import torch
    from tabsyn.vae.model import Encoder_model, Decoder_model
    from utils_train import preprocess
    import tabsyn.vae.main as vmain   # for the hyper-constants only

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_dir = REPO / "tabsyn" / "vae" / "ckpt" / DATANAME
    model_path = ckpt_dir / "model.pt"
    assert model_path.exists(), f"FATAL: no {model_path}"

    data_dir = REPO / "data" / DATANAME
    info = json.load(open(data_dir / "info.json"))

    X_num, X_cat, categories, d_numerical = preprocess(
        str(data_dir), task_type=info["task_type"])
    X_train_num, _ = X_num
    X_train_cat, _ = X_cat
    X_train_num = torch.tensor(X_train_num).float().to(device)
    X_train_cat = torch.tensor(X_train_cat).to(device)
    print(f"train rows {X_train_num.shape[0]:,}   "
          f"num cols {d_numerical}   cat cardinalities {categories}")
    print(f"constants from repo: NUM_LAYERS={vmain.NUM_LAYERS} "
          f"D_TOKEN={vmain.D_TOKEN} N_HEAD={vmain.N_HEAD} "
          f"FACTOR={vmain.FACTOR}")

    # the repo builds Model_VAE and copies its weights into these two
    pre_encoder = Encoder_model(
        vmain.NUM_LAYERS, d_numerical, categories, vmain.D_TOKEN,
        n_head=vmain.N_HEAD, factor=vmain.FACTOR).to(device)
    pre_decoder = Decoder_model(
        vmain.NUM_LAYERS, d_numerical, categories, vmain.D_TOKEN,
        n_head=vmain.N_HEAD, factor=vmain.FACTOR).to(device)

    from tabsyn.vae.model import Model_VAE
    model = Model_VAE(vmain.NUM_LAYERS, d_numerical, categories,
                      vmain.D_TOKEN, n_head=vmain.N_HEAD,
                      factor=vmain.FACTOR, bias=True).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"loaded {model_path.name} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    pre_encoder.eval()
    pre_decoder.eval()
    with torch.no_grad():
        # EXACTLY the repo's post-loop block
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)
        torch.save(pre_encoder.state_dict(), ckpt_dir / "encoder.pt")
        torch.save(pre_decoder.state_dict(), ckpt_dir / "decoder.pt")
        train_z = pre_encoder(X_train_num,
                              X_train_cat).detach().cpu().numpy()
        np.save(ckpt_dir / "train_z.npy", train_z)

    print(f"train_z shape {train_z.shape}  "
          f"finite={np.isfinite(train_z).all()}")
    for f in ("model.pt", "encoder.pt", "decoder.pt", "train_z.npy"):
        p = ckpt_dir / f
        print(f"  {f:<14} {p.stat().st_size:>12,} bytes")
    print("\nVAE artifacts complete - diffusion stage can now run.")
    print("CAVEAT: converged (val ACC 1.0) but the 4000-epoch protocol "
          "was NOT completed (stopped at epoch 650).")


if __name__ == "__main__":
    main()
