#!/usr/bin/env bash
# Post-VAE pipeline for dataname dunnhumby_season_v2.
# Waits for the VAE to COMPLETE (train_z.npy is written only after the
# training loop, vae/main.py:195-212, so its existence proves all 4000 epochs
# finished), then: diffusion -> sample -> pre-registered conditional test ->
# category-marginal check.
set -u
cd "D:/GenAI for Warehouse" || exit 1
PY_T="./.venv_tabsyn/Scripts/python.exe"
PY="./.venv/Scripts/python.exe"
Z="tabsyn_repo/tabsyn/vae/ckpt/dunnhumby_season_v2/train_z.npy"
OUT="results/dunnhumby/v2"
mkdir -p "$OUT" data/dunnhumby/v2

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for VAE to complete ($Z)"
while [ ! -f "$Z" ]; do sleep 60; done
sleep 20   # let the final writes flush
log "VAE COMPLETE -- train_z.npy present ($(stat -c%s "$Z") bytes)"

log "=== diffusion ==="
$PY_T -u dunnhumby_diffusion_v2.py > "$OUT/diffusion_train.out.log" 2> "$OUT/diffusion_train.err.log"
rc=$?; log "diffusion exit=$rc"
[ $rc -ne 0 ] && { log "ABORTING"; exit 1; }

log "=== sampling 600k rows ==="
$PY_T -u dunnhumby_tabsyn_sample.py --dataname dunnhumby_season_v2 \
      --num-samples 600000 --out data/dunnhumby/v2/synthetic_season_v2.csv \
      > "$OUT/sample.log" 2>&1
rc=$?; log "sampling exit=$rc"
[ $rc -ne 0 ] && { log "ABORTING"; exit 1; }

log "=== pre-registered conditional test (v2) ==="
$PY -u dunnhumby_conditional_test_v2.py > "$OUT/conditional_test.log" 2>&1
log "conditional test exit=$?"

log "=== category-marginal check (v2) ==="
$PY -u dunnhumby_category_marginal.py \
    --synth data/dunnhumby/v2/synthetic_season_v2.csv \
    --dataname dunnhumby_season_v2 \
    --out "$OUT/category_marginal_v2.json" > "$OUT/category_marginal.log" 2>&1
log "category marginal exit=$?"

log "=== V2 CHAIN COMPLETE ==="
