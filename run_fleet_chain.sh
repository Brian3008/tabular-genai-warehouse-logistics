#!/usr/bin/env bash
# Unattended chain for the two fleet experiments.
# Each comparison is GATED on its own power fixture passing at the SAME
# orders/run scale (both scripts enforce this and exit 1 otherwise).
#
# Order: rwseason first (fast, high power, brief-critical), then rwstyle.
set -u
cd "D:/GenAI for Warehouse" || exit 1
PY="./.venv_rware/Scripts/python.exe"

SEASON_ORDERS=150
STYLE_ORDERS=100
DRAWS=8

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== 1/4  rwseason fixtures @ ${SEASON_ORDERS} orders ==="
$PY -u rwseason_compare.py --fixtures --orders $SEASON_ORDERS --draws $DRAWS --maps 11 \
    > results/rwseason/fixture_run.log 2>&1
if [ $? -ne 0 ]; then log "rwseason FIXTURE FAILED - skipping its comparison"; else
  log "=== 2/4  rwseason comparison ==="
  $PY -u rwseason_compare.py --orders $SEASON_ORDERS --draws $DRAWS --maps 11 22 \
      > results/rwseason/compare_run.log 2>&1
  log "rwseason comparison exit=$?"
fi

log "=== 3/4  rwstyle fixtures @ ${STYLE_ORDERS} orders ==="
$PY -u rwstyle_compare.py --fixtures --orders $STYLE_ORDERS --draws $DRAWS --maps 11 \
    > results/rwstyle/fixture_run.log 2>&1
if [ $? -ne 0 ]; then log "rwstyle FIXTURE FAILED - skipping its comparison"; else
  log "=== 4/4  rwstyle comparison ==="
  $PY -u rwstyle_compare.py --orders $STYLE_ORDERS --draws $DRAWS --maps 11 22 \
      > results/rwstyle/compare_run.log 2>&1
  log "rwstyle comparison exit=$?"
fi

log "=== CHAIN COMPLETE ==="
