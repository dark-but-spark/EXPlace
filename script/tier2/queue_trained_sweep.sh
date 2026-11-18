#!/usr/bin/env bash
# Queue the confound-free trained sweep behind the running gap-fill batch.
# Both batches stage a single EXPlace_defs/<nick>.gp.def and share one results
# directory, so their flow phases must never overlap.
set -uo pipefail
P=/data0/cjy_cxsj/EXPlace-test/EXPlace
LOG=$P/experiment_records/20261118_p1_1_trained_sweep/queue.log
mkdir -p "$(dirname "$LOG")"

echo "[queue] waiting for the gap-fill batch to finish ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOG"
while pgrep -f 'spread_sampler\.sh' > /dev/null 2>&1; do sleep 60; done
while pgrep -x openroad > /dev/null 2>&1; do sleep 30; done
echo "[queue] gap-fill done ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOG"

rows=$(($(wc -l < "$P/experiment_records/20261117_p1_1_spread/labels.csv") - 1))
echo "[queue] spread labels.csv now has $rows data rows (expected 24)" | tee -a "$LOG"

echo "[queue] launching trained sweep" | tee -a "$LOG"
bash "$P/script/tier2/run_trained_sweep.sh"
echo "[queue] trained sweep exited rc=$? ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOG"
