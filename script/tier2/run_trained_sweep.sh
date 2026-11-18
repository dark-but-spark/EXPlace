#!/usr/bin/env bash
# Clean test of the confound: hold the training length fixed, vary only the
# placement knobs and seeds. This isolates "does gp_hpwl order placements the
# same way post-GRT timing does?" from "does training longer improve both?".
#
# The previous sweep mixed 20/150/600 episodes, and episode count turned out to
# correlate with both gp_hpwl (-0.79) and post-GRT TNS (+0.68), i.e. a confound.
set -uo pipefail
cd /data0/cjy_cxsj/EXPlace-test/EXPlace || exit 1
export OUT="$PWD/experiment_records/20261118_p1_1_trained_sweep"
export SKIP_PROBE=1
export LENGTHS="600"
export SEEDS="3 7 11 17"
export CONFIGS="halo0_cab0 halo2_cab0 halo0_cab2 halo2_cab2"
export GPUS="${GPUS:-3 4 5 6 7}"
bash script/tier2/spread_sampler.sh
