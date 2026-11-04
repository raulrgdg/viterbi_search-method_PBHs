#!/bin/bash
# Injections of the calibration set of a (weight of -log10 NMSE_eff in R_a).
#
# 10 packs outside the campaign (drawn at random, seed 20261020, among 91-96 and
# 104-140), each with one 200-signal slice of the population, with the per-Tsft
# products (the hybrid selection depends on a, so it is redone offline for every a).
#   PBH_CONDA_ENV=/path/to/env bash studies/a_calibration/submit_condor.sh
# Output: <CALIB_RESULTS>/search/per_tsft_injected_pack-<p>.csv and search_results_injected_pack-<p>.csv
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

RESULTS="${CALIB_RESULTS:-results/a_calibration}"
# pack:signal slice (4 light, 3 intermediate, 3 heavy)
SLICES="104:0:200 115:0:200 127:0:200 131:0:200 113:200:400 128:200:400 134:200:400 95:400:600 126:400:600 140:400:600"
mkdir -p "${RESULTS}/logs"
echo "# $(date '+%F %T') commit $(git rev-parse --short HEAD); ${SLICES}" >> "${RESULTS}/logs/submit.log"
for item in ${SLICES}; do
    pack="${item%%:*}"
    signals="${item#*:}"
    echo -n "pack ${pack} signals ${signals}: " | tee -a "${RESULTS}/logs/submit.log"
    condor_submit "pack=${pack}" "n_jobs=${N_JOBS:-20}" "memory=${MEMORY:-5GB}" "results_dir=${RESULTS}" \
        "conda_env=${PBH_CONDA_ENV:-pbh-viterbi}" "search_args=--per-tsft-output --signals ${signals}" \
        workflows/condor/injected_search.sub | tail -1 | tee -a "${RESULTS}/logs/submit.log"
done
