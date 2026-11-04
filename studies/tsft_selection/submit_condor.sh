#!/bin/bash
# Tsft-selection study: noise search of the 141 packs and injected
# search of a few campaign packs, with the 79 Tsft of the "union" list and the
# per-Tsft output (--per-tsft-output), so that every selection rule of
# pbh_viterbi.search.selection can be evaluated offline on the same products.
#
# Run from anywhere (the jobs need a filesystem shared with the execute nodes):
#   PBH_CONDA_ENV=/path/to/env bash studies/tsft_selection/submit_condor.sh
# Environment:
#   STUDY_RESULTS  output folder (default results/tsft_selection)
#   PACKS          injection packs (default 3 per signal slice: 1 2 3 13 14 15 25 26 27)
#   N_JOBS         jobs per injection pack (default 40, i.e. 5 signals per job)
#   STAGES         "noise injected" (default) or one of them
# Then: python studies/tsft_selection/collect.py && python studies/tsft_selection/analyse.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

RESULTS="${STUDY_RESULTS:-results/tsft_selection}"
PACKS="${PACKS:-1 2 3 13 14 15 25 26 27}"
N_JOBS="${N_JOBS:-40}"
STAGES="${STAGES:-noise injected}"
COMMON=("tsft_list=union" "search_args=--per-tsft-output" "memory=${MEMORY:-7GB}"
        "conda_env=${PBH_CONDA_ENV:-pbh-viterbi}" "results_dir=${RESULTS}")
mkdir -p "${RESULTS}/logs"
log="${RESULTS}/logs/submit.log"

{
    echo "# $(date '+%F %T') commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '-dirty')"
    echo "# stages: ${STAGES}; injection packs: ${PACKS}; ${N_JOBS} jobs per pack; Tsft list union"
} >> "${log}"
for stage in ${STAGES}; do
    case "${stage}" in
        noise)
            condor_submit "${COMMON[@]}" n_jobs=141 workflows/condor/noise_search.sub | tail -1 | tee -a "${log}"
            ;;
        injected)
            for pack in ${PACKS}; do
                echo -n "pack ${pack}: " | tee -a "${log}"
                condor_submit "${COMMON[@]}" "pack=${pack}" "n_jobs=${N_JOBS}" \
                    workflows/condor/injected_search.sub | tail -1 | tee -a "${log}"
            done
            ;;
        *) echo "Unknown stage ${stage}" >&2; exit 1 ;;
    esac
done
