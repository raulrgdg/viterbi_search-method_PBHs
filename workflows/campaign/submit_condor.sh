#!/bin/bash
# Submit the injection campaign on an HTCondor cluster: one Condor cluster of
# 10 jobs per pack, all packs at once (jobs are independent).
#
# Run from anywhere:
#   bash workflows/campaign/submit_condor.sh                 # the 90 packs of the paper
#   CAMPAIGN=all bash workflows/campaign/submit_condor.sh    # all 108 packs
#   PACKS="1 2 3" TSFT_LIST=90 bash workflows/campaign/submit_condor.sh
#   PBH_CONDA_ENV=vit bash workflows/campaign/submit_condor.sh  # other conda environment
# Packs whose result file is already complete (200 rows) are skipped, so an
# interrupted campaign can be resumed by running the same command again (after
# removing the held or failed jobs). Once every job has finished, run
# workflows/campaign/finalize.sh to merge any pack left unmerged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

PACKS="${PACKS:-$(python -m pbh_viterbi.campaign "${CAMPAIGN:-paper}")}"
RESULTS_DIR="${PBH_VITERBI_RESULTS:-results}"
mkdir -p "${RESULTS_DIR}/logs"
for pack in ${PACKS}; do
    out="${RESULTS_DIR}/search/search_results_injected_pack-${pack}.csv"
    if [[ -f "${out}" && "$(($(wc -l < "${out}") - 1))" -ge 200 ]]; then
        echo "pack ${pack} already complete, skipped"
        continue
    fi
    condor_submit "pack=${pack}" "tsft_list=${TSFT_LIST:-95}" "sft_method=${SFT_METHOD:-fast}" \
        "conda_env=${PBH_CONDA_ENV:-pbh-viterbi}" "results_dir=${RESULTS_DIR}" \
        workflows/condor/injected_search.sub | tail -1
done
