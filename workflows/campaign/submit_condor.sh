#!/bin/bash
# Submit the injection campaign on an HTCondor cluster: one Condor cluster of
# 10 jobs per pack, all packs at once (jobs are independent).
#
# Run from anywhere:
#   bash workflows/campaign/submit_condor.sh                 # the 90 packs of the paper
#   CAMPAIGN=all bash workflows/campaign/submit_condor.sh    # all 108 packs
#   PACKS="1 2 3" TSFT_LIST=90 bash workflows/campaign/submit_condor.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

PACKS="${PACKS:-$(python -m pbh_viterbi.campaign "${CAMPAIGN:-paper}")}"
mkdir -p results/logs
for pack in ${PACKS}; do
    condor_submit "pack=${pack}" "tsft_list=${TSFT_LIST:-95}" "sft_method=${SFT_METHOD:-fast}" \
        workflows/condor/injected_search.sub | tail -1
done
