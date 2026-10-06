#!/bin/bash
# Submit the injected search of the HPC1 (Condor) share of the 600-signal
# campaign, one pack after another: each pack is a 200-job cluster and the next
# pack is submitted once the previous cluster has finished.
#
# Run from anywhere:
#   bash workflows/campaign_600/submit_condor_chain.sh
#   PACKS="1 2 3" bash workflows/campaign_600/submit_condor_chain.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

PACKS="${PACKS:-$(python -c 'from pbh_viterbi.campaign import packs_for_cluster; print(*packs_for_cluster("HPC1"))')}"
mkdir -p results/logs

for pack in ${PACKS}; do
    echo "Submitting injected search for pack ${pack}"
    cluster_id="$(condor_submit "pack=${pack}" workflows/condor/injected_search.sub | sed -n 's/.*submitted to cluster \([0-9]*\).*/\1/p')"
    log_file="results/logs/injected_search_pack-${pack}.${cluster_id}.log"
    echo "Waiting for cluster ${cluster_id} (${log_file})"
    condor_wait "${log_file}"
done
echo "All Condor packs finished."
