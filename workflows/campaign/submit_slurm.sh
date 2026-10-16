#!/bin/bash
# Submit the injection campaign on a Slurm cluster: one 10-task job array per
# pack, all packs at once (jobs are independent).
#
#   bash workflows/campaign/submit_slurm.sh                  # the 90 packs of the paper
#   CAMPAIGN=all bash workflows/campaign/submit_slurm.sh     # all 108 packs
#   PACKS="25 26" TSFT_LIST=90 bash workflows/campaign/submit_slurm.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

PACKS="${PACKS:-$(python -m pbh_viterbi.campaign "${CAMPAIGN:-paper}")}"
mkdir -p results/logs
for pack in ${PACKS}; do
    job="$(sbatch --parsable --export=ALL,PACK="${pack}",TSFT_LIST="${TSFT_LIST:-95}",SFT_METHOD="${SFT_METHOD:-fast}" \
        workflows/slurm/injected_search.slurm)"
    echo "pack ${pack} -> job ${job}"
done
