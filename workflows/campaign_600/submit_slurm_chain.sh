#!/bin/bash
# Submit the injected search of the HPC2 or HPC3 (Slurm) share of the
# 600-signal campaign as a chain of job arrays, one per pack; each array starts
# when the previous one ends (DEPENDENCY_TYPE=afterok by default, use afterany
# to continue after failures).
#
#   CLUSTER=HPC2 bash workflows/campaign_600/submit_slurm_chain.sh
#   CLUSTER=HPC3 PACKS="25 26" bash workflows/campaign_600/submit_slurm_chain.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

CLUSTER="${CLUSTER:-HPC2}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterok}"
PACKS="${PACKS:-$(python -c "from pbh_viterbi.campaign import packs_for_cluster; print(*packs_for_cluster('${CLUSTER}'))")}"
mkdir -p results/logs

previous=""
for pack in ${PACKS}; do
    dependency=()
    [[ -n "${previous}" ]] && dependency=(--dependency="${DEPENDENCY_TYPE}:${previous}")
    previous="$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --export=ALL,PACK="${pack}" workflows/slurm/injected_search.slurm)"
    echo "${CLUSTER}: pack ${pack} -> job ${previous}"
done
echo "Chain submitted; last job ${previous}."
