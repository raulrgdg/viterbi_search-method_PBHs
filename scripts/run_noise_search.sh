#!/bin/bash
# Usage: scripts/run_noise_search.sh N_JOBS JOB_ID [MODE] [PACKS]
#   MODE: search (default) or background
#   Environment: TSFT_LIST (95 | 90 | paper), SFT_METHOD (fast | makesfts), MAKE_SFT_THREADS
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

python -u -m pbh_viterbi.workflows.noise_search \
    --n-jobs "$1" --job-id "$2" --mode "${3:-search}" --packs "${4:-all}" \
    --tsft-list "${TSFT_LIST:-95}" --sft-method "${SFT_METHOD:-fast}" --threads "${MAKE_SFT_THREADS:-256}"
