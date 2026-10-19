#!/bin/bash
# Usage: scripts/run_injected_search.sh N_JOBS JOB_ID PACK
#   Environment: SEARCH_ARGS (extra options, e.g. "--per-tsft-output"), TSFT_LIST (95 | 90 | paper), SFT_METHOD (fast | makesfts), MAKE_SFT_THREADS
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

python -u -m pbh_viterbi.workflows.injected_search \
    --n-jobs "$1" --job-id "$2" --pack "$3" \
    --tsft-list "${TSFT_LIST:-95}" --sft-method "${SFT_METHOD:-fast}" --threads "${MAKE_SFT_THREADS:-256}" ${SEARCH_ARGS:-}
