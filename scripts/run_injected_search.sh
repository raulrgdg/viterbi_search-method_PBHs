#!/bin/bash
# Usage: scripts/run_injected_search.sh N_JOBS JOB_ID PACK
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

python -u -m pbh_viterbi.workflows.injected_search \
    --n-jobs "$1" --job-id "$2" --pack "$3" \
    --threads "${MAKE_SFT_THREADS:-256}"
