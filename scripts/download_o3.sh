#!/bin/bash
# Usage: scripts/download_o3.sh N_JOBS JOB_ID [PACKS]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

python -u -m pbh_viterbi.o3.download --n-jobs "$1" --job-id "$2" --packs "${3:-all}"
