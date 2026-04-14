#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate vit

cd "${PROJECT_ROOT}"
PACK="${3:?Usage: $0 <n_jobs> <job_id> <pack>}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python3 -u -m pipeline.injected_search.main \
    --n-jobs "$1" \
    --job-id "$2" \
    --pack "${PACK}"
