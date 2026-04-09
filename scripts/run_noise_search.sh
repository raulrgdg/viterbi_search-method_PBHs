#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate vit

cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python3 -u -m pipeline.noise_search.main \
    --n-jobs "$1" \
    --job-id "$2"
