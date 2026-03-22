#!/bin/bash
set -euo pipefail

# Resolve the repository root so Condor can launch from any working directory.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate vit

# Preserve the repo-relative import/layout assumptions used by the Python entrypoint.
cd "${PROJECT_ROOT}"
python3 -u src/search_candidate.py \
    --n-jobs "$1" \
    --job-id "$2"
