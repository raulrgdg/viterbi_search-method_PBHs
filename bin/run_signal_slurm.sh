#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

module purge
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vit

cd "${PROJECT_ROOT}"
python3 -u src/data_generation-new_pipeline.py \
    --n-jobs "$1" \
    --job-id "$2" \
    --pack "${3:-3}"
