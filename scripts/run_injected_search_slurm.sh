#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set for Slurm jobs}"

module purge
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vit

cd "${PROJECT_ROOT}"
PACK="${3:?Usage: $0 <n_jobs> <job_id> <pack>}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python3 -u -m pipeline.injected_search.main \
    --n-jobs "$1" \
    --job-id "$2" \
    --pack "${PACK}"
