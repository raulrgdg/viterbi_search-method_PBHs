#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set for Slurm jobs}"

module purge
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vit

cd "${PROJECT_ROOT}"
python3 -u src/noise-track_new-map-data_generation.py \
    --n-jobs "$1" \
    --job-id "$2"
