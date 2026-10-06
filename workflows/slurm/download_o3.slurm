#!/bin/bash
# Download the 108 O3b packs from GWOSC. Submit from the repository root:
#   sbatch workflows/slurm/download_o3.slurm
#SBATCH -J download-o3
#SBATCH -o results/logs/download_o3.%A.%a.out
#SBATCH -e results/logs/download_o3.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 08:00:00
#SBATCH --array=0-4
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

bash scripts/download_o3.sh "${SLURM_ARRAY_TASK_COUNT}" "${SLURM_ARRAY_TASK_ID}" "${PACKS:-all}"
