#!/bin/bash
# Noise-only search, one pack per array task. Submit from the repository root:
#   sbatch workflows/slurm/noise_search.slurm
#   sbatch --export=ALL,MODE=background workflows/slurm/noise_search.slurm
#   sbatch --export=ALL,TSFT_LIST=90 workflows/slurm/noise_search.slurm
# In-memory SFTs: one core, ~2 min per pack (for SFT_METHOD=makesfts use -c 16 --mem=8G).
#SBATCH -J noise-search
#SBATCH -o results/logs/noise_search.%A.%a.out
#SBATCH -e results/logs/noise_search.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH --array=0-107
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

bash scripts/run_noise_search.sh "${SLURM_ARRAY_TASK_COUNT}" "${SLURM_ARRAY_TASK_ID}" "${MODE:-search}" "${PACKS:-all}"
