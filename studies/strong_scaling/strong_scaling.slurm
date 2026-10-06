#!/bin/bash
# Submit from the repository root:  sbatch [--exclusive] studies/strong_scaling/strong_scaling.slurm
#SBATCH -J strong-scaling
#SBATCH -o results/logs/strong_scaling.%j.out
#SBATCH -e results/logs/strong_scaling.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=1G
#SBATCH -t 01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash studies/strong_scaling/strong_scaling.sh ${NUM_THREADS_LIST:-16 32 64 128 256 512}
