#!/bin/bash
#SBATCH -J signal-generation
#SBATCH -o outputs/logs/slurm_signal.%A.%a.out
#SBATCH -e outputs/logs/slurm_signal.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -t 08:00:00
#SBATCH --array=0-340

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

N_JOBS="${N_JOBS:-341}"
PACK="${PACK:-3}"
JOB_ID="${SLURM_ARRAY_TASK_ID}"

bash bin/run_signal_slurm.sh "${N_JOBS}" "${JOB_ID}" "${PACK}"
