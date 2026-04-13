#!/bin/bash
#SBATCH -J sft-strong-scale
#SBATCH -o studies/strong_scaling/slurm_logs/job.%j.out
#SBATCH -e studies/strong_scaling/slurm_logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=3G
#SBATCH -t 02:00:00

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_ROOT}/studies/strong_scaling"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

module purge
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vit

# Hydra quickstart batch uses sbatch scripts with #SBATCH directives.
# Hydra tutorial recommends N=1, n=1 and either --exclusive or -c <cores>
# for one-node multithreaded jobs.
export OMP_NUM_THREADS=1
export NUM_THREADS_LIST="${NUM_THREADS_LIST:-16 32 64 128 256 512}"
export STRONG_SCALE_RESULTS_ROOT="${STRONG_SCALE_RESULTS_ROOT:-${SCRIPT_DIR}/results}"

FRAMECACHE_PATH="${FRAMECACHE_PATH:-${SCRIPT_DIR}/data/framecache_raw_strain_512HZ}"

bash "${SCRIPT_DIR}/run_make_SFT-SS_test_slurm.sh" "${FRAMECACHE_PATH}" ${NUM_THREADS_LIST}
