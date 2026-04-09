#!/bin/bash
#SBATCH -J sft-strong-scale
#SBATCH -o strong_scale_test/slurm_logs/job.%j.out
#SBATCH -e strong_scale_test/slurm_logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH --mem=32G
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
