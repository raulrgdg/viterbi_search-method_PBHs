#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/make_SFT-SS_test.sh"

FRAMECACHE_PATH="${1:-}"
shift || true
if (( $# > 0 )); then
    THREADS_LIST="$*"
else
    THREADS_LIST="${NUM_THREADS_LIST:-16 32 64 128}"
fi

if [[ -n "${FRAMECACHE_PATH}" && ! -f "${FRAMECACHE_PATH}" ]]; then
    echo "Error: framecache no encontrado en ${FRAMECACHE_PATH}"
    exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${STRONG_SCALE_RESULTS_ROOT:-${SCRIPT_DIR}/results}"
BASE_PATH="${BASE_PATH:-${RESULTS_ROOT}/${RUN_ID}}"
SFT_BASE_PATH="${SFT_BASE_PATH:-$(mktemp -d -t sft-ss-slurm-XXXXXX)}"
mkdir -p "${BASE_PATH}"

if [[ -n "${FRAMECACHE_PATH}" ]]; then
    export framecache="${FRAMECACHE_PATH}"
fi
export BASE_PATH
export RUN_ID
export SFT_BASE_PATH
export NUM_THREADS_LIST="${THREADS_LIST}"

echo "Run ID: ${RUN_ID}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "framecache=${framecache:-auto}"
echo "NUM_THREADS_LIST=${NUM_THREADS_LIST}"
echo "BASE_PATH=${BASE_PATH}"
echo "SFT_BASE_PATH=${SFT_BASE_PATH}"

bash "${TARGET_SCRIPT}"

RESULTS_CSV="${BASE_PATH}/strong_scale_results.csv"
BEST_TXT="${BASE_PATH}/best_threads.txt"

if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "Error: no se encontro ${RESULTS_CSV}"
    exit 1
fi

awk -F',' '
NR == 2 {
    best_threads = $1
    best_time = $2 + 0
}
NR > 2 {
    current_time = $2 + 0
    if (current_time < best_time) {
        best_time = current_time
        best_threads = $1
    }
}
END {
    if (NR < 2) {
        print "No hay datos suficientes en el CSV."
        exit 1
    }
    printf "best_threads=%s\nbest_elapsed_sec=%.3f\n", best_threads, best_time
}
' "${RESULTS_CSV}" | tee "${BEST_TXT}"

echo "Resumen guardado en: ${BEST_TXT}"
