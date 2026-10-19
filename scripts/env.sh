#!/bin/bash
# Environment set-up sourced by every job wrapper in this folder.
#
# Edit the block below for your cluster, or export PBH_ENV_SETUP before
# submitting, e.g.
#   export PBH_ENV_SETUP="module load Miniconda3; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate pbh-viterbi"
#
# The default activates the conda environment `pbh-viterbi` (environment.yml)
# through the IGWN conda installation available on LIGO clusters via CVMFS.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

if [[ -n "${PBH_ENV_SETUP:-}" ]]; then
    eval "${PBH_ENV_SETUP}"
elif [[ -f /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
    conda activate "${PBH_CONDA_ENV:-pbh-viterbi}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The search is single-threaded: one thread per process avoids oversubscribing
# shared nodes (and per-user thread limits) when many jobs run side by side.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Keep freed memory in the process instead of returning it to the kernel: the
# per-Tsft arrays are allocated and freed 45 times per chunk, and with glibc's
# default mmap of large blocks half of the CPU time went to page faults
# (system time 32-78 s -> 6 s per injection, identical results, +7% memory).
export MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=100000000000

# Search options read by the job wrappers (override before submitting):
#   TSFT_LIST   95 (default) | 90 | paper
#   SFT_METHOD  fast (default) | makesfts
cd "${PROJECT_ROOT}"
