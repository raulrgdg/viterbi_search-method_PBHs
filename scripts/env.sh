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
cd "${PROJECT_ROOT}"
