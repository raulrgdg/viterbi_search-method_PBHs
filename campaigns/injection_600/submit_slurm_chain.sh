#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SUBMIT_FILE="${SUBMIT_FILE:-workflows/slurm/run_injected_search.slurm}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterok}"
CLUSTER="${CLUSTER:-HPC2}"

case "${CLUSTER}" in
    HPC2)
        DEFAULT_PACKS="13 14 15 16 17 18 19 20 21 22 23 24 49 50 51 52 53 54 55 56 57 58 59 60 85 86 87 88 89 90 91 92 93 94 95 96"
        ;;
    HPC3)
        DEFAULT_PACKS="25 26 27 28 29 30 31 32 33 34 35 36 61 62 63 64 65 66 67 68 69 70 71 72 97 98 99 100 101 102 103 104 105 106 107 108"
        ;;
    *)
        echo "Error: CLUSTER must be HPC2 or HPC3, got ${CLUSTER}" >&2
        exit 1
        ;;
esac

PACKS="${PACKS:-${DEFAULT_PACKS}}"
previous_job_id=""

for pack in ${PACKS}; do
    if [[ -z "${previous_job_id}" ]]; then
        job_id="$(sbatch --parsable --export=ALL,PACK="${pack}" "${SUBMIT_FILE}")"
    else
        job_id="$(sbatch --parsable --dependency="${DEPENDENCY_TYPE}:${previous_job_id}" --export=ALL,PACK="${pack}" "${SUBMIT_FILE}")"
    fi

    echo "Submitted ${CLUSTER} pack=${pack} job_id=${job_id} dependency_on=${previous_job_id:-none}"
    previous_job_id="${job_id}"
done

echo "Submitted ${CLUSTER} Slurm chain ending at job_id=${previous_job_id}"
