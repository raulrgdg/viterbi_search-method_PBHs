#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SUBMIT_FILE="${SUBMIT_FILE:-workflows/condor/run_injected_search.sub}"
PACKS="${PACKS:-1 2 3 4 5 6 7 8 9 10 11 12 37 38 39 40 41 42 43 44 45 46 47 48 73 74 75 76 77 78 79 80 81 82 83 84}"

mkdir -p results/logs

for pack in ${PACKS}; do
    tmp_sub="$(mktemp -t injected-pack-${pack}-XXXXXX.sub)"
    condor_log="results/logs/injected_search_pack-${pack}.condor.log"

    rm -f "${condor_log}"
    sed \
        -e "s/^pack=.*/pack=${pack}/" \
        -e "s|^arguments = .*|arguments = scripts/run_injected_search.sh \$(n_jobs) \$(Process) \$(pack)|" \
        -e "s|^log = .*|log = ${condor_log}|" \
        -e "s|^output = .*|output = results/logs/injected_search_pack-${pack}.\$(ClusterId).\$(Process).out|" \
        -e "s|^error = .*|error = results/logs/injected_search_pack-${pack}.\$(ClusterId).\$(Process).err|" \
        "${SUBMIT_FILE}" > "${tmp_sub}"

    echo "Submitting Condor injected search for pack=${pack}"
    condor_submit "${tmp_sub}"

    echo "Waiting for Condor pack=${pack} to finish via ${condor_log}"
    condor_wait "${condor_log}"

    rm -f "${tmp_sub}"
    echo "Finished Condor pack=${pack}"
done

echo "All Condor packs finished."
