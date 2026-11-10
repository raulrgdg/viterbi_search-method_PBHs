#!/bin/bash
# Run a search stage on a single machine with N parallel processes (one thread each).
#
# Usage: scripts/run_local.sh STAGE N_PROC
#   download     download the O3b packs from GWOSC (141; 1-108 used in the paper)
#   background   noise track power of PACKS (default all) for the Tsft of every list, merged into
#                calibration/noise_track_power<suffix>.csv, and the nsigma background of every list
#                (calibration/noise_background_tsft<list><suffix>.csv), suffix "_chunkhp_sftnorm"
#                for the default maps of v1.3 (paths.config_suffix). To add new noise packs:
#                PACKS="109-120" bash scripts/run_local.sh background 8
#   noise        noise search of PACKS (default all, 141) -> results/search/search_results_noise.csv
#   injected     injection campaign: every pack of CAMPAIGN (paper: 90 packs, all: 108)
#                with its 200-signal slice -> results/search/search_results_injected_pack-<pack>.csv
#
# Environment: TSFT_LIST (95 | 90 | paper, default 95), SFT_METHOD (fast | makesfts),
#              CAMPAIGN (paper | all), PACKS (background/noise, default all),
#              PBH_VITERBI_O3_DIR (location of the O3b-packN folders).
# Packs whose injected-search file is already complete (200 rows) are skipped, so an
# interrupted campaign can be resumed by running the same command again.
#
# Example (run it inside tmux or screen):
#   TSFT_LIST=95 bash scripts/run_local.sh background 8
#   TSFT_LIST=95 bash scripts/run_local.sh injected 8
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

STAGE="${1:?stage: download | background | noise | injected}"
N_PROC="${2:-8}"
TSFT_LIST="${TSFT_LIST:-95}"
SFT_METHOD="${SFT_METHOD:-fast}"
CAMPAIGN="${CAMPAIGN:-paper}"
export TSFT_LIST SFT_METHOD
mkdir -p results/logs results/search
log() { echo "$(date '+%F %T') $*"; }

run_jobs() {  # run_jobs MODULE ARGS... : N_PROC jobs of a sharded module, then wait
    local module="$1"; shift
    for ((k = 0; k < N_PROC; k++)); do
        python -u -m "${module}" "$@" --n-jobs "${N_PROC}" --job-id "${k}" \
            > "results/logs/${STAGE}_job${k}.log" 2>&1 &
    done
    wait
}

case "${STAGE}" in
    download)
        run_jobs pbh_viterbi.o3.download --packs all
        ;;
    background)
        all_tsft="$(python -c 'from pbh_viterbi.config import TSFT_LISTS; print(*sorted(set().union(*TSFT_LISTS.values())))')"
        log "Noise track power of packs ${PACKS:-all}, Tsft of every list, ${N_PROC} processes"
        run_jobs pbh_viterbi.workflows.noise_search --mode background --packs "${PACKS:-all}" \
            --tsft ${all_tsft} --sft-method "${SFT_METHOD}"
        python analysis/noise_background_stats.py results/search/noise_track_power.csv --add
        ;;
    noise)
        log "Noise search, Tsft list ${TSFT_LIST}, ${N_PROC} processes"
        run_jobs pbh_viterbi.workflows.noise_search --mode search --packs "${PACKS:-all}" \
            --tsft-list "${TSFT_LIST}" --sft-method "${SFT_METHOD}"
        ;;
    injected)
        packs="$(python -m pbh_viterbi.campaign "${CAMPAIGN}")"
        log "Injection campaign '${CAMPAIGN}' ($(wc -w <<< "${packs}") packs), Tsft list ${TSFT_LIST}, ${N_PROC} processes"
        for pack in ${packs}; do
            out="results/search/search_results_injected_pack-${pack}.csv"
            if [[ -f "${out}" && "$(($(wc -l < "${out}") - 1))" -ge 200 ]]; then
                log "pack ${pack} already complete, skipped" >&2
                continue
            fi
            echo "${pack}"
        done | xargs -P "${N_PROC}" -I{} sh -c \
            'python -u -m pbh_viterbi.workflows.injected_search --pack {} --tsft-list "$TSFT_LIST" --sft-method "$SFT_METHOD" > results/logs/injected_pack{}.log 2>&1 && echo "$(date "+%F %T") pack {} done" || echo "$(date "+%F %T") pack {} FAILED (results/logs/injected_pack{}.log)"'
        ;;
    *)
        echo "Unknown stage ${STAGE}" >&2
        exit 1
        ;;
esac
log "${STAGE} finished"
