#!/bin/bash
# Quick efficiency test (studies/multi_detector/README.md): 36 signals (quick_signals.csv) in
# 3 calibration packs + 10 evaluation packs. Every configuration passes its map options
# explicitly, so the test does not depend on the defaults:
#   A  H1, v1.2 maps (per-segment high-pass, 'bin')    B  H1, chunk high-pass, 'bin'
#   C  H1+L1, chunk high-pass, 'bin'
# plus the noise search (per-Tsft rows) of B and C on the 61 packs with L1 (A: noise of the
# v1.2 campaign). NORM=bin+sft runs B2 and C2 instead, the v1.3 default maps (chunk high-pass,
# 'bin+sft') with H1 and H1+L1.
#
#   PBH_CONDA_ENV=<conda env> bash studies/multi_detector/submit_quick_test.sh
#
# B and C need their nsigma backgrounds on the 61 packs with L1 first, e.g. for B2:
#   NORM=bin+sft bash studies/multi_detector/submit_quick_test.sh background
#   python analysis/noise_background_stats.py results/qt_B_sftnorm_background/search/noise_track_power.csv \
#       --tsft-list 95 --sft-highpass chunk --map-normalisation bin+sft
#   (C2: results/qt_C_sftnorm_background and --detectors H1 L1 as well)
# Outputs: results/qt_{A,B,C}[_sftnorm] (injections), results/qt_{B,C}[_sftnorm]_noise.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ENV="${PBH_CONDA_ENV:-pbh-viterbi}"
NORM="${NORM:-bin}"
PACKS="104 113 126 3 21 32 46 55 77 90 100 118 141"
L1_PACKS=$(grep -v '^#' calibration/l1_packs.txt | awk '{print $1}' | paste -sd, -)
declare -A ARGS=([A]="--sft-highpass segment --map-normalisation bin" [B]="--sft-highpass chunk --map-normalisation bin"
                [C]="--detectors H1 L1 --sft-highpass chunk --map-normalisation bin")
CFGS="A B C"; tag=""
if [[ "$NORM" == bin+sft ]]; then
    CFGS="B C"; tag="_sftnorm"
    for c in B C; do ARGS[$c]="${ARGS[$c]/--map-normalisation bin/--map-normalisation bin+sft}"; done
elif [[ "$NORM" != bin ]]; then
    echo "NORM must be bin or bin+sft"; exit 1
fi

if [[ "${1:-search}" == background ]]; then
    for cfg in $CFGS; do
        [[ "$cfg" == A ]] && continue
        out="results/qt_${cfg}${tag}_background"; mkdir -p "$out/logs"
        condor_submit n_jobs=61 mode=background "packs=${L1_PACKS}" "conda_env=${ENV}" "results_dir=${out}" memory=4GB \
            "search_args=${ARGS[$cfg]}" workflows/condor/noise_search.sub | tail -1
    done
    exit 0
fi

for cfg in $CFGS; do
    [[ "$cfg" == A ]] && continue
    out="results/qt_${cfg}${tag}_noise"; mkdir -p "$out/logs"
    condor_submit n_jobs=61 mode=search "packs=${L1_PACKS}" "conda_env=${ENV}" "results_dir=${out}" memory=4GB \
        "search_args=--per-tsft-output ${ARGS[$cfg]}" workflows/condor/noise_search.sub | tail -1
done
for cfg in $CFGS; do
    out="results/qt_${cfg}${tag}"; mkdir -p "$out/logs"
    for pack in $PACKS; do
        condor_submit "pack=${pack}" n_jobs=9 "conda_env=${ENV}" "results_dir=${out}" memory=5GB \
            "search_args=--signal-list studies/multi_detector/quick_signals.csv --per-tsft-output ${ARGS[$cfg]}" \
            workflows/condor/injected_search.sub | tail -1
    done
done
