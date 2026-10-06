#!/bin/bash
# Strong-scaling benchmark of lalpulsar_MakeSFTs: time to produce the SFTs of
# the same data stretch while splitting it across N parallel processes.
#
# Usage (from anywhere):
#   bash studies/strong_scaling/strong_scaling.sh 16 32 64 128 256 512
#
# Environment overrides:
#   PACK (3)  NUM_FRAMES (1)  Tseg (8)  fmin (61.1)  fmax (126.8)
#   RESULTS_DIR (results/strong_scaling/<date>)
# Requires the pack to be downloaded (python -m pbh_viterbi.o3.download --packs 3).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
export OMP_NUM_THREADS=1

PACK=${PACK:-3}
NUM_FRAMES=${NUM_FRAMES:-1}
Tseg=${Tseg:-8}
fmin=${fmin:-61.1}
fmax=${fmax:-126.8}
RESULTS_DIR=${RESULTS_DIR:-results/strong_scaling/$(date +%Y%m%d_%H%M%S)}
threads_list=(16 32 64 128)
(( $# > 0 )) && threads_list=("$@")

work_dir="$(mktemp -d -t strong-scaling-XXXXXX)"
trap 'rm -rf "${work_dir}"' EXIT
mkdir -p "${RESULTS_DIR}"

read -r t_start framecache < <(python - "${PACK}" "${NUM_FRAMES}" "${work_dir}" <<'PY'
import sys
from pbh_viterbi.o3.frames import write_raw_framecache
from pbh_viterbi.o3.packs import pack_window
from pbh_viterbi.paths import o3_pack_dir
pack, n_frames, work = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t_start = pack_window(pack)[0]
print(t_start, write_raw_framecache(f"{work}/framecache", o3_pack_dir(pack), t_start, num_frames=n_frames))
PY
)
duration=$((NUM_FRAMES * 4096))
t_end=$((t_start + duration))
Band=$(awk "BEGIN {printf \"%.1f\", ${fmax} - ${fmin}}")

results_csv="${RESULTS_DIR}/strong_scaling.csv"
echo "threads,elapsed_sec,t_start,t_end,Tseg,Band" > "${results_csv}"
echo "Pack ${PACK}: [${t_start}, ${t_end}), Tseg=${Tseg} s, band ${fmin}+${Band} Hz"

for threads in "${threads_list[@]}"; do
    chunk=$((duration / threads))
    if (( duration % threads != 0 || chunk % Tseg != 0 )); then
        echo "Skipping threads=${threads}: the stretch cannot be split evenly."
        continue
    fi
    sft_dir="${work_dir}/threads-${threads}"
    mkdir -p "${sft_dir}"
    start_ns=$(date +%s%N)
    for ((i = 0; i < threads; i++)); do
        s=$((t_start + i * chunk))
        lalpulsar_MakeSFTs -f "${fmin}" -t "${Tseg}" -p "${sft_dir}" -C "${framecache}" \
            -s "${s}" -e "$((s + chunk))" -O 0 -N H1:GWOSC-4KHZ_R1_STRAIN \
            -w rectangular -F 0 -B "${Band}" -X MSFT >/dev/null &
    done
    wait
    elapsed=$(awk "BEGIN {printf \"%.3f\", ($(date +%s%N) - ${start_ns}) / 1e9}")
    echo "threads=${threads} elapsed=${elapsed} s"
    echo "${threads},${elapsed},${t_start},${t_end},${Tseg},${Band}" >> "${results_csv}"
    rm -rf "${sft_dir}"
done

sort -t, -k2 -g <(tail -n +2 "${results_csv}") | head -1 | awk -F, '{print "Fastest: threads=" $1 ", " $2 " s"}'
echo "Results: ${results_csv}"
