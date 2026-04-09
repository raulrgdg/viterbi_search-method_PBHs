#!/bin/bash
set -euo pipefail

# Strong scaling test using two bundled O3b strain frames.
# Override any value via environment variables or pass thread counts as args.

# Resultado para Pizza cluster:  hasta threads=64, el tiempo disminuye proporcionalmente (la mitad cuando aumento 2x los threads). Después:
# threads=128, t=12.160 ; threads=256, t=11.340 ; threads=512, t=11.204 -- alcanza el plateau en alrededor de threads=256. 

t_start=${t_start:-1257029632}
frame_length=${frame_length:-4096}
num_frames=${num_frames:-1}
t_end=${t_end:-$((t_start + frame_length * num_frames))}
Tseg=${Tseg:-8}
fmin=${fmin:-61.1}
fmax=${fmax:-126.8}
windowtype=${windowtype:-rectangular}
channel_name=${channel_name:-H1:GWOSC-4KHZ_R1_STRAIN}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/data}
RESULTS_ROOT=${RESULTS_ROOT:-${SCRIPT_DIR}/results}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

if [[ -z "${Band:-}" ]]; then
    Band=$(awk "BEGIN {printf \"%.1f\", ${fmax} - ${fmin}}")
fi

build_framecache_from_data_dir() {
    local source_dir="$1"
    local output_path="$2"
    local gwf_file
    local gwf_name
    local gps_start
    local duration

    : > "$output_path"
    for gwf_file in "${source_dir}"/*.gwf; do
        if [[ ! -f "$gwf_file" ]]; then
            continue
        fi

        gwf_name="$(basename "$gwf_file" .gwf)"
        if [[ "$gwf_name" =~ -([0-9]+)-([0-9]+)_ ]]; then
            gps_start="${BASH_REMATCH[1]}"
            duration="${BASH_REMATCH[2]}"
        else
            echo "Error: cannot infer GPS start/duration from $gwf_file"
            return 1
        fi

        printf 'H %s %s %s file://localhost%s\n' \
            "$gwf_name" "$gps_start" "$duration" "$(cd "$(dirname "$gwf_file")" && pwd)/$(basename "$gwf_file")" \
            >> "$output_path"
    done

    if [[ ! -s "$output_path" ]]; then
        echo "Error: no .gwf files found in $source_dir"
        return 1
    fi
}

framecache=${framecache:-${DATA_DIR}/framecache_raw_strain_512HZ}
if [[ ! -f "$framecache" ]]; then
    framecache="${SFT_BASE_PATH:-$(mktemp -d -t sft-strong-scale-test-v2-XXXXXX)}/framecache_raw_strain_512HZ"
    build_framecache_from_data_dir "$DATA_DIR" "$framecache"
fi

if grep -q '/lhome/\|/home/' "$framecache"; then
    temp_framecache_dir="$(mktemp -d -t strong-scale-framecache-XXXXXX)"
    temp_framecache="${temp_framecache_dir}/framecache_raw_strain_512HZ"
    build_framecache_from_data_dir "$DATA_DIR" "$temp_framecache"
    framecache="$temp_framecache"
fi

SFT_BASE_PATH=${SFT_BASE_PATH:-$(mktemp -d -t sft-strong-scale-test-v2-XXXXXX)}
BASE_PATH=${BASE_PATH:-${RESULTS_ROOT}/${RUN_ID}}
trap 'rm -rf "$SFT_BASE_PATH" "${temp_framecache_dir:-}"' EXIT

mkdir -p "$BASE_PATH"

if [[ ! -f "$framecache" ]]; then
    echo "Error: framecache not found at $framecache"
    exit 1
fi

if (( t_end <= t_start )); then
    echo "Error: t_end ($t_end) must be greater than t_start ($t_start)"
    exit 1
fi

duration=$((t_end - t_start))
if (( duration % Tseg != 0 )); then
    echo "Error: Total duration ($duration) is not divisible by Tseg ($Tseg)."
    exit 1
fi

if (( $# > 0 )); then
    num_threads_list=("$@")
else
    num_threads_list=(${NUM_THREADS_LIST:-"16 32 64 128"})
fi

results_csv="${BASE_PATH}/strong_scale_results.csv"
if [[ ! -f "$results_csv" ]]; then
    echo "threads,elapsed_sec,t_start,t_end,Tseg,Band,fmin,windowtype,channel" > "$results_csv"
fi

echo "Strong scaling test starting..."
echo "t_start=$t_start t_end=$t_end duration=$duration Tseg=$Tseg Band=$Band fmin=$fmin windowtype=$windowtype"
echo "framecache=$framecache"
echo "SFT_BASE_PATH=$SFT_BASE_PATH"

for threads in "${num_threads_list[@]}"; do
    if ! [[ "$threads" =~ ^[0-9]+$ ]]; then
        echo "Skipping invalid thread count: $threads"
        continue
    fi
    if (( threads <= 0 )); then
        echo "Skipping non-positive thread count: $threads"
        continue
    fi

    if (( duration % threads != 0 )); then
        echo "Skipping threads=$threads: duration $duration not divisible by threads."
        continue
    fi

    tanda_duration=$((duration / threads))
    if (( tanda_duration % Tseg != 0 )); then
        echo "Skipping threads=$threads: segment duration $tanda_duration not divisible by Tseg $Tseg."
        continue
    fi

    SFTPATH="${SFT_BASE_PATH}/threads-${threads}"
    mkdir -p "$SFTPATH"

    echo "Running threads=$threads with segment duration=$tanda_duration"
    start_ns=$(date +%s%N)

    for ((i=0; i<threads; i++)); do
        gps_start=$((t_start + i * tanda_duration))
        gps_end=$((gps_start + tanda_duration))

        lalpulsar_MakeSFTs -f "$fmin" -t "$Tseg" -p "$SFTPATH" -C "$framecache" \
            -s "$gps_start" -e "$gps_end" -O 0 -N "$channel_name" \
            -w "$windowtype" -F 0 -B "$Band" -X MSFT &
    done

    wait
    end_ns=$(date +%s%N)
    elapsed_sec=$(awk "BEGIN {printf \"%.3f\", (${end_ns} - ${start_ns})/1000000000}")
    echo "threads=$threads elapsed_sec=$elapsed_sec"
    echo "$threads,$elapsed_sec,$t_start,$t_end,$Tseg,$Band,$fmin,$windowtype,$channel_name" >> "$results_csv"
done

echo "Strong scaling test completed. Results: $results_csv"
