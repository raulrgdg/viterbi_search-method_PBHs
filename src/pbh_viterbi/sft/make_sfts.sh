#!/bin/bash
# Generate the SFTs of one pack for one coherence time with lalpulsar_MakeSFTs.
#
# The [t_start, t_end) interval is split into `num_threads` contiguous chunks,
# each handled by an independent MakeSFTs process running in the background.
# Called from pbh_viterbi/sft/make_sfts.py; all inputs come from environment
# variables:
#   t_start, t_end   GPS interval to process
#   Tseg             SFT duration (s)
#   num_threads      maximum number of parallel MakeSFTs processes
#   SFTPATH          output directory
#   framecache       frame cache listing the input GWF files
#   fmin, Band       lowest frequency and bandwidth (Hz)
#   windowtype       MakeSFTs window (e.g. rectangular)
#   channel_name     strain channel
#   remainder_mode   strict (fail) | trim (drop the tail) if Tseg does not divide the interval
#   sft_verbose      1 to print progress
set -euo pipefail

t_start=${t_start}
t_end=${t_end}
num_threads=${num_threads}
Tseg=${Tseg}
SFTPATH=${SFTPATH}
framecache=${framecache}
Band=${Band}
fmin=${fmin}
windowtype=${windowtype}
channel_name=${channel_name}
remainder_mode=${remainder_mode:-strict}
sft_verbose=${sft_verbose:-0}

mkdir -p "$SFTPATH"

if [[ -z "${t_start}" || -z "${t_end}" || -z "${Tseg}" || -z "${num_threads}" ]]; then
    echo "Error: Missing required env vars (t_start, t_end, Tseg, num_threads)."
    exit 1
fi

if (( t_end <= t_start )); then
    echo "Error: t_end must be greater than t_start."
    exit 1
fi

if (( Tseg <= 0 )); then
    echo "Error: Tseg must be > 0."
    exit 1
fi

if (( num_threads <= 0 )); then
    echo "Error: num_threads must be > 0."
    exit 1
fi

duration=$((t_end - t_start))
full_sfts=$((duration / Tseg))
remainder=$((duration % Tseg))

if (( full_sfts == 0 )); then
    echo "Error: duration (${duration}) is shorter than Tseg (${Tseg})."
    exit 1
fi

if (( remainder != 0 )); then
    if [[ "$remainder_mode" == "strict" ]]; then
        echo "Error: duration (${duration}) is not divisible by Tseg (${Tseg})."
        echo "Hint: set remainder_mode=trim to drop the tail (${remainder}s)."
        exit 1
    elif [[ "$remainder_mode" == "trim" ]]; then
        echo "Warning: trimming tail of ${remainder}s (duration not divisible by Tseg)."
    else
        echo "Error: remainder_mode must be 'strict' or 'trim'."
        exit 1
    fi
fi

workers=$num_threads
if (( workers > full_sfts )); then workers=$full_sfts; fi
if (( workers < 1 )); then workers=1; fi

if [[ "$sft_verbose" == "1" ]]; then
    echo "duration=${duration}s, Tseg=${Tseg}s, full_sfts=${full_sfts}, remainder=${remainder}s"
    echo "requested_threads=${num_threads}, workers=${workers}"
fi

base=$((full_sfts / workers))
extra=$((full_sfts % workers))

current_start=$t_start
pids=()
for ((i=0; i<workers; i++)); do
    seg_count=$base
    if (( i < extra )); then
        seg_count=$((seg_count + 1))
    fi

    if (( seg_count <= 0 )); then
        continue
    fi

    chunk_duration=$((seg_count * Tseg))
    gps_start=$current_start
    gps_end=$((gps_start + chunk_duration))

    if [[ "$sft_verbose" == "1" ]]; then
        echo "Worker ${i}: ${seg_count} SFTs | GPS start=${gps_start}, end=${gps_end}"
    fi

    if [[ "$sft_verbose" == "1" ]]; then
        lalpulsar_MakeSFTs -f "$fmin" -t "$Tseg" -p "$SFTPATH" -C "$framecache" \
            -s "$gps_start" -e "$gps_end" -O 0 -N "$channel_name" \
            -w "$windowtype" -F "$fmin" -B "$Band" -X MSFT &
    else
        lalpulsar_MakeSFTs -f "$fmin" -t "$Tseg" -p "$SFTPATH" -C "$framecache" \
            -s "$gps_start" -e "$gps_end" -O 0 -N "$channel_name" \
            -w "$windowtype" -F "$fmin" -B "$Band" -X MSFT >/dev/null &
    fi
    pids+=($!)

    current_start=$gps_end
done

# Fail if any MakeSFTs worker failed.
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "Error: at least one lalpulsar_MakeSFTs worker failed." >&2
    exit 1
fi

effective_end=$((t_start + full_sfts * Tseg))
if [[ "$sft_verbose" == "1" ]]; then
    if (( remainder != 0 )) && [[ "$remainder_mode" == "trim" ]]; then
        echo "Completed with trim: processed [$t_start, $effective_end), dropped [$effective_end, $t_end)."
    else
        echo "Parallel SFT processing completed: processed [$t_start, $effective_end)."
    fi
fi
