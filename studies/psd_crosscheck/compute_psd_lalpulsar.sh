#!/usr/bin/env bash
# Cross-check of the average noise PSD (Appendix A) with lalpulsar_ComputePSD.
# Same segmentation as our Welch estimate: 512 s Hann-windowed SFTs with 50%
# overlap (lalpulsar_MakeSFTs), then lalpulsar_ComputePSD with its defaults
# (running median over 101 bins, harmonic mean over SFTs).
#
#   bash studies/psd_crosscheck/compute_psd_lalpulsar.sh [packs...]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PACKS=("${@:-1 4 13 25 37 49}")
PACKS=(${PACKS[@]})
TSFT=512; FMIN=10; BAND=246; CHANNEL="H1:GWOSC-4KHZ_R1_STRAIN"
OUT="$HERE/output"
mkdir -p "$OUT"

sfts=()
for pack in "${PACKS[@]}"; do
    pdir="$ROOT/data/o3/O3b-pack${pack}"
    [[ -d "$pdir" ]] || { echo "skip pack $pack"; continue; }
    sftdir="$OUT/sfts_pack${pack}"; mkdir -p "$sftdir"
    cache="$OUT/pack${pack}.cache"; : > "$cache"
    files=("$pdir"/*.gwf)
    for f in "${files[@]}"; do
        b=$(basename "$f" .gwf); gps=$(echo "$b" | awk -F- '{print $3}')
        echo "H $b $gps 4096 file://localhost$f" >> "$cache"
    done
    t0=$(basename "${files[0]}" | awk -F- '{print $3}')
    t1=$(( $(basename "${files[-1]}" | awk -F- '{print $3}') + 4096 ))
    lalpulsar_MakeSFTs -f "$FMIN" -t "$TSFT" -p "$sftdir" -C "$cache" -s "$t0" -e "$t1" \
        -O 0 -N "$CHANNEL" -w hann -P 0.5 -F "$FMIN" -B "$BAND" -X PSDX >/dev/null
    sfts+=("$sftdir"/*.sft)
    echo "pack $pack: SFTs done"
done

# lalpulsar does not expand wildcards in directories: pass a ';'-separated list.
lalpulsar_ComputePSD -i "$(IFS=';'; echo "${sfts[*]}")" -o "$OUT/computepsd_T${TSFT}.txt" \
    --Freq 15 --FreqBand 230 --blocksRngMed 101 --PSDmthopSFTs harmmean

