#!/usr/bin/env bash
# Steps 2-3 only (step 1 done). The paper background is reproduced exactly with
# the 61.1-127.2 Hz band (noise search band), not 126.8 Hz; Tsft = 1 s gives
# the same 66 bins with both bands, so its step 1 background is valid as is.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.conda/envs/vit/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=8
echo "$(date) step 2: injections"
for k in $(seq 0 $((N - 1))); do
    $PY step2_injections.py --n-jobs $N --job-id $k > logs/step2_job$k.log 2>&1 &
done
wait
echo "$(date) step 3: analysis"
$PY step3_analysis.py | tee logs/step3_analysis.log
echo "$(date) ALL DONE"
