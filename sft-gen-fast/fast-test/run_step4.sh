#!/usr/bin/env bash
# Step 4: optimal-Tsft formula check (100 injections, 8 jobs, one thread each).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.conda/envs/vit/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=8
echo "$(date) step 4"
for k in $(seq 0 $((N - 1))); do
    $PY step4_tsft_formula.py --n-jobs $N --job-id $k > logs/step4_job$k.log 2>&1 &
done
wait
$PY step4_tsft_formula.py --summary | tee logs/step4_summary.log
echo "$(date) STEP4 DONE"
