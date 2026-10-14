#!/usr/bin/env bash
# Steps 1-3 of the 14-Tsft test, in order. Waits for the download of packs 87
# and 88 (logs/download.done), then runs the noise background (8 jobs), checks
# that it reproduces the paper background for the 13 paper Tsft, and only then
# runs the 900 injections (8 jobs) and the analysis.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.conda/envs/vit/bin/python}
# One thread per process: the work is single-threaded, and the default thread
# pools (one per core) exceed the 500 processes/threads per user of the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=8
mkdir -p logs results

until [ -f logs/download.done ]; do sleep 30; done
echo "$(date) step 1: noise background"
for k in $(seq 0 $((N - 1))); do
    $PY step1_noise_background.py --n-jobs $N --job-id $k > logs/step1_job$k.log 2>&1 &
done
wait
$PY step1_noise_background.py --stats | tee logs/step1_stats.log

$PY - <<'EOF'
import csv, sys
rows = [r for r in csv.DictReader(open("results/noise_background_14tsft.csv")) if r["rel_diff_mean"] != ""]
worst = max(max(abs(float(r["rel_diff_mean"])), abs(float(r["rel_diff_std"]))) for r in rows)
n = {r["n_packs"] for r in csv.DictReader(open("results/noise_background_14tsft.csv"))}
print(f"13-Tsft background vs paper: max relative difference {worst:.2e}; packs per Tsft {n}")
sys.exit(0 if worst < 1e-6 and n == {"108"} else 1)
EOF

echo "$(date) step 2: injections"
for k in $(seq 0 $((N - 1))); do
    $PY step2_injections.py --n-jobs $N --job-id $k > logs/step2_job$k.log 2>&1 &
done
wait
echo "$(date) step 3: analysis"
$PY step3_analysis.py | tee logs/step3_analysis.log
echo "$(date) ALL DONE"
