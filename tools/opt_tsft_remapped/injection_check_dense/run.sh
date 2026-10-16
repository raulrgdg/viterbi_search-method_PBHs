#!/usr/bin/env bash
# Noise background for the 45 Tsft (108 packs), then 45 injections; 8 jobs, one thread each.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.conda/envs/vit/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "$(date) background"
for k in $(seq 0 7); do $PY run.py background --n-jobs 8 --job-id $k > logs/background_job$k.log 2>&1 & done
wait
$PY run.py stats
echo "$(date) injections"
for k in $(seq 0 7); do $PY run.py inject --n-jobs 8 --job-id $k > logs/inject_job$k.log 2>&1 & done
wait
echo "$(date) ALL DONE"
