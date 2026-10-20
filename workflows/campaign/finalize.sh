#!/bin/bash
# Merge the shards of every campaign pack whose jobs have all finished.
#
# Each job merges its pack when it is the last one to finish, but on a shared
# filesystem two jobs can miss each other's .done marker and leave the pack
# unmerged. Run this once the campaign has finished (it is harmless to repeat):
#   bash workflows/campaign/finalize.sh            # N_JOBS=20 jobs per pack
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"  # activates the environment and cd to the repo root

PACKS="${PACKS:-$(python -m pbh_viterbi.campaign "${CAMPAIGN:-paper}")}"
python - "${N_JOBS:-20}" ${PACKS} <<'PY'
import logging, sys
from pbh_viterbi.paths import SEARCH_RESULTS_DIR
from pbh_viterbi.results import (FIELDNAMES, PER_TSFT_FIELDNAMES, csv_name, merge_shards_if_ready,
                                  per_tsft_csv_name, shard_path)

logging.basicConfig(level=logging.WARNING)
n_jobs, packs = int(sys.argv[1]), [int(p) for p in sys.argv[2:]]
complete, pending, per_tsft_packs = 0, [], 0
for pack in packs:
    # main result file and, if the jobs wrote it (--per-tsft-output), the per-Tsft file
    for name, fields in ((csv_name(injected=True, pack=pack), FIELDNAMES),
                         (per_tsft_csv_name(injected=True, pack=pack), PER_TSFT_FIELDNAMES)):
        if any(shard_path(name, n_jobs, j).exists() for j in range(n_jobs)):
            merge_shards_if_ready(name, n_jobs, fields)
    output = SEARCH_RESULTS_DIR / csv_name(injected=True, pack=pack)
    rows = sum(1 for _ in output.open()) - 1 if output.exists() else 0
    per_tsft_packs += (SEARCH_RESULTS_DIR / per_tsft_csv_name(injected=True, pack=pack)).exists()
    if rows >= 200:
        complete += 1
    else:
        pending.append(f"{pack}({rows})")
print(f"{complete}/{len(packs)} packs complete; per-Tsft files for {per_tsft_packs} packs")
if pending:
    print("incomplete (rows):", *pending)
PY
