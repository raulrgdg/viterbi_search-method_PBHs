#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

PACKS="${PACKS:-25,26,27,28,29,30,31,32,33,34,35,36,61,62,63,64,65,66,67,68,69,70,71,72,97,98,99,100,101,102,103,104,105,106,107,108}"
LOCAL_O3_DIR="${LOCAL_O3_DIR:-${PROJECT_ROOT}/data/raw/o3}"
REMOTE_DEST="${REMOTE_DEST:-uib729081@glogin1.bsc.es:/home/uib/uib729081/O3-data}"

if [[ -f /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh ]]; then
    source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
    conda activate vit
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vit
else
    echo "Error: conda no encontrado. Activa el entorno vit antes de ejecutar este script." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python3 -u -m pipeline.download.download_o3 \
    --packs "${PACKS}" \
    --n-jobs 1 \
    --job-id 0

IFS="," read -r -a packs <<< "${PACKS}"
for pack in "${packs[@]}"; do
    pack_dir="${LOCAL_O3_DIR}/O3b-pack${pack}"
    if [[ ! -d "${pack_dir}" ]]; then
        echo "Error: no existe el directorio descargado ${pack_dir}" >&2
        exit 1
    fi
done

for pack in "${packs[@]}"; do
    pack_dir="${LOCAL_O3_DIR}/O3b-pack${pack}"
    echo "Subiendo ${pack_dir} -> ${REMOTE_DEST}/"
    scp -r "${pack_dir}" "${REMOTE_DEST}/"
done

echo "Packs O3 descargados y subidos a ${REMOTE_DEST}"
