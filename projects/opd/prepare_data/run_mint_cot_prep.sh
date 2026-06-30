#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/projects/opd/data_raw}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/projects/opd/data_parquet}"
RAW_MINT_FILE="${RAW_MINT_FILE:-${RAW_ROOT}/mint_cot_dataset/rl/MINT-CoT_interleave_rl_54k_filtered.parquet}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/mint_cot_dataset}"

mkdir -p "$(dirname "${RAW_MINT_FILE}")" "${OUT_DIR}"

if [[ ! -f "${RAW_MINT_FILE}" && "${SKIP_DOWNLOAD:-False}" != "True" ]]; then
    URL="${MINT_COT_URL:-https://huggingface.co/datasets/xy06/MINT-CoT-Dataset/resolve/main/rl/MINT-CoT_interleave_rl_54k_filtered.parquet?download=true}"
    echo "Downloading MINT-CoT raw parquet:"
    echo "  ${URL}"
    echo "  -> ${RAW_MINT_FILE}"
    curl -L --fail --retry 5 --retry-delay 2 --retry-all-errors "${URL}" -o "${RAW_MINT_FILE}.part"
    mv "${RAW_MINT_FILE}.part" "${RAW_MINT_FILE}"
fi

if [[ -f "${RAW_MINT_FILE}" ]]; then
    LOCAL_DATASET_PATH="${RAW_MINT_FILE}"
else
    echo "Raw MINT-CoT parquet not found; falling back to datasets.load_dataset('xy06/MINT-CoT-Dataset')." >&2
    LOCAL_DATASET_PATH=""
fi

if [[ -n "${LOCAL_DATASET_PATH}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/projects/opd/prepare_data/mint_cot_dataset.py" \
        --local_dataset_path "${LOCAL_DATASET_PATH}" \
        --local_save_dir "${OUT_DIR}"
else
    "${PYTHON_BIN}" "${REPO_ROOT}/projects/opd/prepare_data/mint_cot_dataset.py" \
        --local_save_dir "${OUT_DIR}"
fi

echo "Prepared MINT-CoT OPD data under ${OUT_DIR}"
