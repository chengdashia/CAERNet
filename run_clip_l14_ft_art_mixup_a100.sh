#!/usr/bin/env bash
set -euo pipefail

CAERNET_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${CAERNET_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HARDWARE="${HARDWARE:-a100}"
LOG_DIR="classify/outputs/logs/A100"

CONFIGS=(
  "${CAERNET_DIR}/configs/experiments/clip_adapter_vit_l14_ft_art_mixup_no_energy.yaml"
  "${CAERNET_DIR}/configs/experiments/clip_adapter_vit_l14_ft_art_mixup_no_contrastive.yaml"
  "${CAERNET_DIR}/configs/experiments/clip_adapter_vit_l14_ft_art_mixup.yaml"
)

mkdir -p "${LOG_DIR}"

for data_dir in \
  "classify/data/artbench10/train" \
  "classify/data/artbench10_paper/val" \
  "classify/data/artbench10/test"
do
  if [[ ! -d "${data_dir}" ]]; then
    echo "ERROR: dataset path not found: ${data_dir}" >&2
    echo "Run from CAERNet once: python prepare_data.py" >&2
    exit 1
  fi
done

for config in "${CONFIGS[@]}"; do
  run_name="$(basename "${config}" .yaml)"
  log_file="${LOG_DIR}/${run_name}_$(date +%Y%m%d_%H%M%S).log"

  echo
  echo "============================================================"
  echo "Running: ${config}"
  echo "Log: ${log_file}"
  echo "============================================================"

  PYTHONPATH="${CAERNET_DIR}" "${PYTHON_BIN}" "${CAERNET_DIR}/run_training.py" \
    --config "${config}" \
    --hardware "${HARDWARE}" \
    "$@" 2>&1 | tee "${log_file}"
done