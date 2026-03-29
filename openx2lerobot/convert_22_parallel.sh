#!/usr/bin/env bash
set -euo pipefail

# Batch convert 22 Open-X datasets with limited parallelism.
# - Default parallel workers: 2
# - One log file per dataset
# - Final summary with success/failure lists
# - Resume: set RESUME=1 to skip datasets that already have output (meta/info.json)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/fox/miniconda3/envs/convert/bin/python}"
CONVERTER="${CONVERTER:-${SCRIPT_DIR}/openx_rlds.py}"

RAW_ROOT="${RAW_ROOT:-/mnt/data/data/open/RT-X/tensorflow_datasets}"
OUT_ROOT="${OUT_ROOT:-/mnt/data/data/yyh/openx_lerobot_22}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
# Resume: skip datasets that already have output (OUT_ROOT/<dataset>_<version>_lerobot/meta/info.json)
RESUME="${RESUME:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/_batch_logs}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
MASTER_LOG="${RUN_LOG_DIR}/master.log"
STATUS_DIR="${RUN_LOG_DIR}/status"

mkdir -p "${RUN_LOG_DIR}" "${STATUS_DIR}" "${OUT_ROOT}"

DATASETS=(
  "kuka:0.1.0"
  "taco_play:0.1.0"
  "jaco_play:0.1.0"
  "berkeley_cable_routing:0.1.0"
  "roboturk:0.1.0"
  "viola:0.1.0"
  "berkeley_autolab_ur5:0.1.0"
  "toto:0.1.0"
  "language_table:0.1.0"
  "stanford_hydra_dataset_converted_externally_to_rlds:0.1.0"
  "austin_buds_dataset_converted_externally_to_rlds:0.1.0"
  "nyu_franka_play_dataset_converted_externally_to_rlds:0.1.0"
  "furniture_bench_dataset_converted_externally_to_rlds:0.1.0"
  "ucsd_kitchen_dataset_converted_externally_to_rlds:0.1.0"
  "austin_sailor_dataset_converted_externally_to_rlds:0.1.0"
  "austin_sirius_dataset_converted_externally_to_rlds:0.1.0"
  "dlr_edan_shared_control_converted_externally_to_rlds:0.1.0"
  "iamlab_cmu_pickup_insert_converted_externally_to_rlds:0.1.0"
  "utaustin_mutex:0.1.0"
  "berkeley_fanuc_manipulation:0.1.0"
  "cmu_stretch:0.1.0"
  "bc_z:0.1.0"
  "bridge_dataset:1.0.0"
)

log_master() {
  local msg="$1"
  local now
  now="$(date '+%F %T')"
  echo "[${now}] ${msg}" | tee -a "${MASTER_LOG}"
}

run_one() {
  local dataset="$1"
  local version="$2"
  local raw_dir="${RAW_ROOT}/${dataset}/${version}"
  local ds_log="${RUN_LOG_DIR}/${dataset}.log"
  local status_file="${STATUS_DIR}/${dataset}.status"

  {
    echo "==== START ${dataset} (${version}) @ $(date '+%F %T') ===="
    echo "RAW_DIR=${raw_dir}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "CONVERTER=${CONVERTER}"
    echo "PYTHON_BIN=${PYTHON_BIN}"

    if [[ ! -d "${raw_dir}" ]]; then
      echo "ERROR: dataset path not found: ${raw_dir}"
      echo "==== END ${dataset} (MISSING) @ $(date '+%F %T') ===="
      exit 2
    fi

    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${CONVERTER}" \
      --raw-dir "${raw_dir}" \
      --local-dir "${OUT_ROOT}" \
      --use-videos

    echo "==== END ${dataset} (OK) @ $(date '+%F %T') ===="
  } > "${ds_log}" 2>&1

  echo "$?" > "${status_file}"
}

cleanup_on_interrupt() {
  log_master "Interrupted. Stopping child jobs..."
  jobs -pr | xargs -r kill || true
  wait || true
  exit 130
}
trap cleanup_on_interrupt INT TERM

log_master "Batch run started."
log_master "MAX_PARALLEL=${MAX_PARALLEL}"
log_master "RESUME=${RESUME}"
log_master "RAW_ROOT=${RAW_ROOT}"
log_master "OUT_ROOT=${OUT_ROOT}"
log_master "RUN_LOG_DIR=${RUN_LOG_DIR}"

for item in "${DATASETS[@]}"; do
  dataset="${item%%:*}"
  version="${item##*:}"
  done_marker="${OUT_ROOT}/${dataset}_${version}_lerobot/meta/info.json"

  if [[ "${RESUME}" == "1" ]] && [[ -f "${done_marker}" ]]; then
    echo "==== RESUME: skip (already done) ${dataset}:${version} @ $(date '+%F %T') ====" > "${RUN_LOG_DIR}/${dataset}.log"
    echo "0" > "${STATUS_DIR}/${dataset}.status"
    log_master "Skip (done): ${dataset}:${version}"
    continue
  fi

  while true; do
    running_jobs=( $(jobs -pr) )
    (( ${#running_jobs[@]} < MAX_PARALLEL )) && break
    sleep 2
  done

  log_master "Launch: ${dataset}:${version}"
  run_one "${dataset}" "${version}" &
done

wait

success=0
failed=0
missing=0
SUCCESS_FILE="${RUN_LOG_DIR}/success.txt"
FAILED_FILE="${RUN_LOG_DIR}/failed.txt"
MISSING_FILE="${RUN_LOG_DIR}/missing.txt"
: > "${SUCCESS_FILE}"
: > "${FAILED_FILE}"
: > "${MISSING_FILE}"

for item in "${DATASETS[@]}"; do
  dataset="${item%%:*}"
  version="${item##*:}"
  status_file="${STATUS_DIR}/${dataset}.status"

  if [[ ! -f "${status_file}" ]]; then
    failed=$((failed + 1))
    echo "${dataset}:${version} (no status file)" >> "${FAILED_FILE}"
    continue
  fi

  code="$(cat "${status_file}")"
  if [[ "${code}" == "0" ]]; then
    success=$((success + 1))
    echo "${dataset}:${version}" >> "${SUCCESS_FILE}"
  elif [[ "${code}" == "2" ]]; then
    missing=$((missing + 1))
    echo "${dataset}:${version}" >> "${MISSING_FILE}"
  else
    failed=$((failed + 1))
    echo "${dataset}:${version} (exit=${code})" >> "${FAILED_FILE}"
  fi
done

log_master "Batch finished."
log_master "Success=${success}, Failed=${failed}, Missing=${missing}"
log_master "Success list: ${SUCCESS_FILE}"
log_master "Failed list:  ${FAILED_FILE}"
log_master "Missing list: ${MISSING_FILE}"
log_master "Per-dataset logs: ${RUN_LOG_DIR}/*.log"

echo
echo "Run complete."
echo "Master log: ${MASTER_LOG}"
echo "Success:    ${SUCCESS_FILE}"
echo "Failed:     ${FAILED_FILE}"
echo "Missing:    ${MISSING_FILE}"
echo "Logs dir:   ${RUN_LOG_DIR}"
