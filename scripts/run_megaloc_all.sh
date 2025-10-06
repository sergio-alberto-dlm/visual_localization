#!/usr/bin/env bash
# Run MegaLoc embedding extraction per *subsession* for each device type present.
# Device types supported by your dataset: ios, hl (HoloLens), spot.
#
# Examples:
#   ./scripts/run_megaloc_all.sh --sessions_root /DATA/capture/HYDRO --out_dir /DATA/embeddings/megaloc
#   ./scripts/run_megaloc_all.sh --sessions_root /DATA/capture/HYDRO --out_dir out --batch_size 256 --compute_device cuda:1 --dtype float16 --normalize --skip-existing
#
set -euo pipefail

# ---------- defaults ----------
SESSIONS_ROOT=""
OUT_DIR=""
BATCH_SIZE=128
COMPUTE_DEVICE="cuda:0"
DTYPE="float32"             # float32 | float16
NORMALIZE=true             # L2-normalize embeddings
SKIP_EXISTING=false         # skip subsessions that already have outputs
PYTHON_BIN="${PYTHON_BIN:-python}"  # allow override via env var

usage() {
  cat <<EOF
Usage: $(basename "$0") --sessions_root <path> --out_dir <path> [options]

Options:
  --sessions_root PATH     Root containing <device>_map/ (ios_map, hl_map, spot_map)  [required]
  --out_dir PATH           Output directory root                                     [required]
  --batch_size INT         Batch size per forward (default: ${BATCH_SIZE})
  --compute_device STR     e.g., cuda:0 or cpu (default: ${COMPUTE_DEVICE})
  --dtype {float32,float16}  On-disk dtype (default: ${DTYPE})
  --normalize              L2-normalize embeddings
  --no-normalize           Disable normalization (default)
  --skip-existing          Skip subsessions that already exist in out_dir
  --python PATH            Python executable to use (default: env \$PYTHON_BIN or 'python')
  -h|--help                Show this help

Notes:
  • Device types auto-detected: ios_map/, hl_map/, spot_map/. Missing ones are skipped.
  • HoloLens device key is "hl" (not "hololens") to match your dataset code.
EOF
}

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sessions_root) SESSIONS_ROOT="$2"; shift 2 ;;
    --out_dir)       OUT_DIR="$2"; shift 2 ;;
    --batch_size)    BATCH_SIZE="$2"; shift 2 ;;
    --compute_device)COMPUTE_DEVICE="$2"; shift 2 ;;
    --dtype)         DTYPE="$2"; shift 2 ;;
    --normalize)     NORMALIZE=true; shift ;;
    --no-normalize)  NORMALIZE=false; shift ;;
    --skip-existing) SKIP_EXISTING=true; shift ;;
    --python)        PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -z "${SESSIONS_ROOT}" ]] && { echo "[ERR] --sessions_root is required"; usage; exit 1; }
[[ -z "${OUT_DIR}"       ]] && { echo "[ERR] --out_dir is required";       usage; exit 1; }

SESSIONS_ROOT="$(realpath "${SESSIONS_ROOT}")"
# OUT_DIR="$(realpath "${OUT_DIR}")"

echo "[INFO] sessions_root: ${SESSIONS_ROOT}"
echo "[INFO] out_dir      : ${OUT_DIR}"
echo "[INFO] batch_size   : ${BATCH_SIZE}"
echo "[INFO] compute_device: ${COMPUTE_DEVICE}"
echo "[INFO] dtype        : ${DTYPE}"
echo "[INFO] normalize    : ${NORMALIZE}"
echo "[INFO] skip_existing: ${SKIP_EXISTING}"
echo "[INFO] python       : ${PYTHON_BIN}"

# ---------- helper to run one device ----------
run_device() {
  local DEV="$1"     # ios | hl | spot
  local MAP_DIR="${SESSIONS_ROOT}/${DEV}_map"

  if [[ ! -d "${MAP_DIR}" ]]; then
    echo "[SKIP] ${DEV}: ${MAP_DIR} not found."
    return 0
  fi

  echo "[RUN ] ${DEV}: extracting per-subsesson embeddings…"
  set -x
  "${PYTHON_BIN}" tasks/glob_feats.py \
    --sessions_root "${SESSIONS_ROOT}" \
    --out_dir "${OUT_DIR}/${DEV}" \
    --device "${DEV}" \
    --batch_size "${BATCH_SIZE}" \
    --compute_device "${COMPUTE_DEVICE}" \
    --dtype "${DTYPE}" \
    $([ "${NORMALIZE}" = true ] && echo --normalize) \
    $([ "${SKIP_EXISTING}" = true ] && echo --skip_existing)
  set +x
  echo "[OK  ] ${DEV}: done."
}

# ---------- run for each device present ----------
run_device "spot"
run_device "hl"      # HoloLens
run_device "ios"

echo "[DONE] All available device types processed."