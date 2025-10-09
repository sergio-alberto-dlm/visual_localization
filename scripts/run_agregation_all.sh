#!/usr/bin/env bash
# Run robust pose aggregation for a SINGLE scene over all JSONs in a folder.

set -euo pipefail

JSON_DIR=""
SCENE=""
OUT_DIR=""
SCRIPT="tasks/poses_agregation.py"  # path to your Python file
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --json_dir DIR --scene NAME --out_dir OUTDIR [--script FILE] [--python BIN]

Required:
  --json_dir DIR   Directory containing retrieval JSON files for the scene
  --scene NAME     Scene name to pass to the script (e.g., HYDRO or SUCCULENT)
  --out_dir OUTDIR Directory where .txt results will be written

Optional:
  --script FILE    Python script path (default: ${SCRIPT})
  --python BIN     Python binary (default: env \$PYTHON_BIN or "python")

Example:
  $(basename "$0") \
      --json_dir /path/to/HYDRO/retrievals \
      --scene HYDRO \
      --out_dir /path/to/HYDRO/poses \
      --script tasks/poses_agregation.py
EOF
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json_dir) JSON_DIR="$2"; shift 2 ;;
    --scene)    SCENE="$2"; shift 2 ;;
    --out_dir)  OUT_DIR="$2"; shift 2 ;;
    --script)   SCRIPT="$2"; shift 2 ;;
    --python)   PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "[ERR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -z "${JSON_DIR}" ]] && { echo "[ERR] --json_dir is required"; usage; exit 1; }
[[ -z "${SCENE}"    ]] && { echo "[ERR] --scene is required"; usage; exit 1; }
[[ -z "${OUT_DIR}"  ]] && { echo "[ERR] --out_dir is required"; usage; exit 1; }

JSON_DIR="$(realpath "${JSON_DIR}")"
SCRIPT="$(realpath "${SCRIPT}")"
OUT_DIR="$(realpath "${OUT_DIR}")"

[[ -d "${JSON_DIR}" ]] || { echo "[ERR] Not a directory: ${JSON_DIR}"; exit 1; }
[[ -f "${SCRIPT}"   ]] || { echo "[ERR] Script not found: ${SCRIPT}"; exit 1; }

mkdir -p "${OUT_DIR}"

echo "[INFO] json_dir: ${JSON_DIR}"
echo "[INFO] scene   : ${SCENE}"
echo "[INFO] out_dir : ${OUT_DIR}"
echo "[INFO] script  : ${SCRIPT}"
echo "[INFO] python  : ${PYTHON_BIN}"
echo

shopt -s nullglob

# Iterate safely even if filenames contain spaces/newlines
while IFS= read -r -d '' jf; do
  base="$(basename "$jf")"
  echo "[RUN ] ${base}"
  set -x
  "${PYTHON_BIN}" "${SCRIPT}" \
      --raw_poses_root "$jf" \
      --scene "${SCENE}" \
      --out_dir "${OUT_DIR}"
  rc=$?
  set +x
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${base} (exit ${rc})"
  else
    echo "[OK  ] ${base}"
  fi
  echo
done < <(find "${JSON_DIR}" -type f -name "*.json" -print0)

echo "[DONE] Processed JSONs in ${JSON_DIR} for scene ${SCENE}"
echo "[RESULTS] written to ${OUT_DIR}"
