#!/usr/bin/env bash
# Run all 3x3 cross-device retrievals.
# Example:
#   ./scripts/run_retieval_all.sh \
#       --sessions_root /.../HYDRO/sessions \
#       --embeddings_root /.../results/HYDRO/embeddings/megaloc \
#       --out_dir /.../results/HYDRO/retrievals \
#       --script tasks/retrieval.py \
#       --topk 10 --normalize --use_faiss --faiss_gpu

set -euo pipefail

# -------------------- defaults --------------------
SESSIONS_ROOT=""
EMBEDDINGS_ROOT=""
OUT_DIR=""
SCRIPT="tasks/retrieval.py"     
TOPK=10
COMPUTE_DEVICE="cuda:0"
NORMALIZE=true
USE_FAISS=true
FAISS_GPU=false
PYTHON_BIN="${PYTHON_BIN:-python}"
PARALLEL=false   # set true to run all 9 in parallel

usage() {
  cat <<EOF
Usage: $(basename "$0") --sessions_root <path> --embeddings_root <path> --out_dir <path> [options]

Required:
  --sessions_root PATH     Path to scene sessions folder (contains ios_query/, hl_map/, etc.)
  --embeddings_root PATH   Root folder that contains subfolders {ios,hl,spot} with per-subsession map embeddings
  --out_dir PATH           Where to write output JSON files

Options:
  --script PATH            Retrieval Python script (default: ${SCRIPT})
  --topk INT               Top-K to retrieve (default: ${TOPK})
  --compute_device STR     e.g. cuda:0 or cpu (default: ${COMPUTE_DEVICE})
  --normalize              L2-normalize features for cosine/IP
  --use_faiss              Use FAISS for search
  --faiss_gpu              Put FAISS index on GPU 0
  --parallel               Run the 9 jobs in parallel
  --python PATH            Python binary (default: env \$PYTHON_BIN or 'python')
  -h | --help              Show this help

Assumptions:
  - EMBEDDINGS_ROOT layout:
        <EMBEDDINGS_ROOT>/
          ios/<subsession>/{<subsession>.npy, manifest.csv}
          hl/<subsession>/{...}
          spot/<subsession>/{...}
  - Output file name is: <OUT_DIR>/<scene_name>_map_<MAP>_query_<Q>.json
EOF
}

# -------------------- parse args --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sessions_root)   SESSIONS_ROOT="$2"; shift 2 ;;
    --embeddings_root) EMBEDDINGS_ROOT="$2"; shift 2 ;;
    --out_dir)         OUT_DIR="$2"; shift 2 ;;
    --script)          SCRIPT="$2"; shift 2 ;;
    --topk)            TOPK="$2"; shift 2 ;;
    --compute_device)  COMPUTE_DEVICE="$2"; shift 2 ;;
    --normalize)       NORMALIZE=true; shift ;;
    --use_faiss)       USE_FAISS=true; shift ;;
    --faiss_gpu)       FAISS_GPU=true; shift ;;
    --parallel)        PARALLEL=true; shift ;;
    --python)          PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -z "${SESSIONS_ROOT}" ]]   && { echo "[ERR] --sessions_root is required"; usage; exit 1; }
[[ -z "${EMBEDDINGS_ROOT}" ]] && { echo "[ERR] --embeddings_root is required"; usage; exit 1; }
[[ -z "${OUT_DIR}" ]]         && { echo "[ERR] --out_dir is required"; usage; exit 1; }

SESSIONS_ROOT="$(realpath "${SESSIONS_ROOT}")"
EMBEDDINGS_ROOT="$(realpath "${EMBEDDINGS_ROOT}")"
OUT_DIR="$(realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

SCENE_NAME="$(basename "${SESSIONS_ROOT}")"
DEVICES=(ios hl spot)

echo "[INFO] sessions_root   : ${SESSIONS_ROOT}"
echo "[INFO] embeddings_root : ${EMBEDDINGS_ROOT}"
echo "[INFO] out_dir         : ${OUT_DIR}"
echo "[INFO] script          : ${SCRIPT}"
echo "[INFO] topk            : ${TOPK}"
echo "[INFO] compute_device  : ${COMPUTE_DEVICE}"
echo "[INFO] normalize       : ${NORMALIZE}"
echo "[INFO] use_faiss       : ${USE_FAISS}"
echo "[INFO] faiss_gpu       : ${FAISS_GPU}"
echo "[INFO] parallel        : ${PARALLEL}"
echo "[INFO] python          : ${PYTHON_BIN}"
echo

run_pair () {
  local MAP_DEV="$1"   # ios|hl|spot
  local QRY_DEV="$2"   # ios|hl|spot

  local MAP_FEATS_ROOT="${EMBEDDINGS_ROOT}/${MAP_DEV}"
  if [[ ! -d "${MAP_FEATS_ROOT}" ]]; then
    echo "[SKIP] map_device=${MAP_DEV}: embeddings dir missing: ${MAP_FEATS_ROOT}"
    return 0
  fi

  local OUT_JSON="${OUT_DIR}/${SCENE_NAME}_map_${MAP_DEV}_query_${QRY_DEV}.json"

  echo "[RUN ] map=${MAP_DEV}  query=${QRY_DEV}"
  set -x
  "${PYTHON_BIN}" "${SCRIPT}" \
    --sessions_root "${SESSIONS_ROOT}" \
    --map_feats_root "${MAP_FEATS_ROOT}" \
    --map_device "${MAP_DEV}" \
    --query_device "${QRY_DEV}" \
    --topk "${TOPK}" \
    --compute_device "${COMPUTE_DEVICE}" \
    $([ "${NORMALIZE}" = true ] && echo --normalize) \
    $([ "${USE_FAISS}" = true ] && echo --use_faiss) \
    $([ "${FAISS_GPU}" = true ] && echo --faiss_gpu) \
    # --out_json "${OUT_JSON}"
  local RC=$?
  set +x
  if [[ $RC -ne 0 ]]; then
    echo "[FAIL] map=${MAP_DEV} query=${QRY_DEV} (exit ${RC})"
  else
    echo "[OK  ] ${OUT_JSON}"
  fi
}

# -------------------- run all 9 combos --------------------
PIDS=()
for MAP_DEV in "${DEVICES[@]}"; do
  for QRY_DEV in "${DEVICES[@]}"; do
    if [[ "${PARALLEL}" = true ]]; then
      run_pair "${MAP_DEV}" "${QRY_DEV}" &
      PIDS+=($!)
    else
      run_pair "${MAP_DEV}" "${QRY_DEV}"
    fi
  done
done

if [[ "${PARALLEL}" = true ]]; then
  echo "[INFO] Waiting for ${#PIDS[@]} jobs…"
  wait
fi

echo "[DONE] All retrieval configurations attempted."
