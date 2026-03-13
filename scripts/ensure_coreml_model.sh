#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH=""
LAYER_COUNT=1
WEIGHT_MODE="random"
COREMLTOOLS_VENV="${COREMLTOOLS_VENV:-/tmp/coremltools312-install-venv}"

usage() {
  cat <<'EOF'
Usage: scripts/ensure_coreml_model.sh --output PATH [--layers N] [--weight-mode random|zero]

Ensures a Python environment with coremltools is available, then generates a benchmark-ready
Core ML `.mlpackage` using `scripts/generate_coreml_model.py`.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      shift
      [[ $# -gt 0 ]] || { echo "--output requires a path" >&2; exit 1; }
      OUTPUT_PATH="$1"
      ;;
    --layers)
      shift
      [[ $# -gt 0 ]] || { echo "--layers requires an integer" >&2; exit 1; }
      LAYER_COUNT="$1"
      ;;
    --weight-mode)
      shift
      [[ $# -gt 0 ]] || { echo "--weight-mode requires random|zero" >&2; exit 1; }
      WEIGHT_MODE="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

[[ -n "$OUTPUT_PATH" ]] || { echo "--output is required" >&2; exit 1; }
[[ "$LAYER_COUNT" =~ ^[0-9]+$ ]] && [[ "$LAYER_COUNT" -gt 0 ]] || {
  echo "--layers must be a positive integer" >&2
  exit 1
}
[[ "$WEIGHT_MODE" == "random" || "$WEIGHT_MODE" == "zero" ]] || {
  echo "--weight-mode must be random or zero" >&2
  exit 1
}
[[ "$OUTPUT_PATH" == *.mlpackage ]] || {
  echo "--output must end with .mlpackage" >&2
  exit 1
}

discover_python() {
  local candidates=()
  if [[ -n "${COREMLTOOLS_PYTHON:-}" ]]; then
    candidates+=("${COREMLTOOLS_PYTHON}")
  fi
  candidates+=(
    "${COREMLTOOLS_VENV}/bin/python"
    "/opt/homebrew/opt/python@3.12/bin/python3.12"
    "/usr/local/opt/python@3.12/bin/python3.12"
  )

  local found
  if found="$(command -v python3.12 2>/dev/null)"; then
    candidates+=("${found}")
  fi
  if found="$(command -v python3 2>/dev/null)"; then
    candidates+=("${found}")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c "import coremltools" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  echo ""
}

bootstrap_python() {
  local bootstrap_python=""

  if [[ -n "${COREMLTOOLS_PYTHON:-}" && -x "${COREMLTOOLS_PYTHON}" ]]; then
    bootstrap_python="${COREMLTOOLS_PYTHON}"
  elif [[ -x "/opt/homebrew/opt/python@3.12/bin/python3.12" ]]; then
    bootstrap_python="/opt/homebrew/opt/python@3.12/bin/python3.12"
  elif [[ -x "/usr/local/opt/python@3.12/bin/python3.12" ]]; then
    bootstrap_python="/usr/local/opt/python@3.12/bin/python3.12"
  elif command -v python3.12 >/dev/null 2>&1; then
    bootstrap_python="$(command -v python3.12)"
  elif command -v python3 >/dev/null 2>&1; then
    bootstrap_python="$(command -v python3)"
  fi

  [[ -n "$bootstrap_python" ]] || {
    echo "Unable to find a Python interpreter to bootstrap coremltools." >&2
    exit 1
  }

  echo "Bootstrapping coremltools virtualenv at ${COREMLTOOLS_VENV}"
  "$bootstrap_python" -m venv "${COREMLTOOLS_VENV}"
  "${COREMLTOOLS_VENV}/bin/pip" install --upgrade pip
  "${COREMLTOOLS_VENV}/bin/pip" install coremltools
  echo "${COREMLTOOLS_VENV}/bin/python"
}

PYTHON_BIN="$(discover_python)"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(bootstrap_python)"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Generating Core ML baseline model"
echo "  python=${PYTHON_BIN}"
echo "  layers=${LAYER_COUNT}"
echo "  weight_mode=${WEIGHT_MODE}"
echo "  output=${OUTPUT_PATH}"

"${PYTHON_BIN}" "${ROOT}/scripts/generate_coreml_model.py" \
  --layers "${LAYER_COUNT}" \
  --weight-mode "${WEIGHT_MODE}" \
  --output "${OUTPUT_PATH}"

echo "Core ML baseline ready at ${OUTPUT_PATH}"
