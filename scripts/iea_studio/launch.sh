#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
  PY="$PWD/.venv/bin/python"
elif [[ -x "$PWD/envs/iea/bin/python" ]]; then
  PY="$PWD/envs/iea/bin/python"
else
  PY="python3"
fi

exec "$PY" -u -m iea_studio.app "$@"
