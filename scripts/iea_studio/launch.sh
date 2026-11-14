#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/scripts"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec envs/iea/bin/python -u -m iea_studio.app "$@"
