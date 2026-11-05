#!/usr/bin/env bash
set -euo pipefail
export IEA_GPU="${IEA_GPU:-2}"
source /mnt/sdc/zhc/IEA/activate.sh
if [[ "${1:-}" == "--check-only" && $# -eq 1 ]]; then
  exec python "$IEA_ROOT/scripts/verify_frozen_training_config.py"
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: IEA_GPU=2 train_stage1_frozen.sh [--check-only]' >&2
  exit 2
fi
python "$IEA_ROOT/scripts/verify_frozen_training_config.py" --require-free-gpu
exec "$IEA_ROOT/scripts/train.sh" "$IEA_ROOT/configs/qlora-stage1-A-3k.yaml"
