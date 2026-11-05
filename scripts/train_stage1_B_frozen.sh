#!/usr/bin/env bash
set -euo pipefail
export IEA_GPU="${IEA_GPU:-2}"
source /mnt/sdc/zhc/IEA/activate.sh
python "$IEA_ROOT/scripts/verify_stage1_B.py"
python "$IEA_ROOT/scripts/verify_frozen_training_config.py" --require-free-gpu
exec "$IEA_ROOT/scripts/train.sh" "$IEA_ROOT/configs/qlora-stage1-B-3k.yaml"
