#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 ]]; then
  exec /mnt/sdc/zhc/IEA/scripts/train_stage1_frozen.sh
fi
export IEA_GPU="${IEA_GPU:-2}"
source /mnt/sdc/zhc/IEA/activate.sh
cd "$IEA_ROOT/repo/LLaMA-Factory"
exec llamafactory-cli train "${1:-$IEA_ROOT/configs/qlora-stage1.example.yaml}" "${@:2}"
