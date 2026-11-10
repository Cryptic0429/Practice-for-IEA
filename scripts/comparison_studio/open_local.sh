#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
trial_dir="$PWD/experiments/comparison-studio-20261110"
mkdir -p "$trial_dir"
if curl --fail --silent --max-time 2 http://127.0.0.1:12348/health >/dev/null; then
  echo '对比台已可访问：http://127.0.0.1:12348/'
  exit 0
fi
ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:12348:127.0.0.1:12348 SJTU1
echo '本机入口：http://127.0.0.1:12348/'
