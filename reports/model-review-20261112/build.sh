#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$task_dir/../.." && pwd)"
task_build="$project_root/tmp/pdfs/model-review-20261112"
task_output="$project_root/output/pdf"
mkdir -p "$task_build" "$task_output"
if command -v tectonic >/dev/null 2>&1; then
  tex_engine="$(command -v tectonic)"
elif [ -x /opt/homebrew/bin/tectonic ]; then
  tex_engine=/opt/homebrew/bin/tectonic
else
  echo 'Tectonic is required. See README.md for font and build requirements.' >&2
  exit 1
fi
cd "$task_dir"
"$tex_engine" --keep-logs --keep-intermediates --outdir "$task_build" report.tex
cp "$task_build/report.pdf" "$task_output/iea-model-review-20261112.pdf"
echo "$task_output/iea-model-review-20261112.pdf"
# Keep this task's log for visual QA. Cleanup is explicit after verification.
