#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$task_dir/../.." && pwd)"
build_dir="$project_root/tmp/pdfs/mainline-technical-20261113"
mkdir -p "$build_dir" "$project_root/output/pdf"
tex_engine="$(command -v tectonic || true)"
if [ -z "$tex_engine" ] && [ -x /opt/homebrew/bin/tectonic ]; then tex_engine=/opt/homebrew/bin/tectonic; fi
if [ -z "$tex_engine" ]; then echo 'Install Tectonic; see report README.' >&2; exit 1; fi
cd "$task_dir"
"$tex_engine" --keep-logs --keep-intermediates --outdir "$build_dir" report.tex
cp "$build_dir/report.pdf" "$project_root/output/pdf/iea-mainline-technical-20261113.pdf"
echo "$project_root/output/pdf/iea-mainline-technical-20261113.pdf"
