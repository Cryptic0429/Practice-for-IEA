#!/usr/bin/env bash
# Run from an IEA checkout with the two existing server environments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
trial=experiments/functional-final-delivery-20261107/trial
mkdir -p "$trial/web-deps"
envs/iea/bin/python - "$trial/web-deps" <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import sys

destination = Path(sys.argv[1])
for package in ("Flask", "Werkzeug", "Jinja2", "itsdangerous", "MarkupSafe", "blinker", "click"):
    installed = distribution(package)
    metadata = next(str(path).split("/")[0] for path in installed.files if ".dist-info/" in str(path))
    for child in (package.lower(), metadata):
        source = Path(installed.locate_file(child)).resolve(strict=True)
        target = destination / child
        if target.is_symlink() and target.resolve() == source:
            continue
        target.symlink_to(source, target_is_directory=True)
PY
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/scripts:$PWD/$trial/web-deps"
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
exec envs/iea-rl/bin/python -u -m functional_delivery.trial "$@"
