"""Command-line entrypoints for IEA Studio and integrity verification."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / 'scripts'
if _SCRIPTS.is_dir() and str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def studio(argv: list[str] | None = None) -> None:
    """Launch the IEA Studio web application."""
    from iea_studio import app as studio_app

    if argv is not None:
        sys.argv = ['iea-studio', *argv]
    studio_app.main()


def verify(argv: list[str] | None = None) -> int:
    """Run repository integrity and frozen-protocol verification."""
    from project import assets

    parser = argparse.ArgumentParser(prog='iea-verify', description='Verify IEA model registry, frozen code, and evidence hashes.')
    parser.add_argument('--require-weights', action='store_true', help='Fail if local binary weights (.safetensors/.pt) are absent.')
    args = parser.parse_args(argv)
    ok = assets.verify(require_weights=args.require_weights)
    raise SystemExit(0 if ok else 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='python -m iea', description='IEA command-line interface.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('studio', help='Launch IEA Studio (passes remaining args through).')
    sub.add_parser('verify', help='Verify model manifest, frozen protocols, and retained evidence.')
    ns, rest = parser.parse_known_args(argv)
    if ns.command == 'studio':
        studio(rest)
    elif ns.command == 'verify':
        verify(rest)


if __name__ == '__main__':
    main()
