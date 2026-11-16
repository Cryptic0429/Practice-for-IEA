# Contributing to IEA

Thank you for your interest in contributing to **Interactive Image Editing Agent (IEA)**.

## Development Setup

```bash
git clone https://github.com/Cryptic0429/Practice-for-IEA.git
cd Practice-for-IEA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running Verification and Tests

Before opening a pull request, please ensure all repository checks pass:

```bash
make verify        # Checks model layout, frozen protocol hashes, and evidence integrity
make test          # Runs pure-stdlib project layout and agent orchestration unit tests
make test-runtime  # Runs CPU deterministic editor invariants and Studio API contract tests
```

## Repository Boundaries

- **Active agent & UI layers (`iea/`, `scripts/iea_agent/`, `scripts/iea_studio/`)**: Primary target for new features, bug fixes, and UI/UX improvements.
- **Frozen protocol modules (`scripts/functional_v*/`) and archived evidence (`experiments/`, `data/training/`)**: These files are SHA256-verified against published training protocols and evaluation records (`make verify`). Do not modify or reformat them unless explicitly intending to create a new versioned experiment protocol.
- **Do not commit large binaries or private data**: `.gitignore` excludes model weights (`.safetensors`, `.pt`, `.bin`), raw image datasets, runtime job/session directories, and compiled PDFs.

## Pull Request Checklist

1. Keep functions modular, well-typed, and covered by unit tests.
2. Run `make verify`, `make test`, and `make test-runtime` locally.
3. Update relevant documentation in `README.md` or `docs/` when adding user-facing behavior or CLI options.
4. Keep evaluation claims grounded in verified logs and metrics—clearly mark experimental or non-production limitations.
