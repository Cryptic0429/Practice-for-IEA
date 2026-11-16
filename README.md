# IEA — Interactive Image Editing Agent

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--VL--7B--Instruct-6E40C9.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)](docs/EVALUATION.md)

**IEA (Interactive Image Editing Agent)** is a multi-turn conversational global photo-editing system built on [`Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). It couples a dual-mode Vision-Language Model (zero-VRAM adapter toggling on a single 7B base), a lightweight magnitude & risk controller head, a deterministic 9-slider non-destructive rendering engine, and a two-stage self-check loop (CPU hard-metric guardrails + VLM visual Critic) inside an interactive darkroom Web Studio.

> [!NOTE]
> **Research Prototype Status (`production_qualified: false`)**: While the primary controller configuration (`seed 20261205 / e20`) improves controlled development pass rate from `75/100` to `82/100` (with 0 major artifacts vs. 1), cross-seed replication (`seed 20261206`) failed to reliably prevent night-scene noise amplification. See [Evaluation & Limitations](docs/EVALUATION.md) for full empirical findings.

---

## Highlights

- **Single-Model Dual-Mode Inference (`DualModeVLMBackend`)**: Uses PEFT `disable_adapter()` for native VLM conversational planning and visual critique, and enables the trained LoRA + controller head for structured parameter prediction—with zero extra base-model VRAM.
- **9-Dimensional Deterministic Non-Destructive Renderer**: Every turn re-renders directly from the immutable original RGB image (`source_rgb`) using cumulative parameters (`exposure`, `brightness`, `contrast`, `highlights`, `shadows`, `saturation`, `temperature`, `tint`, `sepia`), preventing cumulative compression or quantization drift across multi-turn edits.
- **Magnitude & Quality-Risk Controller Head (`functional_v21`)**: A 463,754-parameter MLP over frozen 3584-D hidden states, 16 image statistics, and 12 local luminance/chroma texture residuals that calibrates adjustment strength and flags high-risk inputs (`unreliable_quality`).
- **Two-Stage Self-Check Loop**: Post-render CPU hard-metric guardrails (`evaluate_cpu_guardrail`) check highlight clipping, shadow crushing, and contrast collapse in milliseconds, followed by a VLM Critic check. Failures attenuate deltas (`0.6×`, up to 2 retries) or roll back cleanly.
- **Darkroom Web Studio (`iea_studio`)**: Multi-turn chat workspace with before/after split slider, lossless turn timeline checkout/undo/reset, parameter telemetry, sample gallery, and a side-by-side bilingual baseline comparison view. Runs out of the box on CPU in deterministic demo mode (`cpu_rule_fallback`).

---

## Architecture

```text
User Photo + Multi-Turn Natural Language
   │
   ▼
[1. Conversational Planner]  (Native Qwen2.5-VL-7B, adapter disabled)
   │  • Tracks multi-turn context ("warmer", "halve that", "undo", "reset")
   │  • Rejects out-of-scope local edits (object removal, inpainting, denoise)
   │  • Rewrites fuzzy requests into standalone single-turn instructions
   ▼
[2. Parameter & Risk Decision]  (Semantic Backbone LoRA + Controller Head)
   │  • Semantic backbone selects active tools & adjustment directions
   │  • Controller head calibrates 9-D magnitudes & predicts quality risk
   ▼
[3. Non-Destructive Renderer]  (functional_v15 bounded-fixed editor)
   │  • Always renders from immutable source_rgb + cumulative parameters
   ▼
[4. Two-Stage Self-Check Loop]  (CPU Guardrails + Visual VLM Critic)
      • Checks white/black clipping, contrast collapse, & visual artifacts
      • Attenuates delta by 0.6× (up to 2 retries) or rolls back on persistent risk
```

For mathematical details, tensor contracts, and module interfaces, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Model Registry & Benchmark Summary

All model paths, SHA256 digests, and semantic aliases are defined in [`models/mainline.json`](models/mainline.json).

| Tier | Component | Directory Alias | Summary & Key Observations |
|---|---|---|---|
| **Core Backbone** | Semantic Decision Backbone (`r15`) | `models/core/semantic_backbone` | LoRA adapter (`seed 20261127`, epoch 1) + shared-luminance shadow-gain editor |
| **Core Controller** | Magnitude & Risk Head (`r21_head`) | `models/core/controller_head` | 463K-param MLP (`seed 20261205`, epoch 20); primary seed `82/100` controlled dev (`16/20` raw dev, `5/6` legacy regression, 0 major artifacts), **cross-seed replication failed** |
| **Ablation Baseline** | No-Controller Backbone (`r15`) | `models/baselines/ablation_no_controller` | Semantic backbone without controller head (`75/100` controlled dev, 1 major saturation artifact) |
| **Functional Baseline** | Early Functional Baseline (`r1`) | `models/baselines/early_functional_baseline` | First-generation bilingual functional LoRA (`31/50` frozen acceptance vs. `15/50` legacy baseline) |
| **SFT Baseline** | Calibrated SFT (`calibrated_sft`) | `models/baselines/early_calibrated_sft` | 3k parameter-calibrated SFT baseline (`-9.46%` L1 vs. Stage A on 71 images) |

Detailed ablation tables, dataset provenance, and failure-mode analyses are documented in [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Cryptic0429/Practice-for-IEA.git
cd Practice-for-IEA

python3 -m venv .venv
source .venv/bin/activate

# CPU / Studio Web UI & deterministic editor
pip install -r requirements.txt
pip install -e . --no-deps
```

For GPU dual-mode VLM inference (`Qwen2.5-VL-7B-Instruct` in 4-bit NF4 / BF16), also install GPU dependencies:

```bash
pip install -e ".[gpu]"
```

### 2. Launch IEA Studio (CPU Demo Mode)

Without a GPU or the 7B base weights, IEA Studio launches immediately using the deterministic CPU planner fallback (`cpu_rule_fallback`), the real `functional_v15` non-destructive editor, and live CPU guardrails:

```bash
iea-studio --port 12348
# or:
bash scripts/iea_studio/launch.sh --port 12348
```

Then open [http://127.0.0.1:12348](http://127.0.0.1:12348) in your browser.

### 3. Launch IEA Studio with Full 7B VLM Dual-Mode (GPU)

After placing `Qwen/Qwen2.5-VL-7B-Instruct` and the LoRA/controller weights under `models/` (see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)):

```bash
iea-studio --device cuda --gpu 0 --gpu-budget-hours 1.0 --enable-agent-vlm --port 12348
```

### 4. Python SDK Usage

```python
from pathlib import Path
from iea import ImageEditingAgent, RuleBasedCPUPlannerBackend, SessionState

agent = ImageEditingAgent(backend=RuleBasedCPUPlannerBackend())
session = SessionState(
    session_id="demo-01",
    title="sunset.jpg",
    source_image="source.png",
    source_sha256="...",
    image_size=[1280, 853],
)
result = agent.handle_turn(session, Path("./session_dir"), "Lift shadows slightly and warm up the tone")
print(result["status"], session.current_params)
```

### 5. Integrity Verification & Tests

```bash
make verify        # Verifies model layout, 191 frozen protocol code hashes, and 669 evidence hashes
make test          # Runs pure-stdlib project layout & multi-turn agent orchestration tests
make test-runtime  # Runs CPU editor invariant checks & Studio Flask API contract tests
```

---

## Repository Structure

```text
Practice-for-IEA/
├── iea/                      # Public Python package facade & CLI (`iea-studio`, `iea-verify`)
├── scripts/
│   ├── iea_agent/            # Multi-turn agent orchestrator, dual-mode backend, guardrails, state stack
│   ├── iea_studio/           # Darkroom Web Studio (Flask backend + HTML/CSS/JS frontend)
│   ├── project/              # Repository layout & SHA256 integrity verification (`assets.py`)
│   ├── functional_v1..v21/   # Frozen protocol modules, deterministic editors, & controller head
│   └── goal_v2_*, stage2_*   # Historical training & evaluation scripts
├── models/
│   ├── core/                 # Symlinks to core semantic backbone & controller head
│   ├── baselines/            # Symlinks to ablation & historical baselines
│   ├── from-*/               # Adapter configs & metadata (binary weights excluded from Git)
│   └── mainline.json         # Authoritative model registry & SHA256 manifest
├── configs/                  # Environment requirements & frozen weight integrity configs
├── data/                     # Training metadata, case definitions, & benchmark splits
├── experiments/              # Frozen training protocols, evaluation logs, & review records
├── reports/                  # LaTeX sources for technical & model-review reports
├── tests/                    # Runtime editor contract tests
├── docs/                     # Architecture, Studio guide, evaluation, & reproducibility docs
├── pyproject.toml            # Packaging metadata & console entrypoints
├── requirements.txt          # CPU / Studio dependencies
├── Makefile                  # Common verify / test / launch targets
└── LICENSE                   # Apache-2.0 License
```

---

## Documentation

- [Architecture & Design](docs/ARCHITECTURE.md) — Dual-mode VLM toggling, controller head, 9-tool editor math, and guardrails
- [Studio & Web UI Guide](docs/STUDIO_GUIDE.md) — Multi-turn workspace, comparison mode, and REST API reference
- [Evaluation & Limitations](docs/EVALUATION.md) — Ablations, dataset composition, and cross-seed replication analysis
- [Reproducibility & Weight Setup](docs/REPRODUCIBILITY.md) — Base model setup, weight verification, and training protocols
- [Contributing Guide](CONTRIBUTING.md) — Development workflow and frozen-protocol boundaries

---

## Acknowledgments & License

- Built on [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
- Licensed under the [Apache License 2.0](LICENSE).
