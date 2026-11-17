# Model Registry

Authoritative paths, roles, and SHA256 digests are defined in [`mainline.json`](mainline.json). Always reference models through their **semantic aliases**:

| Tier | Component | Semantic Alias | Role |
|---|---|---|---|
| **Core** | Semantic Backbone | `models/core/semantic_backbone` | Structured 9-tool direction decision; paired with the shared-luminance shadow-gain renderer |
| **Core** | Controller Head | `models/core/controller_head` | 463K-param MLP calibrating parameter magnitudes & predicting quality risk |
| **Baseline** | Backbone-Only Ablation | `models/baselines/ablation_no_controller` | Runs the Semantic Backbone standalone without the Controller Head |
| **Baseline** | Early Functional Baseline | `models/baselines/early_functional_baseline` | First-generation bilingual functional LoRA baseline |
| **Baseline** | Calibrated SFT Baseline | `models/baselines/early_calibrated_sft` | Early parameter-calibrated SFT checkpoint; initialization for the Early Functional Baseline |

Layout notes:

- The aliases are symlinks into physical archive directories (`from-*/`) whose names record training lineage. This preserves frozen protocol path hashes while exposing clean entrypoints; the full mapping (registry IDs, seeds, archive paths) is in [Model Versions & Provenance](../docs/MODEL_VERSIONS.md).
- Binary weights (`.safetensors`, `.pt`) and the `Qwen/Qwen2.5-VL-7B-Instruct` base model are excluded from Git via `.gitignore`.
- See [Reproducibility & Weight Setup](../docs/REPRODUCIBILITY.md) for base model download instructions and `make verify-strict` verification.
- Research prototype status: `production_qualified: false` (independent controller replication failed on night-scene brightening; see [Evaluation & Limitations](../docs/EVALUATION.md)).