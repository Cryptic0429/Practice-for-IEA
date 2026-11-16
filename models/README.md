# Model Registry

Authoritative paths, roles, and SHA256 digests are defined in [`mainline.json`](mainline.json).

To preserve frozen protocol path hashes while exposing clean semantic entrypoints, physical weight directories live under `from-*/` and are linked from `core/` and `baselines/`:

| Tier | Component | Semantic Alias | Physical Directory | Role |
|---|---|---|---|---|
| **Core Backbone** | Semantic Decision Backbone (`r15`) | `models/core/semantic_backbone` | `from-functional-v11/sft-shadow-correction/seed-01` | Structured 9-tool direction decision & shared-luminance shadow-gain editor |
| **Core Controller** | Magnitude & Risk Head (`r21_head`) | `models/core/controller_head` | `from-functional-v15/semantic-preserving-controller/seed-01` | 463K-param MLP calibrating parameter magnitudes & predicting quality risk |
| **Ablation** | No-Controller Ablation (`r15`) | `models/baselines/ablation_no_controller` | `from-functional-v11/sft-shadow-correction/seed-01` | Runs `r15` standalone without the controller head |
| **Baseline** | Early Functional Baseline (`r1`) | `models/baselines/early_functional_baseline` | `from-sft-calibrated/sft-functional/seed-01` | First-generation bilingual functional LoRA baseline |
| **Baseline** | Calibrated SFT (`calibrated_sft`) | `models/baselines/early_calibrated_sft` | `from-sft-standard/sft-calibrated/seed-02` | Early parameter-calibrated SFT checkpoint & `r1` initialization |

- Binary weights (`.safetensors`, `.pt`) and the `Qwen/Qwen2.5-VL-7B-Instruct` base model are excluded from Git via `.gitignore`.
- See [Reproducibility & Weight Setup](../docs/REPRODUCIBILITY.md) for base model download instructions and `make verify-strict` verification.
- Research prototype status: `production_qualified: false` (cross-seed controller replication failed on night-scene brightening).
