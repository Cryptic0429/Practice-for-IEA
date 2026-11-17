# Model Versions & Experiment Provenance

Public documentation refers to model components by their semantic names (**Semantic Backbone**, **Controller Head**, **Early Functional Baseline**, …). This appendix is the single place that maps those names to internal registry IDs, training runs, physical archive directories, and frozen experiment records. You do not need any of these details to install or use IEA—they exist for auditing and reproducibility.

The authoritative machine-readable registry remains [`models/mainline.json`](../models/mainline.json) (paths, SHA256 digests, alias symlinks).

---

## 1. Component ↔ Registry Mapping

| Public Name | Registry ID | Semantic Alias | Physical Archive Directory | Selected Training Run |
|---|---|---|---|---|
| Semantic Backbone | `r15` | `models/core/semantic_backbone` | `models/from-functional-v11/sft-shadow-correction/seed-01` | seed `20261127`, epoch-1 checkpoint |
| Controller Head | `r21_head` | `models/core/controller_head` | `models/from-functional-v15/semantic-preserving-controller/seed-01` | seed `20261205`, epoch-20 checkpoint of an 80-epoch run |
| Backbone-Only Ablation | `r15` (head disabled) | `models/baselines/ablation_no_controller` | same adapter as the Semantic Backbone | seed `20261127`, epoch-1 checkpoint |
| Early Functional Baseline | `r1` | `models/baselines/early_functional_baseline` | `models/from-sft-calibrated/sft-functional/seed-01` | seed `20261104`, epoch-1 checkpoint (step 110) of a 4-epoch run |
| Calibrated SFT Baseline | `calibrated_sft` | `models/baselines/early_calibrated_sft` | `models/from-sft-standard/sft-calibrated/seed-02` | seed `20261104`, 50/50 weight average of steps 512 & 1024 |

The failed controller replication run used seed `20261206` (epoch 20); its analysis is preserved in the Controller Head experiment record below.

## 2. Archive Directory Naming

Physical weight directories follow `models/from-<parent-protocol>/<route>/seed-NN`, where the prefix records the **initialization lineage**, not the released component version. For example, the Semantic Backbone adapter was initialized from an intermediate protocol-v11 checkpoint—hence `from-functional-v11`—rather than from the Early Functional Baseline. At inference time only a single final LoRA adapter is loaded onto the frozen 7B base; ancestor adapters are never stacked.

The semantic aliases under `models/core/` and `models/baselines/` are symlinks to these archive directories and are verified by `make verify`. Prefer the alias paths in scripts and documentation; the archive layout is an implementation detail of the registry.

## 3. Protocol Module Map (`scripts/`)

| Module | Role |
|---|---|
| `functional_v15/` | Semantic Backbone training/evaluation protocol; shared-luminance shadow-gain renderer (`editor.py`) used at runtime |
| `functional_v19/` | Global photometric statistics (`photometry.py`) consumed by the Controller Head |
| `functional_v21/` | Controller Head (`head.py`), local texture-residual features (`texture.py`), strict composition contract (`compose()`) |
| `functional_v1/` … `functional_v20/` | Historical protocol iterations, retained byte-exact |
| `goal_v2_*`, `stage2_*`, `evaluation_core.py` | Early SFT/RL calibration, training, and audit scripts |

All frozen modules are covered by the 191 SHA256 code checks in the published training protocols; do not edit them in place (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

## 4. Training Recipes

| Component | Initialization | Recipe Summary | Entrypoint |
|---|---|---|---|
| Calibrated SFT Baseline | Stage A | 3,000 samples, `lr=1e-4`, 1,024 steps, seed `20261104`, 50/50 weight average (steps 512 & 1024) | `scripts/goal_v2_train_json3k_long.py` |
| Early Functional Baseline | Calibrated SFT Baseline | 440 bilingual records, 4 epochs, `lr=1e-4`, effective batch 4, NF4 QLoRA (`r=8, alpha=16`), seed `20261104` | `python -m functional_v1.train --start current --gpu 0 --seed 20261104` |
| Semantic Backbone | intermediate v11 checkpoint | 1,850 bilingual records, 2 epochs (926 steps), `lr=5e-5`, batch 4, seed `20261127`; epoch-1 checkpoint selected | `python -m functional_v15.train --route language_merger --gpu 0 --seed 20261127` |
| Controller Head | frozen Semantic Backbone | 925 semantic pairs, CPU MLP (463,754 params), 80 epochs, `lr=1e-3`, seed `20261205`; epoch-20 checkpoint selected | `python -m functional_v21.train --seed 20261205` |

## 5. Experiment Record Index

`experiments/` preserves the original lab-notebook records (written in Chinese) byte-exact: frozen pre-training protocols, training logs, per-image AI-assisted review observations, and negative/replication-failure results. [docs/EVALUATION.md](EVALUATION.md) is the English summary of what these records establish.

| Record | What It Documents |
|---|---|
| `experiments/functional-v1-20261104/` | Early Functional Baseline trial: protocol, training logs, `results.json`, `REPORT.md` |
| `experiments/functional-v15-20261106/` | Semantic Backbone training and shadow-gain editor; frozen `training-protocol.json` |
| `experiments/functional-v21-20261107/` | Controller Head training, both seeds, and the cross-seed replication-failure analysis |
| `experiments/functional-v23-20261107/` | Follow-up direct-LoRA route on verified quality data; failed admission gates, not adopted |
| `experiments/functional-v7-20261105/`, `functional-v9-20261106/`, `quality-v2-20261103/` | Earlier protocol iterations retained for evidence verification (incl. base-model origin audit) |
| `experiments/comparison-studio-20261110/` | Historical comparison-studio resource ledger |

Files referenced by the published `training-protocol.json` and `evidence.json` manifests are SHA256-verified by `make verify` and must not be modified or reformatted.