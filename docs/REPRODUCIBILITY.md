# Reproducibility & Weight Setup

This repository tracks all source code, frozen training/evaluation protocols, adapter configurations, and verified experimental evidence in Git while excluding large binary model weights and raw image corpora via `.gitignore`.

---

## 1. Repository Integrity Verification

Run the built-in verifier to confirm model layout, symlink aliases, frozen protocol code hashes, and retained evidence integrity:

```bash
make verify
```

In a fresh Git checkout (without local `.safetensors` / `.pt` files), `make verify` checks:
- `models/mainline.json` schema and semantic symlink targets (`models/core/*`, `models/baselines/*`)
- **191** frozen protocol source-code SHA256 checks (`experiments/functional-v15-20261106/training-protocol.json` and `experiments/functional-v21-20261107/training-protocol.json`)
- **669** retained experimental evidence SHA256 checks (`reports/mainline-technical-20261113/generated/evidence.json`)

If local weights are present and you want strict binary checksum verification:

```bash
make verify-strict
# or: iea-verify --require-weights
```

---

## 2. Base Model & Adapter Setup (For GPU Inference)

To enable `DualModeVLMBackend` (`--enable-agent-vlm --device cuda`):

1. **Download the Base Model** (`Qwen/Qwen2.5-VL-7B-Instruct`, pinned revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`):
   ```bash
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
     --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
     --local-dir models/Qwen2.5-VL-7B-Instruct
   ```
2. **Place Trained Adapters & Controller Head** at the canonical paths specified in [`models/mainline.json`](../models/mainline.json):
   - `models/from-functional-v11/sft-shadow-correction/seed-01/adapter_model.safetensors` (`sha256: 97c84f1799fb...`)
   - `models/from-functional-v15/semantic-preserving-controller/seed-01/controller.pt` (`sha256: 838342a21670...`)
   - *(Optional baselines)* `models/from-sft-calibrated/sft-functional/seed-01/adapter_model.safetensors` and `models/from-sft-standard/sft-calibrated/seed-02/adapter_model.safetensors`
3. **Verify Binary Checksums**:
   ```bash
   make verify-strict
   ```

> **Note on `origin-audit.json`**: `scripts/functional_v9/runtime.py::verify_origin('base')` checks base shard sizes and `mtime_ns` against `experiments/functional-v9-20261106/origin-audit.json` for strict historical training replay. When running inference on newly downloaded weights, audit your local base directory hashes independently rather than reusing machine-specific timestamps.

---

## 3. Training Recipes (Reference)

| Stage | Initialization | Recipe Summary | Entrypoint |
|---|---|---|---|
| `calibrated_sft` | Stage A | 3,000 samples, `lr=1e-4`, 1,024 steps, `seed=20261104`, 50/50 weight average (ckpt 512 & 1024) | `scripts/goal_v2_train_json3k_long.py` |
| `r1` | `calibrated_sft` | 440 bilingual records, 4 epochs, `lr=1e-4`, effective batch 4, NF4 QLoRA (`r=8, alpha=16`) | `python -m functional_v1.train --start current --gpu 0 --seed 20261104` |
| `r15` | `r11` final | 1,850 bilingual records, 2 epochs (926 steps), `lr=5e-5`, batch 4, `seed=20261127` | `python -m functional_v15.train --route language_merger --gpu 0 --seed 20261127` |
| `r21_head` | Frozen `r15` | 925 semantic pairs, CPU MLP (`463,754` params), 80 epochs, `lr=1e-3`, `seed=20261205` (e20) | `python -m functional_v21.train --seed 20261205` |

*Lineage Note*: `r15` was initialized from `r11` (reflected in the physical directory prefix `from-functional-v11`), **not** directly from `r1`. At inference time, only a single final LoRA adapter is loaded onto the frozen 7B base—ancestor LoRAs are never stacked.

---

## 4. Technical Reports

LaTeX sources for the technical and model-review reports are retained in `reports/`. To compile the mainline technical report (requires Tectonic + CJK/Arial fonts):

```bash
make report
make clean-report
```
