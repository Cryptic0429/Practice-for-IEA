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
2. **Place Trained Adapters & Controller Head** under the semantic aliases (symlinks that resolve to the canonical archive paths in [`models/mainline.json`](../models/mainline.json), where expected SHA256 digests are listed):
   - `models/core/semantic_backbone/adapter_model.safetensors` — Semantic Backbone LoRA
   - `models/core/controller_head/controller.pt` — Controller Head
   - *(Optional baselines)* `models/baselines/early_functional_baseline/adapter_model.safetensors` and `models/baselines/early_calibrated_sft/adapter_model.safetensors`
3. **Verify Binary Checksums**:
   ```bash
   make verify-strict
   ```

> **Note on `origin-audit.json`**: `scripts/functional_v9/runtime.py::verify_origin('base')` checks base shard sizes and `mtime_ns` against `experiments/functional-v9-20261106/origin-audit.json` for strict historical training replay. When running inference on newly downloaded weights, audit your local base directory hashes independently rather than reusing machine-specific timestamps.

---

## 3. Training Recipes (Reference)

Per-component training recipes, initialization lineage, seeds, selected checkpoints, and entrypoint commands are documented in [Model Versions & Provenance §4](MODEL_VERSIONS.md#4-training-recipes), alongside the [experiment record index](MODEL_VERSIONS.md#5-experiment-record-index).

*Lineage Note*: the Semantic Backbone was initialized from an intermediate protocol checkpoint, **not** directly from the Early Functional Baseline (this is what the `from-functional-v11` archive prefix records). At inference time, only a single final LoRA adapter is loaded onto the frozen 7B base—ancestor LoRAs are never stacked.

---

## 4. Technical Reports

LaTeX sources for the technical and model-review reports are retained in `reports/`. To compile the mainline technical report (requires Tectonic + CJK/Arial fonts):

```bash
make report
make clean-report
```
