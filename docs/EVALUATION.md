# Evaluation & Limitations

This page summarizes the verified empirical results for the released models. The complete raw records—frozen pre-training protocols, training logs, and per-image review observations—are preserved under `experiments/` and `reports/` so every metric can be audited directly; see the [experiment record index](MODEL_VERSIONS.md#5-experiment-record-index) for where each result comes from.

---

## 1. Benchmark Scope & Metric Definitions

All functional evaluations distinguish clearly between split types and never aggregate incompatible metrics (such as training loss, pixel L1, execution rate, and bilingual functional completion) into a single score:

- **Bilingual Functional Completion**: A case counts as passed only when **both** Chinese and English prompts produce a valid parameter edit (or appropriate behavior) **and** pass direct visual inspection without major artifacts.
- **Controlled Dev (100 cases / 200 bilingual items)**: Structured test suite spanning sepia, compound edits, contrast, keep-unchanged, lighting/exposure, saturation, and white balance.
- **Raw Dev (20 cases)** & **Legacy Regression (6 cases)**: Natural/unconstrained phrasing probes and historical regression checks.
- **Evaluation Method**: AI-assisted visual inspection (direct image review with model identities hidden during comparison); not an independent human user study.

---

## 2. Mainline & Baseline Results

### Early Functional Baseline vs. Calibrated SFT Baseline

| Model | Controlled Dev (50 groups) | Frozen Acceptance (50 groups) | Notes |
|---|---:|---:|---|
| Calibrated SFT Baseline (with new prompt & editor) | 10 / 50 | 15 / 50 | 1 major artifact, 1 non-executable output |
| **Early Functional Baseline** | **43 / 50** | **31 / 50** | 100/100 executable; strong sepia (`10/10`) & lighting (`8/10`), contrast regression (`2/10` vs. `6/10`) |

### Controller Head Ablation

The Semantic Backbone alone is compared against the same backbone paired with the Controller Head. The magnitude-only and magnitude + quality-risk variants share **identical weights** and differ only in whether the trained risk output is acted upon; the replication row is an independent training run of the same recipe.

| Configuration | Legacy Reg. (6) | Raw Dev (20) | Keep / Reject Control (20) | Controlled Dev (100) | Major Artifacts | Status |
|---|---|---|---|---|---|---|
| Semantic Backbone (CPU editor replay, diagnostic) | 4 / 6 | 15 / 20 | 20 / 20 | 76 / 100 | 1 | Diagnostic baseline |
| Backbone-Only Ablation | 4 / 6 | 16 / 20 | 20 / 20 | 75 / 100 | 1 (orange skin tone) | Below `81/100` gate |
| IEA Core — magnitude-only controller | 5 / 6 | 16 (1 major) | 20 / 20 | — | 1 (night-scene noise) | Failed raw dev gate |
| **IEA Core — magnitude + quality-risk (primary)** | **5 / 6** | **16 / 20** | **20 / 20** | **82 / 100** | **0** | **Primary research configuration** |
| IEA Core — independent replication run | 5 / 6 | ≥11 (1 major) | 20 / 20 | — | 1 (night-scene noise) | **Replication failed** |

Exact training seeds and selected checkpoints for every row are listed in [Model Versions & Provenance](MODEL_VERSIONS.md).

---

## 3. Key Findings & Known Limitations

1. **Cross-Seed Instability of Quality Risk Rejection (`production_qualified: false`)**:
   - On the same night-scene brightening case (`zh`), uncalibrated risk `sigmoid` values varied sharply across training runs: `0.6533` (primary run, epoch 20 — correctly triggered refusal at the `0.5` threshold), `~1.23e-10` (primary run, epoch 80), and `0.00254` (replication run, epoch 20 — failed to refuse and amplified red/green sensor noise).
   - Cause hypotheses recorded in the Controller Head experiment diagnostics include the limited number of distinct image groups for quality rejections (23 groups) and 512px thumbnail residuals conflating real texture/edges with sensor noise.
2. **Global Scope Only**:
   - Supports 9 global color/tone adjustments. Local edits, object removal, inpainting, cropping, and denoising are intentionally rejected or explained by the Planner layer.
3. **Research-Grade Runtime**:
   - IEA Core (Semantic Backbone + Controller Head) is retained as the primary research configuration for the agent and Studio, where the outer CPU guardrail and visual Critic provide an additional safety net—not as a production-certified photo editor.