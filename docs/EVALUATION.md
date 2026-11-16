# Evaluation & Limitations

This repository preserves complete experimental protocols, training logs, and observation records under `experiments/` and `reports/` so every metric can be audited directly.

---

## 1. Benchmark Scope & Metric Definitions

All functional evaluations distinguish clearly between split types and never aggregate incompatible metrics (such as training loss, pixel L1, execution rate, and bilingual functional completion) into a single score:

- **Bilingual Functional Completion**: A case counts as passed only when **both** Chinese and English prompts produce a valid parameter edit (or appropriate behavior) **and** pass direct visual inspection without major artifacts.
- **Controlled Dev (100 cases / 200 bilingual items)**: Structured test suite spanning sepia, compound edits, contrast, keep-unchanged, lighting/exposure, saturation, and white balance.
- **Raw Dev (20 cases)** & **Legacy Regression (6 cases)**: Natural/unconstrained phrasing probes and historical regression checks.
- **Evaluation Method**: AI-assisted visual inspection (direct image review with model identities hidden during comparison); not an independent human user study.

---

## 2. Mainline & Baseline Results

### Early Functional Baseline (`r1`) vs. Prior Calibrated SFT (`calibrated_sft`)

From `experiments/functional-v1-20261104/REPORT.md`:

| Model | Controlled Dev (50 groups) | Frozen Acceptance (50 groups) | Notes |
|---|---:|---:|---|
| `calibrated_sft` (with new prompt & editor) | 10 / 50 | 15 / 50 | 1 major artifact, 1 non-executable output |
| `r1` (`seed 20261104`, epoch 1) | **43 / 50** | **31 / 50** | 100/100 executable; strong sepia (`10/10`) & lighting (`8/10`), contrast regression (`2/10` vs. `6/10`) |

### Semantic Backbone (`r15`) & Controller Head (`r21_head`) Ablation

From `experiments/functional-v15-20261106/REPORT.md` and `experiments/functional-v21-20261107/REPORT.md`:

| Configuration | Seed / Checkpoint | Legacy Reg. (6) | Raw Dev (20) | Keep / Reject Control (20) | Controlled Dev (100) | Major Artifacts | Status |
|---|---|---|---|---|---|---|---|
| `r15` (CPU editor replay from R11 parent) | — | 4 / 6 | 15 / 20 | 20 / 20 | 76 / 100 | 1 | Diagnostic baseline |
| `r15` (`ablation_no_controller`) | `20261127` / e1 | 4 / 6 | 16 / 20 | 20 / 20 | 75 / 100 | 1 (orange skin tone) | Below `81/100` gate |
| `r15 + r21_head` (magnitude only) | `20261205` / e20 | 5 / 6 | 16 (1 major) | 20 / 20 | — | 1 (night-scene noise) | Failed raw dev gate |
| **`r15 + r21_head` (magnitude + quality)** | **`20261205` / e20** | **5 / 6** | **16 / 20** | **20 / 20** | **82 / 100** | **0** | **Primary research checkpoint** |
| `r15 + r21_head` (replication seed) | `20261206` / e20 | 5 / 6 | ≥11 (1 major) | 20 / 20 | — | 1 (night-scene noise) | **Replication failed** |

---

## 3. Key Findings & Known Limitations

1. **Cross-Seed Instability of Quality Risk Rejection (`production_qualified: false`)**:
   - On the same night-scene brightening case (`zh`), uncalibrated risk `sigmoid` values varied sharply across training runs: `0.6533` (`seed 1, e20`, triggered refusal at `0.5`), `~1.23e-10` (`seed 1, e80`), and `0.00254` (`seed 2, e20`, failed to refuse and amplified red/green sensor noise).
   - Cause hypotheses noted in `functional-v21` diagnostics include limited distinct image groups for quality rejections (23 groups) and 512px thumbnail residuals conflating real texture/edges with sensor noise.
2. **Global Scope Only**:
   - Supports 9 global color/tone adjustments. Local edits, object removal, inpainting, cropping, and denoising are intentionally rejected or explained by the Planner layer.
3. **Research-Grade Runtime**:
   - `r15 + r21_head` is retained as the primary research configuration for `iea_agent` + `iea_studio`, where the outer CPU guardrail and visual Critic provide an additional safety net—not as a production-certified photo editor.
