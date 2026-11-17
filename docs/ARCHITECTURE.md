# System Architecture

IEA separates **high-level conversational reasoning**, **structured parameter & risk prediction**, **deterministic pixel rendering**, and **quality safety verification** into four composable layers.

> Module paths such as `scripts/functional_v15/` and `scripts/functional_v21/` are internal protocol version directories; the mapping between public component names and these versions is documented in [Model Versions & Provenance](MODEL_VERSIONS.md).

---

## 1. Dual-Mode Single-VLM Backend (`scripts/iea_agent/backend.py`)

Rather than loading separate LLM/VLM instances for chat planning, retouching, and visual critique, `DualModeVLMBackend` loads one 4-bit NF4 / BF16 `Qwen2.5-VL-7B-Instruct` base model and toggles PEFT adapters with zero VRAM overhead:

| Stage | Adapter State | Active Weights | Responsibility |
|---|---|---|---|
| **Planner** | Disabled (`disable_adapter()`) | Native `Qwen2.5-VL-7B` | Multi-turn intent classification (`edit`, `adjust_relative`, `undo`, `reset`, `keep`, `unsupported`, `chat`), relative delta/scale extraction, and instruction normalization |
| **Predictor** | Enabled (`semantic_backbone`) | Semantic Backbone LoRA + Controller Head MLP | Structured 9-tool decision (`edit`, `keep`, `decline`), direction locking, magnitude calibration, and quality-risk estimation |
| **Critic** | Disabled (`disable_adapter()`) | Native `Qwen2.5-VL-7B` | Side-by-side visual comparison of `source_rgb` vs. rendered output against luminance/clipping deltas |

For local development without a GPU, `RuleBasedCPUPlannerBackend` (`mode_name = "cpu_rule_fallback"`) provides deterministic bilingual intent parsing so the entire multi-turn state machine, renderer, and guardrail pipeline can be exercised end-to-end on CPU.

---

## 2. Magnitude & Risk Controller Head (`scripts/functional_v21/head.py`)

The Semantic Backbone determines **whether** to edit/keep/decline, **which** tools to activate, and **in which direction** (sign). The frozen backbone adapter is then paired with a 463,754-parameter MLP (`Head`) trained on 925 bilingual semantic pairs:

- **Inputs (3,612-D)**:
  - `3584-D` RMS-normalized last-token hidden state from the frozen 7B backbone
  - `16-D` global image statistics (`functional_v19/photometry.py`)
  - `12-D` local luminance/chroma median-filter residual features (`functional_v21/texture.py`)
- **Outputs**:
  - `9-D` normalized magnitudes in `[0, 1]` mapped to `[1, 100]` (`sigmoid`)
  - `1-D` uncalibrated risk logit (`unreliable_quality` threshold `sigmoid(q) >= 0.5`)
- **Strict Composition Contract (`compose()`)**:
  - Preserves parent `keep` and `decline` outputs verbatim.
  - Never invents tools that the parent did not activate, never flips signs, and raises `head_zero_for_parent_active_tool` rather than silently dropping an active tool.

---

## 3. Deterministic Non-Destructive Renderer (`scripts/functional_v15/editor.py`)

All image edits are parameterized by 9 global sliders:

`exposure`, `brightness`, `contrast` (`[-100, 100]`, scale `2^(value/100)`), `highlights`, `shadows`, `saturation`, `temperature`, `tint` (`[-100, 100]`), and `sepia` (`[0, 100]`).

Key invariants verified by `tests/test_runtime.py`:
- **Source-Anchored Re-Rendering**: `SessionState` stores the immutable `source.png` and a stack of turn snapshots. Every turn updates `current_params` and calls `edit(source_rgb, current_params, "bounded-fixed")` directly from the original pixels.
- **Zero Cumulative Degradation**: Multi-turn adjustments, undo (`steps_back`), checkout to any historical turn, and reset (`turn 0`) never re-encode intermediate outputs.
- **Exact No-Op Guarantee**: Empty parameter dicts or `keep` actions return bit-identical RGB pixels.
- **Shared-Luminance Shadow Gain**: Shadows are lifted using a shared luminance gain mask rather than per-channel gamma, reducing gray haze and hue shifts in dark regions.

---

## 4. Two-Stage Self-Check Loop (`scripts/iea_agent/orchestrator.py`, `guardrails.py`)

After each candidate render, `ImageEditingAgent.handle_turn` runs two checks:

1. **CPU Hard-Metric Guardrail (`evaluate_cpu_guardrail`)**:
   - Flags `highlight_clipping` if `white_clip_delta > 0.08` or (`white_before <= 0.08` and `white_after > 0.18`).
   - Flags `shadow_crushing` if `black_clip_delta > 0.08` or (`black_before <= 0.10` and `black_after > 0.22`).
   - Flags `contrast_collapse` if `std_before >= 0.06` and `std_after < 0.025`.
2. **Visual Critic (`backend.critique_render`)**:
   - Inspects original and edited images alongside parameter values and clipping metrics.

If either check fails, the agent attenuates the turn's parameter delta by `attenuation = 0.6` (configurable via `ImageEditingAgent(..., max_retries=2, attenuation=0.6)`) and re-renders from `source_rgb`. If risks persist after retries, the turn rolls back cleanly to the pre-turn state.
