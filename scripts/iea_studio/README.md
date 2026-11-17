# IEA Studio (`scripts/iea_studio/`)

Web workspace combining a **multi-turn conversational retouching studio** (default view) and a **bilingual baseline comparison view**.

## Modes

1. **Conversational Retouching (Default View)**:
   - Always re-renders from the immutable original upload (`source.png`) using cumulative 9-D global parameters—supporting lossless undo, reset, and timeline turn checkout.
   - Runs in `cpu_rule_fallback` mode out of the box without a GPU, or switches to `gpu_dual_mode` (`--enable-agent-vlm --device cuda`) when the 7B base + weights are configured.
   - Executes post-render highlight/shadow/contrast guardrails and Critic checks, attenuating or rolling back risky edits automatically.
2. **Baseline Comparison**:
   - Compares **IEA Core (Semantic Backbone + Controller Head)**, the **Backbone-Only Ablation**, and the **Early Functional Baseline** side by side across Chinese & English prompts, with optional blind-test mode and 1:1 pixel inspection.

## Quick Launch & Testing

```bash
# Launch locally on CPU
iea-studio --port 12348
# or: bash scripts/iea_studio/launch.sh --port 12348

# Run Flask API & editor contract tests
make test-runtime
```
