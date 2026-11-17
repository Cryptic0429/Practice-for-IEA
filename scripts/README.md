# Source & Scripts Directory

## Active Application & Agent Modules

- `iea_agent/`: Multi-turn conversational image editing orchestrator (`orchestrator.py`), immutable session state & parameter stack (`state.py`), CPU hard-metric guardrails (`guardrails.py`), Planner/Critic prompts (`prompts.py`), dual-mode PEFT + CPU fallback backends (`backend.py`), and stdlib unit tests (`test_agent.py`).
- `iea_studio/`: Darkroom-style Flask Web Studio (`app.py`, `sessions.py`, `worker.py`, `common.py`, `templates/`, `static/`, `test_app.py`) supporting interactive multi-turn editing and bilingual baseline comparisons.
- `project/`: Repository integrity & layout verifier (`assets.py`, `test_assets.py`).

## Frozen Protocol & Legacy Research Modules

These directories carry internal protocol version numbers; the mapping to public component names is in [Model Versions & Provenance](../docs/MODEL_VERSIONS.md#3-protocol-module-map-scripts).

- `functional_v15/`: Semantic Backbone training/evaluation protocol and the shared-luminance shadow-gain renderer (`editor.py`) used at runtime.
- `functional_v21/`: Controller Head (`head.py`), texture residual extractor (`texture.py`), strict composition contract (`compose()`), and training/evaluation scripts.
- `functional_v1/` – `functional_v20/`: Historical protocol modules retained byte-exact to preserve the 191 SHA256 code checks in `make verify`.
- `goal_v2_*`, `stage2_*`, `evaluation_core.py`: Historical calibration and RL training/audit scripts.

> For a clean top-level Python API and CLI entrypoints (`iea-studio`, `iea-verify`), import from the root `iea` package (`pip install -e .`).
