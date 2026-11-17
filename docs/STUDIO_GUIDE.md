# IEA Studio Guide

**IEA Studio** (`scripts/iea_studio/`) is a darkroom-inspired Flask web application with two integrated modes:

1. **Conversational Retouching Workspace (Default View)**: Multi-turn natural language photo editing with lossless version timeline, split before/after slider, 1:1 pixel viewer, 9-parameter telemetry, guardrail status badge, and ZIP session export.
2. **Baseline Comparison View**: Side-by-side bilingual (`zh` / `en`) evaluation workbench comparing **IEA Core (Semantic Backbone + Controller Head)**, the **Backbone-Only Ablation**, and the **Early Functional Baseline**, with optional blind-test mode and structured feedback export.

---

## Launching the Server

```bash
# CPU demo mode (no GPU or 7B weights required)
iea-studio --port 12348

# Or via launcher script
bash scripts/iea_studio/launch.sh --port 12348
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--port` | `12348` | Loopback HTTP port (`127.0.0.1`) |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |
| `--gpu` | `0` | CUDA device index when `--device cuda` |
| `--gpu-budget-hours` | `0` | Positive GPU wall-time cap required for `--device cuda` |
| `--enable-agent-vlm` | `false` | Load `DualModeVLMBackend` instead of `RuleBasedCPUPlannerBackend` |
| `--worker-python` | `envs/iea-rl/bin/python` | Python executable for comparison worker subprocesses |

---

## REST API Overview

### Multi-Turn Agent Sessions
- `GET /api/sessions` — List recent editing sessions
- `POST /api/sessions` — Create a session from an uploaded image (`image`) or sample (`sample_id`)
- `GET /api/sessions/<session_id>` — Retrieve session state, active turn, parameters, and message history
- `POST /api/sessions/<session_id>/messages` — Submit a conversational turn (`{"message": "..."}`)
- `POST /api/sessions/<session_id>/checkout` — Jump losslessly to any historical turn (`{"turn": N}`)
- `GET /api/sessions/<session_id>/files/<filename>` — Serve `source.png` or rendered turn PNGs
- `GET /api/sessions/<session_id>/download` — Export full session JSON and renders as a ZIP archive

### Baseline Comparison Jobs
- `GET /api/config` — Return active model metadata (`core_model`, `baselines`) and sample gallery items
- `POST /api/jobs` — Submit a bilingual (`zh`, `en`) comparison job across all 3 models (6 outputs)
- `GET /api/jobs/<job_id>` — Poll job status and per-model render URLs
- `POST /api/jobs/<job_id>/feedback` — Record blinded or unblinded quality ratings (`training_use: false`)
- `GET /api/jobs/<job_id>/download` — Download comparison job archive

---

## Privacy & Safety Isolation

- All uploaded images, comparison jobs, and session histories are stored strictly under runtime directories ignored by `.gitignore` and marked `training_use: false`.
- Same-origin checks protect `POST` endpoints, and maximum image size is bounded (`40M` pixels / `22 MB` upload).
