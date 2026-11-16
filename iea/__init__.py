"""Public package facade for Interactive Image Editing Agent (IEA)."""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / 'scripts'
if _SCRIPTS.is_dir() and str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from iea_agent import (  # noqa: E402
    GuardrailResult,
    ImageEditingAgent,
    SessionState,
    TOOL_LABELS,
    TOOLS,
    clamp_params,
    compute_statistics,
    evaluate_cpu_guardrail,
    merge_params,
    scale_delta,
)
from iea_agent.backend import DualModeVLMBackend, RuleBasedCPUPlannerBackend  # noqa: E402

__all__ = [
    'TOOLS',
    'TOOL_LABELS',
    'SessionState',
    'GuardrailResult',
    'ImageEditingAgent',
    'DualModeVLMBackend',
    'RuleBasedCPUPlannerBackend',
    'clamp_params',
    'merge_params',
    'scale_delta',
    'compute_statistics',
    'evaluate_cpu_guardrail',
]

__version__ = '0.1.0'
