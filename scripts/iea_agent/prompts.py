"""Planner and Critic prompts plus JSON extraction for the native VLM base."""
from __future__ import annotations

import json
import re
from typing import Any
from .state import TOOLS, TOOL_LABELS, SessionState, clamp_params

PLANNER_SYSTEM_PROMPT = """You are the conversational planner of an interactive global photo editing assistant.
Supported global tools (only these 9 parameters exist):
- exposure (-100..100)
- brightness (-100..100)
- contrast (-100..100)
- highlights (-100..100)
- shadows (-100..100)
- saturation (-100..100)
- temperature (-100..100, positive warms)
- tint (-100..100, positive adds green, negative adds magenta)
- sepia (0..100)

Operations NOT supported: object removal, local inpainting, background replacement, denoising, super-resolution, cropping.
Given the image, the current cumulative parameters, recent conversation turns, and the latest user message, return ONLY one JSON code block with fields:
{
  "intent": "edit" | "adjust_relative" | "undo" | "reset" | "keep" | "unsupported" | "chat",
  "rewritten_instruction": "clean standalone single-turn instruction when intent is edit, else empty string",
  "delta_parameters": {"tool": number},
  "scale_factor": number | null,
  "target_tools": ["tool"],
  "steps_back": 1,
  "reply": "helpful natural-language response in the same language as the user, without mentioning internal model IDs"
}
Rules:
- Use "edit" when starting a new editing goal or changing style from the original request.
- Use "adjust_relative" for incremental follow-ups like "brighter", "warmer", "reduce contrast by half" (provide delta_parameters or scale_factor + target_tools).
- Use "undo" for "undo", "go back", "撤销"; "reset" for "reset", "还原原图".
- Use "unsupported" if the user asks for local editing, object removal, or denoising; explain clearly what global tools can do instead.
- Never mention internal model codes in "reply".
"""

CRITIC_SYSTEM_PROMPT = """You are the visual quality critic for a global photo editing assistant.
Compare the ORIGINAL image and the EDITED image alongside the user request, applied parameters, and luminance/clipping deltas.
Return ONLY one JSON code block:
{
  "passed": true | false,
  "issue": "none" | "overexposed" | "noisy_shadows" | "color_cast" | "under_edited" | "unnatural",
  "suggested_scale": 0.6,
  "reason": "concise natural-language explanation in the user's language, without mentioning internal model IDs"
}
Fail only if highlights blow out, dark shadows show severe amplified noise, strong unnatural color cast appears, or the edit clearly contradicts the request.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from fenced or raw model output."""
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    raw = fenced.group(1) if fenced else text.strip()
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError('expected_json_object')
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError('expected_json_object')
    return data


def build_planner_prompt(session: SessionState, user_message: str) -> str:
    """Render concise context for the native VLM planner."""
    recent = session.messages[-6:]
    turns_text = '\n'.join(f"- {m['role']}: {m['text']}" for m in recent) if recent else '(none)'
    params_text = json.dumps(session.current_params, ensure_ascii=False)
    return (
        f"{PLANNER_SYSTEM_PROMPT}\n"
        f"Current cumulative parameters: {params_text}\n"
        f"Active turn: {session.active_turn}\n"
        f"Recent conversation:\n{turns_text}\n"
        f"Latest user message: {user_message}\n"
    )


def build_critic_prompt(
    user_message: str,
    params: dict[str, float],
    before_stats: dict[str, float],
    after_stats: dict[str, float],
) -> str:
    """Render context for visual critic inspection of (source, edited)."""
    return (
        f"{CRITIC_SYSTEM_PROMPT}\n"
        f"User request: {user_message}\n"
        f"Applied cumulative parameters: {json.dumps(params, ensure_ascii=False)}\n"
        f"Before stats: {json.dumps(before_stats, ensure_ascii=False)}\n"
        f"After stats: {json.dumps(after_stats, ensure_ascii=False)}\n"
    )


def format_params_brief(params: dict[str, float]) -> str:
    clean = clamp_params(params)
    if not clean:
        return '无调整（保持原图）'
    parts = []
    for key in TOOLS:
        if key in clean:
            val = clean[key]
            sign = '+' if val > 0 else ''
            parts.append(f"{TOOL_LABELS[key]} {sign}{val}")
    return '、'.join(parts)
