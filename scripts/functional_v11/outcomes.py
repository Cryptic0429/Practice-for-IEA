"""Explicit decisions for the next experimental model; legacy schemas stay frozen.

A decline contains no result image. This module never guesses a decision from an
old flat-parameter response or turns a failed edit into an unchanged success.
"""
from __future__ import annotations

import json
import re

from functional_v1.common import TOOLS, _pairs, image_path, validate


DECLINE_REASONS = ("unreliable_quality", "unsupported_tools")
SCHEMA_VERSION = "functional-outcome-v1"
EDITOR_VERSION = "functional-editor-v11-negative-value-gain-candidate"


def validate_decision(value):
    if not isinstance(value, dict):
        raise ValueError("decision_must_be_an_object")
    action = value.get("action")
    if action == "edit":
        if set(value) != {"action", "parameters"}:
            raise ValueError("edit_requires_only_action_and_parameters")
        parameters = validate(value["parameters"])
        if not any(parameters.values()):
            raise ValueError("edit_requires_a_nonzero_parameter")
        return {"action": "edit", "parameters": parameters}
    if action == "keep":
        if set(value) != {"action"}:
            raise ValueError("keep_requires_only_action")
        return {"action": "keep"}
    if action == "decline":
        if set(value) != {"action", "reason"}:
            raise ValueError("decline_requires_only_action_and_reason")
        if value["reason"] not in DECLINE_REASONS:
            raise ValueError("invalid_decline_reason")
        return {"action": "decline", "reason": value["reason"]}
    raise ValueError("unknown_decision_action")


def parse(response):
    match = re.fullmatch(r"\s*```json\s*([\s\S]*?)\s*```\s*", response)
    if not match:
        raise ValueError("expected_one_json_code_block_without_other_text")
    return validate_decision(json.loads(match.group(1), object_pairs_hook=_pairs))


def target_text(decision):
    return "```json\n" + json.dumps(validate_decision(decision), ensure_ascii=False, indent=2) + "\n```"


def system_prompt():
    return (
        "Edit the supplied photograph according to the original user request. "
        "Follow the requested direction and strength, preserve unrelated content, "
        "and retain highlights, shadow detail and natural colour unless a colour style is requested. "
        "Return exactly one fenced json code block, using one of three decisions. "
        'For a feasible edit: {"action":"edit","parameters":{"exposure":20}}. '
        "The example is a format example, not a suggested edit. "
        "Parameters may contain only: " + ", ".join(TOOLS) + ". "
        "All parameters range from -100 to 100 except sepia, which ranges from 0 to 100. "
        "Omitted or zero parameters mean no change; an edit must contain a nonzero parameter. "
        "Positive exposure/brightness lighten with bounded curves; negative darken. "
        "Positive contrast increases tonal separation with a soft shoulder and toe; negative reduces it. "
        "Positive highlights/shadows lift that tonal range; negative lowers it. "
        "Positive saturation strengthens colour. Positive temperature warms, negative cools. "
        "Positive tint adds green, negative adds magenta. Sepia makes an overall brown vintage tone; "
        "warming white balance alone does not make green foliage brown. "
        'Only when explicitly asked to leave the photograph unchanged, return {"action":"keep"}. '
        "If the requested result cannot be completed reliably with these tools while preserving basic "
        'image quality, return {"action":"decline","reason":"unreliable_quality"}. '
        "This means no edited image will be returned. Do not decline merely because an image is dark "
        "or imperfect if a suitable edit can complete the request without serious added damage. "
        "If the request requires unavailable operations such as object removal, local replacement, "
        'denoising or resolution recovery, return {"action":"decline","reason":"unsupported_tools"}. '
        "Do not substitute an unrelated global enhancement. Chinese and English requests have equal priority. "
        "Do not output explanations or any other text."
    )


def input_row(case, language):
    """Only the original input photograph, one request and tool instructions."""
    path = str(image_path(case))
    return {
        "images": [path], "extra_info": {"origin_image_path": path},
        "conversations": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": "<image>\n" + case["instructions"][language]},
        ],
    }


def execute(rgb, decision):
    """Return pixels only for edit/keep; callers must not create a declined output."""
    decision = validate_decision(decision)
    identity = {"schema": SCHEMA_VERSION, "editor": EDITOR_VERSION}
    if decision["action"] == "decline":
        return dict(identity, status="declined", reason_code=decision["reason"], parameters=None, result=None)
    if decision["action"] == "keep":
        return dict(identity, status="unchanged", reason_code=None, parameters={}, result=rgb.copy())
    from .editor import edit
    import numpy as np
    result = edit(rgb, decision["parameters"], "bounded-fixed")
    if np.array_equal(rgb, result):
        raise ValueError("edit_produced_no_pixel_change")
    return dict(identity, status="edited", reason_code=None, parameters=decision["parameters"], result=result)
