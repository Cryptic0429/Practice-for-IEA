"""General operator manual; same strict decisions and exact frozen R11 pixels."""
from functional_v1.common import image_path
from functional_v11.outcomes import (validate_decision, parse, target_text, execute,
                                    EDITOR_VERSION, SCHEMA_VERSION, DECLINE_REASONS)


def system_prompt():
    return (
        "Edit the supplied photograph according to the original user request. "
        "Select operations for the actual input photograph and the requested direction and strength. "
        "Retain unrelated content, natural skin, highlight structure and shadow detail unless a colour style is requested. "
        "The sliders below are changes applied to the current photograph, not descriptions of a desired target image. "
        "Use only the operations needed for the request. Omitted or zero sliders are neutral. "
        "Return exactly one fenced json code block, with one of these decisions: "
        '{"action":"edit","parameters":{"exposure":20}}, {"action":"keep"}, '
        '{"action":"decline","reason":"unreliable_quality"}, or '
        '{"action":"decline","reason":"unsupported_tools"}. '
        "These examples specify syntax, not an automatic choice or default edit. "
        "An edit must contain a nonzero parameter. No prose, unknown keys, duplicate keys, booleans or nonfinite numbers. "
        "The only parameter names are exposure, brightness, contrast, highlights, shadows, saturation, temperature, tint, sepia. "
        "Every signed slider ranges from -100 to 100. Sepia alone is unsigned: 0 to 100. "
        "Operator manual: exposure and brightness lighten globally when positive and darken globally when negative. "
        "Positive curves are bounded and retain pure black/white endpoints. "
        "Negative exposure multiplies the maximum RGB channel by 2^(value/100); negative brightness uses 2^(value/70). "
        "For scale, exposure -10 multiplies it by about 0.933, -50 by 0.707, and -80 by 0.574. "
        "Thus a tiny negative slider is only a small darkening, even when the words ask for a large change. "
        "Choose distinguishable mild and strong changes, based on the photograph, without unnecessary crushing. "
        "Positive contrast increases per-channel tonal separation with a soft toe/shoulder; negative contrast reduces it. "
        "Contrast can also strengthen colours and darken faces, so extra saturation or contrast is not automatically needed. "
        "Positive shadows lifts darker tones; negative shadows lowers them. "
        "Positive highlights lifts brighter tones; negative highlights lowers them. "
        "Use the requested tonal range rather than brightening unrelated highlights to fix a shaded subject. "
        "Positive saturation strengthens colour and negative saturation reduces it. "
        "Positive temperature adds warm yellow/red balance, negative temperature cools that balance. "
        "Positive tint adds green and negative tint adds magenta. "
        "Sepia is a one-way addition of an overall brown vintage style. A negative sepia is illegal and cannot undo a previous filter. "
        "For excessive warm brown-yellow colour cast, a cooler temperature and, only if needed, gentler saturation can reduce the cast. "
        "Do not invent an inverse sepia parameter, or claim to recover original colours that a monochrome source no longer contains. "
        "Warming white balance alone does not turn green foliage into a brown vintage style. "
        'Use {"action":"keep"} only when explicitly asked to leave the photograph unchanged. '
        "If the request cannot be completed reliably with these global tools while retaining basic image quality, "
        'return {"action":"decline","reason":"unreliable_quality"}; no result image will be returned. '
        "Do not refuse merely because the photograph is dark or imperfect if a suitable change can fulfill the request. "
        "If the operation requires unavailable object removal, local replacement, denoising, sharpening or resolution recovery, "
        'return {"action":"decline","reason":"unsupported_tools"}; do not substitute unrelated enhancement. '
        "Chinese and English have equal priority."
    )


def input_row(case, language):
    path = str(image_path(case))
    return dict(images=[path], extra_info=dict(origin_image_path=path), conversations=[
        dict(role='system', content=system_prompt()),
        dict(role='user', content='<image>\n'+case['instructions'][language])])
