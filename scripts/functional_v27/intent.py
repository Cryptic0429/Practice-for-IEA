"""Explicit positive shadow-strength parser for the R27 diagnostic scope."""
import re

SHADOW = re.compile(r'\b(shadows?|dark (areas|parts|tones)|shadowed)\b|暗部|阴影|暗处|暗侧',re.I)
LIFT = re.compile(r'\b(lift|brighten|lighten|open|raise)\b|提亮|抬亮|抬高|打开',re.I)
MILD = re.compile(r'\b(slight(?:ly)?|gentle|gently|a touch|a little|subtle|a hint)\b|轻微|轻轻|略微|稍微|小幅|一点点|微微',re.I)
STRONG = re.compile(r'\b(noticeable|noticeably|substantial(?:ly)?|marked(?:ly)?|pronounced|significant(?:ly)?|a clear lift)\b|明显|显著|大幅',re.I)
NEGATED_LIFT = re.compile(
    r"\b(?:do not|don't|never)\s+(?:lift|brighten|lighten|open|raise)\b|"
    r'别\s*(?:提亮|抬亮)|不要\s*(?:提亮|抬亮)|无需\s*(?:提亮|抬亮)',re.I)


def explicit_shadow_strength(instruction):
    """Return -1 or +1 for recognized degree. Abstain on unknown or mixed requests."""
    if NEGATED_LIFT.search(instruction) or not SHADOW.search(instruction) or not LIFT.search(instruction):
        return None
    mild,strong = bool(MILD.search(instruction)),bool(STRONG.search(instruction))
    return (-1 if mild else 1) if mild != strong else None
