"""Session state, non-destructive parameter stack, and math helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import math
from typing import Any

TOOLS = (
    'exposure',
    'brightness',
    'contrast',
    'highlights',
    'shadows',
    'saturation',
    'temperature',
    'tint',
    'sepia',
)

TOOL_LABELS = {
    'exposure': '曝光',
    'brightness': '亮度',
    'contrast': '对比度',
    'highlights': '高光',
    'shadows': '阴影',
    'saturation': '饱和度',
    'temperature': '色温',
    'tint': '色调',
    'sepia': '复古',
}


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _validate_mapping(params: dict[str, Any] | None, *, allow_negative_sepia: bool) -> dict[str, float]:
    if not params:
        return {}
    if not isinstance(params, dict):
        raise ValueError('parameters_must_be_an_object')
    cleaned: dict[str, float] = {}
    for key, raw in params.items():
        if key not in TOOLS:
            raise ValueError(f'unsupported_parameter:{key}')
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f'parameter_must_be_numeric:{key}')
        if not math.isfinite(raw):
            raise ValueError(f'parameter_not_finite:{key}')
        low = -100.0 if (allow_negative_sepia or key != 'sepia') else 0.0
        clamped = max(low, min(100.0, float(raw)))
        rounded = round(clamped, 2)
        if abs(rounded) >= 0.5:
            cleaned[key] = int(round(rounded)) if abs(rounded - round(rounded)) < 1e-6 else rounded
    return cleaned


def clamp_params(params: dict[str, Any] | None) -> dict[str, float]:
    """Validate and clamp a 9-tool parameter dictionary, dropping zero values."""
    return _validate_mapping(params, allow_negative_sepia=False)


def validate_delta(delta: dict[str, Any] | None) -> dict[str, float]:
    """Validate incremental parameter deltas (allowing negative sepia reductions)."""
    return _validate_mapping(delta, allow_negative_sepia=True)


def merge_params(base: dict[str, Any] | None, delta: dict[str, Any] | None) -> dict[str, float]:
    """Add parameter deltas onto base parameters and clamp to valid bounds."""
    merged: dict[str, float] = {}
    base_clean = clamp_params(base)
    delta_clean = validate_delta(delta)
    for key in TOOLS:
        total = float(base_clean.get(key, 0.0)) + float(delta_clean.get(key, 0.0))
        if total:
            merged[key] = total
    return clamp_params(merged)


def diff_params(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, float]:
    """Compute parameter delta from before -> after."""
    b = clamp_params(before)
    a = clamp_params(after)
    delta: dict[str, float] = {}
    for key in TOOLS:
        change = float(a.get(key, 0.0)) - float(b.get(key, 0.0))
        if abs(change) >= 0.5:
            delta[key] = round(change, 2)
    return delta


def scale_delta(delta: dict[str, Any] | None, factor: float) -> dict[str, float]:
    """Attenuate a parameter delta by factor (e.g., 0.6 on critic retry)."""
    if not delta or factor <= 0:
        return {}
    scaled = {}
    for key, val in validate_delta(delta).items():
        new_val = float(val) * float(factor)
        if abs(new_val) >= 1.0:
            scaled[key] = round(new_val, 2)
    return validate_delta(scaled)


@dataclass
class SessionState:
    """Persistent multi-turn editing session anchored to an immutable source image."""

    session_id: str
    title: str
    source_image: str
    source_sha256: str
    image_size: list[int]
    created_at: str = field(default_factory=utc_stamp)
    updated_at: str = field(default_factory=utc_stamp)
    source_stats: dict[str, float] = field(default_factory=dict)
    current_params: dict[str, float] = field(default_factory=dict)
    active_turn: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    sample_id: str | None = None
    training_use: bool = False

    def ensure_initial_snapshot(self, source_stats: dict[str, float] | None = None) -> None:
        if source_stats:
            self.source_stats = dict(source_stats)
        if not self.history:
            self.history.append({
                'turn': 0,
                'label': '初始原图',
                'action': 'source',
                'params': {},
                'delta': {},
                'image_file': 'source.png',
                'stats': dict(self.source_stats),
                'critic': {'passed': True, 'stage': 'source', 'retries': 0, 'issues': [], 'summary': '原始输入图像'},
                'created_at': self.created_at,
            })
            self.active_turn = 0

    def active_snapshot(self) -> dict[str, Any]:
        for item in self.history:
            if item['turn'] == self.active_turn:
                return item
        return self.history[-1] if self.history else {'turn': 0, 'params': {}, 'image_file': 'source.png', 'stats': self.source_stats}

    def commit_snapshot(
        self,
        *,
        label: str,
        action: str,
        params: dict[str, Any],
        delta: dict[str, Any],
        image_file: str,
        stats: dict[str, float],
        critic: dict[str, Any],
        rewritten_instruction: str = '',
    ) -> dict[str, Any]:
        next_turn = (self.history[-1]['turn'] + 1) if self.history else 1
        clean = clamp_params(params)
        snapshot = {
            'turn': next_turn,
            'label': label,
            'action': action,
            'params': clean,
            'delta': dict(delta),
            'image_file': image_file,
            'stats': dict(stats),
            'critic': dict(critic),
            'rewritten_instruction': rewritten_instruction,
            'created_at': utc_stamp(),
        }
        self.history.append(snapshot)
        self.current_params = clean
        self.active_turn = next_turn
        self.updated_at = utc_stamp()
        return snapshot

    def checkout_turn(self, turn_index: int) -> dict[str, Any]:
        for item in self.history:
            if item['turn'] == turn_index:
                self.active_turn = turn_index
                self.current_params = clamp_params(item['params'])
                self.updated_at = utc_stamp()
                return item
        raise KeyError(f'turn_not_found:{turn_index}')

    def append_message(self, role: str, text: str, **meta: Any) -> dict[str, Any]:
        entry = {'role': role, 'text': text, 'at': utc_stamp(), **meta}
        self.messages.append(entry)
        self.updated_at = utc_stamp()
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            'session_id': self.session_id,
            'title': self.title,
            'source_image': self.source_image,
            'source_sha256': self.source_sha256,
            'image_size': self.image_size,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'source_stats': self.source_stats,
            'current_params': self.current_params,
            'active_turn': self.active_turn,
            'history': self.history,
            'messages': self.messages,
            'sample_id': self.sample_id,
            'training_use': self.training_use,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'SessionState':
        return cls(
            session_id=payload['session_id'],
            title=payload['title'],
            source_image=payload['source_image'],
            source_sha256=payload['source_sha256'],
            image_size=list(payload['image_size']),
            created_at=payload.get('created_at', utc_stamp()),
            updated_at=payload.get('updated_at', utc_stamp()),
            source_stats=dict(payload.get('source_stats', {})),
            current_params=clamp_params(payload.get('current_params', {})),
            active_turn=int(payload.get('active_turn', 0)),
            history=list(payload.get('history', [])),
            messages=list(payload.get('messages', [])),
            sample_id=payload.get('sample_id'),
            training_use=bool(payload.get('training_use', False)),
        )
