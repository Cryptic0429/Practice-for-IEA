"""CPU hard-metric guardrails and image statistics helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    severity: str = 'ok'
    summary: str = '图像硬指标正常'
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'passed': self.passed,
            'issues': list(self.issues),
            'severity': self.severity,
            'summary': self.summary,
            'metrics': dict(self.metrics),
        }


def compute_statistics(rgb: Any) -> dict[str, float]:
    """Compute image statistics via functional_v15.editor when available."""
    try:
        from functional_v15.editor import statistics
        return statistics(rgb)
    except ImportError:
        # Pure-Python fallback for nested RGB sequences in lightweight unit tests.
        pixels = [px for row in rgb for px in row]
        n = max(1, len(pixels))
        lums = []
        warmths = []
        sats = []
        whites = 0
        blacks = 0
        for r, g, b in pixels:
            rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
            lum = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf
            lums.append(lum)
            warmths.append(rf - bf)
            mx = max(rf, gf, bf)
            mn = min(rf, gf, bf)
            sats.append((mx - mn) / mx if mx > 0 else 0.0)
            if mn >= 254.0 / 255.0:
                whites += 1
            if mx <= 1.0 / 255.0:
                blacks += 1
        mean_lum = sum(lums) / n
        var_lum = sum((x - mean_lum) ** 2 for x in lums) / n
        return {
            'mean_luminance': float(mean_lum),
            'luminance_std': float(var_lum ** 0.5),
            'saturation': float(sum(sats) / n),
            'warmth': float(sum(warmths) / n),
            'white_clip_fraction': float(whites / n),
            'black_clip_fraction': float(blacks / n),
        }


def evaluate_cpu_guardrail(
    before_stats: dict[str, float],
    after_stats: dict[str, float],
) -> GuardrailResult:
    """Fast millisecond guardrail against highlight clipping, shadow crushing, or washed contrast."""
    white_before = float(before_stats.get('white_clip_fraction', 0.0))
    white_after = float(after_stats.get('white_clip_fraction', 0.0))
    black_before = float(before_stats.get('black_clip_fraction', 0.0))
    black_after = float(after_stats.get('black_clip_fraction', 0.0))
    std_before = float(before_stats.get('luminance_std', 0.0))
    std_after = float(after_stats.get('luminance_std', 0.0))

    white_delta = white_after - white_before
    black_delta = black_after - black_before
    issues: list[str] = []

    if white_delta > 0.08 or (white_before <= 0.08 and white_after > 0.18):
        issues.append('highlight_clipping')
    if black_delta > 0.08 or (black_before <= 0.10 and black_after > 0.22):
        issues.append('shadow_crushing')
    if std_before >= 0.06 and std_after < 0.025:
        issues.append('contrast_collapse')

    metrics = {
        'white_clip_delta': round(white_delta, 4),
        'black_clip_delta': round(black_delta, 4),
        'white_clip_fraction': round(white_after, 4),
        'black_clip_fraction': round(black_after, 4),
        'mean_luminance': round(float(after_stats.get('mean_luminance', 0.0)), 4),
        'luminance_std': round(std_after, 4),
    }

    if not issues:
        return GuardrailResult(
            passed=True,
            issues=[],
            severity='ok',
            summary='亮度与高光/暗部裁切在安全范围内',
            metrics=metrics,
        )

    issue_names = {
        'highlight_clipping': f'高光死白比例增加 {white_delta * 100:.1f}%',
        'shadow_crushing': f'暗部死黑比例增加 {black_delta * 100:.1f}%',
        'contrast_collapse': '画面反差过低（存在发灰风险）',
    }
    return GuardrailResult(
        passed=False,
        issues=issues,
        severity='warn',
        summary='；'.join(issue_names[i] for i in issues),
        metrics=metrics,
    )
