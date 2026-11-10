"""CPU diagnostic candidate: global negative value gain; positive R7 curve retained.
Frozen original editor.py and live tools stay unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from functional_v1.common import validate
from functional_v1.editor import edit as legacy_edit


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()


def remap_old_label(parameters):
    """Preserve the old *actual* contrast transform, not the old slider sign."""
    from functional_v1.common import TOOLS
    result = {k: v for k, v in parameters.items() if k in TOOLS}
    value = result.get('contrast', 0)
    if value:
        result['contrast'] = round(100 * math.log2(1.25 + .0075 * value), 4)
    return validate(result)


def bounded_gain(value, slider):
    # A smooth, bounded curve. Midgrey small-signal response approximates the
    # old linear exposure; equal slider values are NOT equal-strength labels.
    if slider < 0:
        return value * (2 ** (slider / 50))
    gain = 2 ** (slider / 50)
    return rational_gain(value, gain)


def rational_gain(value, gain):
    out = gain * value / (1 + (gain - 1) * value)
    # Exact endpoints also under float32 rounding at negative slider values.
    return np.where(value <= 0, 0, np.where(value >= 1, 1, out))


def tonal_value(value, parameters, variant):
    if variant not in ('bounded-fixed', 'bounded-adaptive'):
        raise ValueError('unknown_editor_variant')
    out = value.copy()
    if parameters.get('exposure', 0):
        out = bounded_gain(out, parameters['exposure'])
    if parameters.get('brightness', 0):
        # Independent brightness gain: stronger midpoint response, no addition
        # of grey to exact black. This deliberately changes legacy semantics.
        gain = 2 ** (parameters['brightness'] / 35)
        out = out * (2 ** (parameters["brightness"] / 35)) if parameters["brightness"] < 0 else rational_gain(out, gain)
    if parameters.get('contrast', 0):
        pivot = .5 if variant == 'bounded-fixed' else float(np.clip(np.median(out), .1, .9))
        pivot_logit = math.log(pivot / (1 - pivot))
        gain = 2 ** (parameters['contrast'] / 100)
        interior = np.clip(out, 1e-7, 1-1e-7)
        z = pivot_logit + gain * (np.log(interior / (1-interior)) - pivot_logit)
        out = np.where(out <= 0, 0, np.where(out >= 1, 1, 1 / (1+np.exp(-z))))
    return out


def soft_contrast(image, slider):
    if not slider:
        return image.copy()
    gain = 2 ** (slider / 100)
    z = .5 + gain * (image - .5)
    if slider < 0:
        # Contrast reduction already lies inside [0,1]; preserve its exact
        # historical linear response, including deliberately lifted black.
        return z
    # Stable subtraction at the shoulder; float32 cancellation can otherwise
    # introduce a one-ulp backwards step in an analytically monotone curve.
    z = .5 + gain * (np.asarray(image,dtype=np.float64) - .5)
    softness = .08 * (gain - 1)
    def shoulder(x):
        return softness * (np.logaddexp(0, x/softness) - np.logaddexp(0, (x-1)/softness))
    low = shoulder(.5-.5*gain)
    high = shoulder(.5+.5*gain)
    out = (shoulder(z)-low)/(high-low)
    # Analytic source endpoints are zero/one; maintain them under float32.
    out = np.where(image <= 0, 0, np.where(image >= 1, 1, out))
    return np.asarray(np.clip(out,0,1),dtype=np.float32)


def edit(rgb, parameters, variant='bounded-adaptive'):
    p = validate(parameters)
    if variant not in ('bounded-fixed', 'bounded-adaptive'):
        raise ValueError('unknown_editor_variant')
    if not any(p.values()):
        return rgb.copy()
    if not any(p.get(k, 0) for k in ('exposure', 'brightness', 'contrast')):
        return legacy_edit(rgb, p)
    image = rgb.astype(np.float32) / 255
    value = image.max(axis=-1)
    target = tonal_value(value, {k:v for k,v in p.items() if k != 'contrast'}, variant)
    image *= np.divide(target, value, out=np.ones_like(value), where=value > 0)[..., None]
    if p.get('contrast', 0):
        image = soft_contrast(image, p['contrast'])
    for name in ('highlights', 'shadows'):
        amount = p.get(name, 0) * (.004 if name == 'highlights' else .008)
        if not amount:
            continue
        # Luminance masks apply the same gain to RGB, preserving hue better than
        # three separate per-channel shadow curves. Endpoints remain bounded.
        lum = image @ np.array([.2126, .7152, .0722], np.float32)
        weight = lum ** 2 if name == 'highlights' else (1 - lum) ** 2
        if name == 'highlights':
            # Lower bright values without globally darkening already-shadowed
            # faces. Unlike gamma anchored at white, this can soften white peaks.
            if amount < 0:
                image *= 1 + amount * weight[..., None]
            else:
                image += amount * weight[..., None] * (1-image)
        else:
            # Preserve true black under a shadow lift; no additive gray veil.
            gamma = 2 ** (-amount)
            adjusted = np.power(image, gamma)
            image = image * (1-weight[..., None]) + adjusted * weight[..., None]
    if p.get('saturation', 0):
        hsv = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + p['saturation'] / 100), 0, 1)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    temperature = p.get('temperature', 0) / 510
    image += np.asarray([temperature, .6 * temperature, -temperature], np.float32)
    image[..., 1] += p.get('tint', 0) * 30 / 25500
    if p.get('sepia', 0):
        image = np.clip(image, 0, 1)
        lum = image @ np.array([.2126, .7152, .0722], np.float32)
        tone = image @ np.array([[.393, .349, .272],
                                 [.769, .686, .534],
                                 [.189, .168, .131]], np.float32)
        tone_lum = tone @ np.array([.2126, .7152, .0722], np.float32)
        tone *= (lum / np.maximum(tone_lum, 1e-6))[..., None]
        # Neutral highlights preserve cloud/white detail under the global style.
        white = np.clip((lum - .7) / .3, 0, 1) ** 2
        tone = tone * (1-white[..., None]) + lum[..., None] * white[..., None]
        strength = p['sepia'] / 100
        image = (1-strength) * image + strength * tone
    return np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)


def save(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format='PNG', compress_level=1)


def statistics(rgb):
    value = rgb.astype(np.float32) / 255
    lum = value @ np.array([.2126, .7152, .0722], np.float32)
    hsv = cv2.cvtColor(value, cv2.COLOR_RGB2HSV)
    return {'mean_luminance': float(lum.mean()), 'luminance_std': float(lum.std()),
            'saturation': float(hsv[..., 1].mean()),
            'warmth': float((value[..., 0] - value[..., 2]).mean()),
            'white_clip_fraction': float((value.min(2) >= 254/255).mean()),
            'black_clip_fraction': float((value.max(2) <= 1/255).mean())}
