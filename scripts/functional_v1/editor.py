"""Versioned editor; never monkey-patches the historical upstream editor."""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .common import validate


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()


def remap_old_label(parameters):
    """Preserve the old *actual* contrast transform, not the old slider sign."""
    from .common import TOOLS
    result = {k: v for k, v in parameters.items() if k in TOOLS}
    value = result.get('contrast', 0)
    if value:
        result['contrast'] = round(100 * math.log2(1.25 + .0075 * value), 4)
    return validate(result)


def edit(rgb, parameters):
    p = validate(parameters)
    if not any(p.values()):
        return rgb.copy()
    image = rgb.astype(np.float32) / 255
    image *= 2 ** (p.get('exposure', 0) / 100)
    image += p.get('brightness', 0) / 200
    image = .5 + 2 ** (p.get('contrast', 0) / 100) * (image - .5)
    image = np.clip(image, 0, 1)
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
