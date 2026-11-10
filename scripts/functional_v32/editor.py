"""CPU prototype for endpoint-protected, hue-preserving global contrast."""
import numpy as np

from functional_v15.editor import edit

RECIPES = ('r15_20', 'r15_40', 'protected_soft', 'protected_strong')
LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)


def protected_curve(luma, amount):
    low, high = .07, .96
    t = np.clip((luma - low) / (high - low), 0, 1)
    offset = 4 * t**2 * (1 - t)**2 * (2 * t - 1)
    return luma + (high - low) * amount * offset


def apply(rgb, direction, recipe):
    sign = {'increase': 1, 'decrease': -1}[direction]
    if recipe in ('r15_20', 'r15_40'):
        return edit(rgb, {'contrast': sign * int(recipe.split('_')[1])})
    amount = {'protected_soft': .55, 'protected_strong': 1.10}[recipe]
    image = rgb.astype(np.float32) / 255
    luma = image @ LUMA
    target = protected_curve(luma, sign * amount)
    # Move along the neutral axis. Keep colour differences whenever they fit;
    # otherwise reduce them uniformly to stay inside the RGB gamut.
    chroma = image - luma[..., None]
    positive = chroma.max(axis=2)
    negative = -chroma.min(axis=2)
    headroom = np.divide(1 - target, positive, out=np.ones_like(target), where=positive > 0)
    footroom = np.divide(target, negative, out=np.ones_like(target), where=negative > 0)
    scale = np.minimum(1, np.minimum(headroom, footroom))
    result = target[..., None] + scale[..., None] * chroma
    return np.clip(np.rint(result * 255), 0, 255).astype(np.uint8)
