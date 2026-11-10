"""Matched native-pixel features for the shadow-artifact feasibility experiment."""
import numpy as np

from functional_v15.editor import load_rgb

_WEIGHTS = np.array([.2126, .7152, .0722], dtype=np.float32)
_TONAL_TOOLS = ('exposure', 'brightness', 'shadows')
INPUT_NAMES = ('shadow_luma_median', 'shadow_luma_p90', 'shadow_patch_fraction',
               'input_fine_chroma_p50', 'input_fine_chroma_p90', 'input_coarse_chroma_p90', 'input_fine_luma_p90')
PAIR_NAMES = INPUT_NAMES + ('output_fine_chroma_p50', 'output_fine_chroma_p90', 'output_coarse_chroma_p90',
                            'output_fine_luma_p90', 'fine_chroma_gain_p90', 'coarse_chroma_gain_p90',
                            'fine_luma_gain_p90', 'fine_chroma_ratio_p90', 'new_neighbour_jump_p90')


def positive_tonal_edit(decision):
    return decision.get('action') == 'edit' and any(decision['parameters'].get(k, 0) > 0 for k in _TONAL_TOOLS)


def tiles(image, side):
    h, w = image.shape[:2]
    ys = np.linspace(0, h-side, min(12, h//side), dtype=int)
    xs = np.linspace(0, w-side, min(12, w//side), dtype=int)
    return np.stack([image[y:y+side, x:x+side] for y in ys for x in xs]).astype(np.float32) / 255


def second_difference(x):
    """Mean absolute change of slope in the vertical and horizontal directions."""
    horizontal = x[:, :, :-2] - 2*x[:, :, 1:-1] + x[:, :, 2:]
    vertical = x[:, :-2, :] - 2*x[:, 1:-1, :] + x[:, 2:, :]
    axes = tuple(range(1, x.ndim))
    return (np.abs(horizontal).mean(axis=axes) + np.abs(vertical).mean(axis=axes)) / 2


def residuals(rgb, luma, side):
    chroma = rgb[..., (0, 2)] - rgb[..., 1:2]
    stride = side // 8
    coarse = chroma.reshape(-1, 8, stride, 8, stride, 2).mean(axis=(2, 4))
    return second_difference(chroma), second_difference(coarse), second_difference(luma)


def describe(values):
    fine_chroma, coarse_chroma, fine_luma = values
    return [np.median(fine_chroma), np.quantile(fine_chroma, .9),
            np.quantile(coarse_chroma, .9), np.quantile(fine_luma, .9)]


def features(input_path, output_path):
    src, dst = load_rgb(input_path), load_rgb(output_path)
    if src.shape != dst.shape:
        raise ValueError('Expected aligned input/output pixels')
    side = 8 * (min(64, *src.shape[:2]) // 8)
    if side < 8:
        raise ValueError('Input too small for a native-pixel tile')
    src_tiles, dst_tiles = tiles(src, side), tiles(dst, side)
    source_luma = src_tiles @ _WEIGHTS
    levels = np.median(source_luma, axis=(1, 2))
    is_dark = levels < .38
    selected = np.flatnonzero(is_dark)
    if len(selected) < min(8, len(levels)):
        selected = np.argsort(levels, kind='stable')[:8]
    src_tiles, dst_tiles, source_luma = src_tiles[selected], dst_tiles[selected], source_luma[selected]
    output_luma = dst_tiles @ _WEIGHTS
    before = residuals(src_tiles, source_luma, side)
    after = residuals(dst_tiles, output_luma, side)
    inp = [np.median(levels[selected]), np.quantile(levels[selected], .9), np.mean(is_dark), *describe(before)]
    paired = describe(after) + [np.quantile(b-a, .9) for a,b in zip(before, after)]
    paired += [np.quantile((after[0]+1/255)/(before[0]+1/255), .9)]
    original_step = np.abs(np.diff(source_luma, axis=2))
    output_step = np.abs(np.diff(output_luma, axis=2))
    new_jumps = ((original_step <= 2/255) & (output_step > 4/255)).mean(axis=(1, 2))
    paired += [np.quantile(new_jumps, .9)]
    return np.asarray(inp + paired, dtype=np.float64)
