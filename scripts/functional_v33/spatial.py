"""Measure where a contrast edit changes already-visible detail."""
import cv2
import numpy as np

from functional_v31.face import FaceRegions, interior
from functional_v32.editor import LUMA

SPATIAL = ('worst_shadow_texture_loss', 'worst_highlight_texture_loss', 'new_black_worst_tile',
           'new_white_worst_tile', 'detected_face_present', 'face_dark_tail_loss',
           'face_highlight_tail_increase', 'face_texture_loss')


def luma(rgb):
    return (rgb.astype(np.float32) @ LUMA) / 255


def texture(lum):
    return np.abs(lum - cv2.GaussianBlur(lum, (7, 7), 0))


def tile_mean(array):
    return cv2.resize(array.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)


class SourceFeatures:
    def __init__(self, rgb, faces: FaceRegions):
        h, w = rgb.shape[:2]
        scale = min(1, 960 / max(h, w))
        self.size = (max(1, round(w * scale)), max(1, round(h * scale)))
        before = self.resize(rgb)
        y = luma(before)
        self.mean = tile_mean(y)
        self.detail = tile_mean(texture(y))
        self.black = before.max(axis=2) <= 2
        self.white = before.min(axis=2) >= 253
        self.boxes = faces.detect(rgb)
        self.face_before = [interior(rgb, box) for box in self.boxes]

    def resize(self, rgb):
        return cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)

    def compare(self, edited):
        after = self.resize(edited)
        detail = tile_mean(texture(luma(after)))
        useful = self.detail > .003
        loss = (self.detail - detail) / np.maximum(self.detail, .003)
        shadows = loss[useful & (self.mean < .30)]
        highlights = loss[useful & (self.mean > .70)]
        faces = [interior(edited, box) for box in self.boxes]
        low, high, face_detail = [], [], []
        for old, new in zip(self.face_before, faces, strict=True):
            b = np.quantile(old, [.10, .90])
            a = np.quantile(new, [.10, .90])
            low.append(b[0] - a[0])
            high.append(a[1] - b[1])
            face_detail.append(float(texture(old).mean() - texture(new).mean()))
        return [float(value) for value in (
            np.quantile(shadows, .90) if shadows.size else 0,
            np.quantile(highlights, .90) if highlights.size else 0,
            tile_mean(~self.black & (after.max(axis=2) <= 2)).max(),
            tile_mean(~self.white & (after.min(axis=2) >= 253)).max(),
            bool(faces), max(low, default=0), max(high, default=0), max(face_detail, default=0),
        )]
