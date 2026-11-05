"""Strict output validation and official-editor scoring shared by all models."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'duplicate_json_key:{key}')
        out[key] = value
    return out


def _constant(value):
    raise ValueError(f'invalid_json_constant:{value}')


def parse_parameters(response, protocol):
    result = dict(json_valid=False, tool_names_valid=False, values_valid=False,
                  valid=False, parameters=None, error=None)
    match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    if not match:
        result['error'] = 'missing_json_fence'
        return result
    try:
        obj = json.loads(match.group(1).strip(), object_pairs_hook=_pairs, parse_constant=_constant)
        if not isinstance(obj, dict):
            raise ValueError('json_root_not_object')
    except (ValueError, TypeError) as exc:
        result['error'] = f'json_error:{exc}'
        return result
    result['json_valid'] = True
    rules = protocol['parser']
    normalized = {}
    for key, value in obj.items():
        name = rules['aliases'].get(key, key)
        if name not in rules['allowed_tools']:
            result['error'] = f'unknown_tool:{key}'
            return result
        if name in normalized:
            result['error'] = f'duplicate_normalized_tool:{name}'
            return result
        normalized[name] = value
    result['tool_names_valid'] = True
    for key, value in normalized.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            result['error'] = f'nonnumeric_value:{key}'
            return result
        if not math.isfinite(value) or not rules['value_min'] <= value <= rules['value_max']:
            result['error'] = f'invalid_numeric_range:{key}'
            return result
    result.update(values_valid=True, valid=True, parameters=normalized)
    return result


def sample_seed(sample_id, base_seed):
    return (base_seed + int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)) % (2**32)


class Scorer:
    def __init__(self, root, protocol):
        sys.path.insert(0, str(Path(root) / 'repo'))
        import cv2
        import numpy as np
        from src import config
        from src.image_editor import ImageEditor
        from src.utils import compute_diff_score, load_and_resize_image
        assert config.IMAGE_RESIZE == protocol['editor']['image_resize']
        assert set(config.NUMERICAL_OPERATORS) == set(protocol['parser']['allowed_tools'])
        assert config.REPLACE_TABLE == protocol['parser']['aliases']
        self.cv2, self.np, self.editor = cv2, np, ImageEditor
        self.diff, self.load, self.protocol = compute_diff_score, load_and_resize_image, protocol

    def score(self, row, parsed, output_path):
        result = dict(editor_success=False, input_readable=False, reference_readable=False,
                      fallback_used=True, metric_error=None, editor_error=None)
        try:
            original = self.load(row['extra_info']['origin_image_path'])
            result['input_readable'] = True
            reference = self.load(row['extra_info']['reference_image_path'])
            result['reference_readable'] = True
            result['baseline_metrics'] = self.diff(original, reference)
        except Exception as exc:
            result['metric_error'] = f'{type(exc).__name__}:{exc}'
            return result
        edited = original
        if parsed['valid']:
            try:
                self.np.random.seed(sample_seed(row['id'], self.protocol['seed']))
                edited = self.editor(row['extra_info']['origin_image_path']).process_image(parsed['parameters'])
                assert edited.shape == original.shape and self.np.isfinite(edited).all()
                result.update(editor_success=True, fallback_used=False)
            except Exception as exc:
                result['editor_error'] = f'{type(exc).__name__}:{exc}'
                edited = original
        result['edited_metrics'] = self.diff(edited, reference)
        result['l1_improvement'] = result['baseline_metrics']['pixel_L1'] - result['edited_metrics']['pixel_L1']
        result['score_improvement'] = result['baseline_metrics']['final_score'] - result['edited_metrics']['final_score']
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            assert self.cv2.imwrite(str(output_path), edited), output_path
            result['edited_image_path'] = str(output_path)
        if parsed['valid']:
            expected = row['extra_info']['label']
            pred_set = {key for key, val in parsed['parameters'].items() if val != 0}
            true_set = {key for key, val in expected.items() if val != 0}
            result['parameter_metrics'] = dict(
                nonzero_true_positive=len(pred_set & true_set), nonzero_predicted=len(pred_set),
                nonzero_target=len(true_set),
                mean_absolute_error_16=sum(abs(parsed['parameters'].get(key, 0) - expected.get(key, 0))
                                           for key in self.protocol['parser']['allowed_tools']) / 16)
        return result
