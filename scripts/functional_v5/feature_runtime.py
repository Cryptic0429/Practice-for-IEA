"""Frozen Qwen features plus an independently trained continuous editing controller."""
import json
import numpy as np
import torch
from torch import nn
from functional_v1.common import TOOLS, validate
from functional_v1.runtime import image_input, processor, load_model
from functional_v4.runtime import encode, measurements

VERSION = 'functional-v5-frozen-vlm-continuous-controller-v1'
TOOLS_ORDER = tuple(TOOLS)
STAT_KEYS = tuple(sorted(k for k in measurements(__import__('PIL.Image', fromlist=['Image']).new('RGB', (28, 28))) if k != 'scale'))


def extract_features(model, p, case, language):
    # Both controller routes see the same original image/request and round-four plain prompt.
    encoded = encode(p, case, language, 'plain')
    moved = {k: v.to('cuda') for k, v in encoded.items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model.get_base_model().model(**moved, use_cache=False, return_dict=True)
    last = result.last_hidden_state[:, -1, :].detach().float().cpu().squeeze(0)
    assert last.ndim == 1 and torch.isfinite(last).all()
    im = image_input(__import__('functional_v1.common', fromlist=['image_path']).image_path(case))
    observed = measurements(im)
    stats = torch.tensor([observed[k]/1000. for k in STAT_KEYS], dtype=torch.float32)
    return last, stats


class Controller(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, 128), nn.GELU(), nn.Dropout(.15),
                                 nn.Linear(128, 2*len(TOOLS_ORDER)))

    def forward(self, x):
        gates, values = self.net(x).split(len(TOOLS_ORDER), dim=-1)
        return gates, values.tanh()


def normalized_features(hidden, stats, route, mean=None, std=None):
    if route not in ('vlm', 'vlm-stats'):
        raise ValueError('Unknown controller feature route')
    # Per-input RMS normalization has no fitted dataset statistics.
    h = hidden/(hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6))
    x = torch.cat([h, stats], dim=-1) if route == 'vlm-stats' else h
    if mean is None:
        mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(.1)
    assert mean.shape == std.shape == x.shape[-1:]
    return (x-mean)/std, mean, std


def decode(gates, values):
    assert gates.shape == values.shape == (len(TOOLS_ORDER),)
    assert torch.isfinite(gates).all() and torch.isfinite(values).all()
    result = {}
    for i, name in enumerate(TOOLS_ORDER):
        if float(gates[i].sigmoid()) < .5:
            continue
        value = int(round(float(values[i])*100))
        if name == 'sepia':
            value = max(0, value)
        if value:
            result[name] = value
    return validate(result)


def load_controller(path):
    item = torch.load(path, map_location='cpu', weights_only=True)
    assert tuple(item['tools']) == TOOLS_ORDER and tuple(item['stat_keys']) == STAT_KEYS
    model = Controller(item['mean'].numel())
    model.load_state_dict(item['state_dict'])
    model.eval()
    return model, item


def predict_controller(backbone, p, controller, item, case, language):
    hidden, stats = extract_features(backbone, p, case, language)
    x, _, _ = normalized_features(hidden[None], stats[None], item['route'], item['mean'], item['std'])
    with torch.inference_mode():
        gate, value = controller(x)
    parameters = decode(gate[0], value[0])
    return '```json\n'+json.dumps(parameters, indent=2)+'\n```', parameters
