"""Assistant-only weighted CE; numerical token masks come from verified character offsets."""
import re
import json
import torch
import torch.nn.functional as F
from functional_v1.common import validate
from functional_v1.runtime import build_batch


def build_numeric_batch(processor, case, language, parameters):
    target = '```json\n'+json.dumps(validate(parameters), indent=2)+'\n```'
    batch = build_batch(processor, case, language, parameters)
    positions = batch['labels'][0].ne(-100).nonzero().flatten()
    supervised = batch['labels'][0, positions].tolist()
    # Chat adds an end-of-turn marker after the exact assistant string.
    encoded = processor.tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    assert supervised[:len(encoded['input_ids'])] == encoded['input_ids'], 'Target/token offsets diverged'
    spans = [(m.start(), m.end()) for m in re.finditer(r'-?\d+(?:\.\d+)?', target)]
    mask = torch.zeros_like(batch['labels'], dtype=torch.bool)
    for pos, (a, b) in zip(positions.tolist(), encoded['offset_mapping']):
        if any(a < y and b > x for x, y in spans):
            mask[0, pos] = True
    assert not mask[batch['labels'].eq(-100)].any()
    if parameters:
        assert mask.any(), 'Numerical JSON lost its numerical supervision'
    else:
        assert not mask.any()
    batch['numeric_mask'] = mask
    return batch


def token_losses(logits, labels):
    shifted = labels[:, 1:]
    active = shifted.ne(-100)
    scores = logits[:, :-1, :][active].float()
    return F.cross_entropy(scores, shifted[active], reduction='none'), active, scores, shifted[active]


def weighted_ce(logits, labels, numeric_mask, numerical_weight=5.):
    if numerical_weight < 1:
        raise ValueError('Numerical supervision weight must be at least one')
    losses, active, _, _ = token_losses(logits, labels)
    numeric = numeric_mask[:, 1:][active]
    weights = torch.where(numeric, numerical_weight, 1.).to(losses.dtype)
    return (losses*weights).sum()/weights.sum()
