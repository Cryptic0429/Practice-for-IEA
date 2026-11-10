"""Assistant-only CE plus explicit visual hard-negative decision ranking."""
import torch
import torch.nn.functional as F


def sequence_logp(logits, labels):
    targets=labels[:,1:]
    mask=targets.ne(-100)
    scores=logits[:,:-1][mask].float()
    assert int(mask.sum())>0
    return -F.cross_entropy(scores,targets[mask],reduction='sum'),int(mask.sum())


def ranking_loss(chosen, rejected, chosen_count, rejected_count, weight=.1, margin=1., scale=32.):
    assert chosen_count>0 and rejected_count>0
    # The numeric strings can have different token lengths (e.g.40 vs100).
    # Normalize each assistant sum before the fixed, predeclared score scale.
    # Both decisions are scored under the SAME original photo and literal request.
    return weight*F.softplus(margin+scale*(rejected/rejected_count-chosen/chosen_count))
