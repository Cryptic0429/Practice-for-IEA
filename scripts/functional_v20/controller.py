"""Learn actions, tool direction and unsigned strength separately; no heuristics."""
import torch
from torch import nn
from functional_v1.common import TOOLS
from functional_v15.outcomes import validate_decision, target_text

ACTIONS = ('edit', 'keep', 'unsupported_tools', 'unreliable_quality')

class Controller(nn.Module):
    def __init__(self, dimensions=3600):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, 128), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(128, 4 + 3 * len(TOOLS) + len(TOOLS)))

    def forward(self, x):
        value = self.net(x)
        actions = value[..., :4]
        directions = value[..., 4:31].reshape(*value.shape[:-1], len(TOOLS), 3)
        # Sepia has no legal negative direction. This rule is also in training.
        mask = torch.zeros_like(directions)
        mask[..., -1, 1] = -10000
        return actions, directions + mask, value[..., 31:].sigmoid()

def labels(cases):
    actions, directions, magnitudes = [], [], []
    for case in cases:
        decision = validate_decision(case['decision'])
        actions.append(ACTIONS.index(decision['action'] if decision['action'] != 'decline' else decision['reason']))
        params = decision.get('parameters', {})
        directions.append([0 if not params.get(k, 0) else 1 if params[k] < 0 else 2 for k in TOOLS])
        magnitudes.append([abs(params.get(k, 0)) / 100 for k in TOOLS])
    return torch.tensor(actions), torch.tensor(directions), torch.tensor(magnitudes, dtype=torch.float32)

def normalize(hidden, stats, mean=None, std=None):
    assert hidden.shape[-1] == 3584 and stats.shape[-1] == 16
    hidden = hidden / hidden.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    x = torch.cat([hidden, stats], -1)
    if mean is None:
        mean = x.mean(0)
        std = x.std(0, unbiased=False).clamp_min(.1)
    assert mean.shape == std.shape == (3600,)
    return (x - mean) / std, mean, std

def decode(actions, directions, magnitudes):
    assert actions.shape == (4,) and directions.shape == (9, 3) and magnitudes.shape == (9,)
    assert all(torch.isfinite(t).all() for t in (actions, directions, magnitudes))
    action = ACTIONS[int(actions.argmax())]
    if action == 'keep': return {'action': 'keep'}
    if action != 'edit': return {'action': 'decline', 'reason': action}
    parameters = {}
    for index, name in enumerate(TOOLS):
        direction = int(directions[index].argmax())
        if not direction: continue
        if name == 'sepia' and direction == 1: raise ValueError('illegal_negative_sepia')
        magnitude = int(round(float(magnitudes[index]) * 100))
        if magnitude: parameters[name] = -magnitude if direction == 1 else magnitude
    # A predicted edit with no nonzero tool is an error, never an unchanged success.
    return validate_decision({'action': 'edit', 'parameters': parameters})

def js_pair(logits):
    assert logits.shape[0] % 2 == 0
    logp = logits.log_softmax(-1).reshape(-1, 2, *logits.shape[1:])
    p = logp.exp()
    mean = p.mean(1, keepdim=True).clamp_min(1e-8)
    return (p * (logp - mean.log())).sum(-1).mean()

def loss_function(outputs, action, direction, magnitude, direction_weights):
    import torch.nn.functional as F
    a, d, m = outputs
    edit = action.eq(0)
    active = direction.ne(0) & edit[:, None]
    action_loss = F.cross_entropy(a, action)
    if edit.any():
        logp = d[edit].log_softmax(-1)
        target = direction[edit]
        weights = direction_weights[torch.arange(9)[None], target]
        direction_loss = -(logp.gather(-1, target[..., None]).squeeze(-1) * weights).sum() / weights.sum()
        consistency = js_pair(d.reshape(-1, 27).reshape(-1, 9, 3))
    else:
        direction_loss = d.sum() * 0
        consistency = d.sum() * 0
    numerical = F.smooth_l1_loss(m[active], magnitude[active], beta=.1) if active.any() else m.sum() * 0
    paired = m.reshape(-1, 2, 9)
    paired_active = active.reshape(-1, 2, 9).any(1)
    magnitude_consistency = (paired[:, 0] - paired[:, 1]).square()[paired_active].mean() if paired_active.any() else m.sum() * 0
    return action_loss + direction_loss + 4 * numerical + .25 * (js_pair(a) + consistency + magnitude_consistency)

def load_controller(path):
    item = torch.load(path, map_location='cpu', weights_only=True)
    assert item['tools'] == list(TOOLS) and item['actions'] == list(ACTIONS)
    net = Controller()
    net.load_state_dict(item['state_dict'], strict=True)
    net.eval()
    return net, item

def predict_cached(net, item, cached):
    x, _, _ = normalize(cached[item['route']][None], cached['stats'][None], item['mean'], item['std'])
    with torch.inference_mode(): outputs = net(x)
    decision = decode(*(part[0] for part in outputs))
    return target_text(decision), decision
