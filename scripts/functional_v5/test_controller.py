"""Independent numerical-mask, gating, normalization and gradient checks."""
import torch
import torch.nn.functional as F
from .objective import weighted_ce, build_numeric_batch
from .feature_runtime import Controller, TOOLS_ORDER, decode, normalized_features
from .common import ROOT, EXP, read, write, stamp


def main():
    torch.manual_seed(42)
    scores = torch.randn(1, 6, 17, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, 4, 5]])
    numeric = torch.tensor([[False, False, False, True, True, False]])
    actual = weighted_ce(scores, labels, numeric, numerical_weight=1.)
    expected = F.cross_entropy(scores[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert scores.grad[:, 0].eq(0).all() and scores.grad[:, -1].eq(0).all()
    assert scores.grad[:, 1:5].abs().sum() > 0
    gates = torch.full((len(TOOLS_ORDER),), -20.)
    values = torch.full_like(gates, .7)
    assert decode(gates, values) == {}
    for i, key in enumerate(TOOLS_ORDER):
        g = gates.clone(); g[i] = 20
        v = values.clone(); v[i] = -.234
        assert decode(g, v) == ({} if key == 'sepia' else {key: -23})
        v[i] = .234
        assert decode(g, v) == {key: 23}
    h, s = torch.randn(40, 16), torch.rand(40, 15)
    train, mean, std = normalized_features(h[:30], s[:30], 'vlm-stats')
    old_mean, old_std = mean.clone(), std.clone()
    held, _, _ = normalized_features(100*h[30:], 100*s[30:], 'vlm-stats', mean, std)
    assert torch.equal(mean, old_mean) and torch.equal(std, old_std)
    net = Controller(train.shape[1]); gate, value = net(train)
    loss = gate.square().mean()+value.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    # Real processor token-offset checks; no model/GPU required.
    from functional_v1.runtime import processor
    p = processor()
    case = read(ROOT/'data/training/functional-v4-20261105/cases.json')[0]
    for language in ('zh', 'en'):
        for params in ({}, {'exposure': -23, 'saturation': 100}, {'temperature': 12.5}):
            b = build_numeric_batch(p, case, language, params)
            text = p.tokenizer.decode(b['input_ids'][b['numeric_mask']].tolist())
            assert (not text) if not params else ('23' in text if params.get('exposure') else '12' in text)
    write(EXP/'controller-cpu-preflight.json', dict(status='passed', at=stamp(),
        checks=['weighted CE unitweight independent parity and causal/prefix mask gradient', 'all9 tool gates and signed/sepia decode',
                'train-only fitted normalization', 'head gradients finite', 'real bilingual assistant offsets and numeric spans'],
        no_gpu=True))
    print('controller CPU preflight passed')


if __name__ == '__main__':
    main()
