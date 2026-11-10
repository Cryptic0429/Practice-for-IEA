"""CPU paired-language training; only TRAIN features fit normalization or weights."""
import argparse
import math
import time
import torch
from .common import ROOT, EXP, ROUTES, PARENT_SHA, TOOLS, protocol, read, write, sha, append, stamp
from .controller import ACTIONS, Controller, labels, normalize, loss_function

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', choices=ROUTES, required=True)
    ap.add_argument('--seed', type=int, choices=[20261203, 20261204], default=20261203)
    args = ap.parse_args()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    p = protocol()
    source = p['feature_sources']['train']
    assert sha(ROOT / source['path']) == source['sha256']
    cases = read(ROOT / source['path'])
    assert len(cases) == 925 and len({c['group_id'] for c in cases}) == 240
    manifest = read(EXP / 'features/train-manifest.json')
    assert manifest['source_sha256'] == source['sha256']
    rows = manifest['rows']
    assert [(r['case_id'], r['language']) for r in rows] == [(c['id'], l) for c in cases for l in ('zh', 'en')]
    cached = []
    for r in rows:
        assert sha(ROOT / r['path']) == r['sha256']
        feature = torch.load(ROOT / r['path'], map_location='cpu', weights_only=True)
        assert feature['protocol_sha256'] == sha(EXP / 'training-protocol.json')
        cached.append(feature)
    x, mean, std = normalize(torch.stack([c[args.route] for c in cached]), torch.stack([c['stats'] for c in cached]))
    action, direction, magnitude = labels([c for c in cases for _ in range(2)])
    counts = torch.stack([(direction[action.eq(0)] == k).sum(0) for k in range(3)], -1)
    direction_weights = (action.eq(0).sum() / counts.clamp_min(1)).sqrt().clamp(1, 5)
    torch.manual_seed(args.seed)
    net = Controller()
    initial = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator().manual_seed(args.seed)
    out = EXP / 'runs' / (args.route if args.seed == 20261203 else f'{args.route}-seed-{args.seed}')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'settings.json', dict(p['training'], seed=args.seed, route=args.route, cpu_only=True,
          semantic_cases=len(cases), language_inputs=len(rows), TRAIN_only_normalization=True,
          trainable_parameters=sum(v.numel() for v in net.parameters()), direction_counts=counts.tolist(),
          direction_weights=direction_weights.tolist(), feature_manifest_sha256=sha(EXP / 'features/train-manifest.json')))
    steps = 0
    started = time.monotonic()
    for epoch in range(1, 81):
        net.train()
        order = torch.randperm(len(cases), generator=generator)
        losses = []
        for start in range(0, len(cases), 16):
            semantic = order[start:start+16]
            indices = torch.stack([2*semantic, 2*semantic+1], -1).flatten()
            loss = loss_function(net(x[indices]), action[indices], direction[indices], magnitude[indices], direction_weights)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step(); steps += 1
            losses.append(float(loss.detach()))
        append(out / 'metrics.jsonl', dict(epoch=epoch, step=steps, loss=sum(losses)/len(losses), diagnostic_only=True))
        if epoch in (20, 80):
            payload = dict(state_dict={k:v.detach().clone() for k,v in net.state_dict().items()},
                           mean=mean, std=std, tools=list(TOOLS), actions=list(ACTIONS), route=args.route,
                           epoch=epoch, step=steps, parent_sha256=PARENT_SHA,
                           protocol_sha256=sha(EXP / 'training-protocol.json'), dataset_sha256=source['sha256'])
            torch.save(payload, out / f'checkpoint-{epoch}.pt')
            torch.save(dict(optimizer=optimizer.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state()),
                       out / f'training-state-{epoch}.pt')
        write(out / 'status.json', dict(status='running', epoch=epoch, step=steps, cpu_only=True, at=stamp()))
    assert all(not torch.equal(initial[k], v) for k,v in net.state_dict().items())
    assert not torch.cuda.is_initialized()
    write(out / 'status.json', dict(status='completed', epoch=80, step=steps, cpu_only=True, all_weights_changed=True,
          wall_seconds=time.monotonic()-started, checkpoint_sha256={str(e):sha(out/f'checkpoint-{e}.pt') for e in (20,80)}, at=stamp()))
    print(read(out / 'status.json'))

if __name__ == '__main__': main()
