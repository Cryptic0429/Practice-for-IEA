"""CPU training of matched controllers on frozen input-only backbone features."""
import argparse
import copy
import json
import math
import time
import torch
import torch.nn.functional as F
from .common import ROOT, EXP, DATA, read, write, append, sha, stamp
from .feature_runtime import Controller, TOOLS_ORDER, STAT_KEYS, normalized_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', choices=['vlm', 'vlm-stats'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized(), 'Head training must not allocate a GPU'
    protocol = read(EXP/'feature-protocol.json')
    assert read(EXP/'controller-cpu-preflight.json')['status'] == 'passed'
    for f, digest in protocol['code_hashes'].items():
        assert sha(ROOT/f) == digest
    source = protocol['training_rows']
    assert sha(ROOT/source['path']) == source['sha256']
    rows = read(ROOT/source['path'])
    cases = {c['id']: c for c in read(ROOT/protocol['sources']['train']['path'])}
    assert len(rows) == 920 and len({(r['case_id'], r['language']) for r in rows}) == 740
    manifest = read(EXP/'features/train-manifest.json')
    assert manifest['source_sha256'] == protocol['sources']['train']['sha256']
    cached = {}
    for r in manifest['rows']:
        assert sha(ROOT/r['path']) == r['sha256']
        c = torch.load(ROOT/r['path'], map_location='cpu', weights_only=True)
        assert c['protocol_sha256'] == sha(EXP/'feature-protocol.json')
        cached[r['case_id'], r['language']] = c
    h = torch.stack([cached[r['case_id'], r['language']]['hidden'] for r in rows])
    s = torch.stack([cached[r['case_id'], r['language']]['stats'] for r in rows])
    x, mean, std = normalized_features(h, s, args.route)
    labels = torch.tensor([[r['parameters'].get(k, 0)/100 for k in TOOLS_ORDER] for r in rows])
    presence = labels.ne(0).float()
    positive = presence.sum(0)
    pos_weight = ((len(rows)-positive)/positive.clamp_min(1)).sqrt().clamp(1, 5)
    out = EXP/'runs'/f'controller-{args.route}-seed-20261109'
    out.mkdir(parents=True, exist_ok=False)
    settings = protocol['training']
    write(out/'settings.json', dict(route=args.route, **settings, training_records=len(rows),
          feature_shape=list(x.shape), trainable_parameters=sum(p.numel() for p in Controller(x.shape[1]).parameters()),
          tool_positive_counts=dict(zip(TOOLS_ORDER, positive.int().tolist())), positive_weights=pos_weight.tolist(),
          dataset_sha256=source['sha256'], feature_manifest_sha256=sha(EXP/'features/train-manifest.json'),
          protocol_sha256=sha(EXP/'feature-protocol.json'), implementation_sha256=sha(__file__), cpu_only=True))
    torch.manual_seed(settings['seed'])
    net = Controller(x.shape[1])
    initial = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    generator = torch.Generator().manual_seed(settings['seed'])
    started = time.monotonic()
    write(out/'status.json', dict(status='running', started_at=stamp(), cpu_only=True))
    for epoch in range(1, settings['epochs']+1):
        net.train()
        order = torch.randperm(len(rows), generator=generator)
        losses = []
        for offset in range(0, len(rows), settings['batch_size']):
            indices = order[offset:offset+settings['batch_size']]
            gate, value = net(x[indices])
            mask = presence[indices].bool()
            gating = F.binary_cross_entropy_with_logits(gate, presence[indices], pos_weight=pos_weight)
            regression = F.smooth_l1_loss(value[mask], labels[indices][mask], beta=.1) if mask.any() else value.sum()*0
            loss = gating+4*regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        append(out/'metrics.jsonl', dict(epoch=epoch, loss=sum(losses)/len(losses), diagnostic_only=True))
        if epoch in settings['checkpoints']:
            payload = dict(state_dict={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                           mean=mean, std=std, tools=list(TOOLS_ORDER), stat_keys=list(STAT_KEYS), route=args.route,
                           epoch=epoch, parent_sha256=protocol['parent_sha256'], dataset_sha256=source['sha256'],
                           protocol_sha256=sha(EXP/'feature-protocol.json'), runtime_sha256=sha(ROOT/'scripts/functional_v5/feature_runtime.py'))
            torch.save(payload, out/f'checkpoint-{epoch}.pt')
        write(out/'status.json', dict(status='running', epoch=epoch, cpu_only=True, at=stamp()))
    assert any(not torch.equal(initial[k], v) for k, v in net.state_dict().items())
    write(out/'status.json', dict(status='completed', epoch=settings['epochs'], cpu_only=True,
                                  cpu_wall_seconds=time.monotonic()-started, finished_at=stamp(), weights_changed=True,
                                  checkpoint_hashes={str(e): sha(out/f'checkpoint-{e}.pt') for e in settings['checkpoints']}))
    print(json.dumps(read(out/'status.json')))


if __name__ == '__main__':
    main()
