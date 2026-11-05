"""Cache only allowed image/request features; never infer on the sealed acceptance set."""
import argparse
import hashlib
import io
from .common import ROOT, EXP, DATA, PARENT, GPUJob, read, write, append, sha, stamp


def prepare():
    if (EXP/'feature-protocol.json').exists():
        raise RuntimeError('Feature inputs already frozen')
    diagnostic = read(EXP/'loss-diagnostic-summary.json')
    assert diagnostic['status'] == 'completed_diagnostic_not_functional_evaluation'
    train = ROOT/'data/training/functional-v4-20261105'
    held = ROOT/'data/training/functional-v2-20261104'
    sources = {'train': train/'cases.json', 'development': held/'development-final-cases.json',
               'natural-use': held/'natural-use-final-cases.json'}
    groups = {s: {c['group_id'] for c in read(p)} for s, p in sources.items()}
    assert not groups['train'] & (groups['development'] | groups['natural-use'])
    assert not groups['train'] & {c['group_id'] for c in read(held/'acceptance-final-cases.json')}
    protocol = dict(frozen_at=stamp(), status='frozen_before_feature_extraction_or_controller_training',
                    parent=str(PARENT.relative_to(ROOT)), parent_sha256=sha(PARENT/'adapter_model.safetensors'),
                    sources={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for s, p in sources.items()},
                    training_rows=dict(path=str((train/'train-accepted.json').relative_to(ROOT)), sha256=sha(train/'train-accepted.json')),
                    input_policy='Exactly round-four plain photo/original-request/tool-note encoding. Last pre-generation hidden vector. Optional current-image statistics passed as numerical features, not target statistics. No labels, references, review or case IDs encoded.',
                    routes=['vlm', 'vlm-stats'], architecture='Frozen NF4 Qwen + frozen round2 curriculum adapter. Train a new 128-hidden GELU controller with 9 tool gates and 9 bounded continuous values. No new LoRA updates.',
                    training=dict(epochs=80, checkpoints=[20, 80], batch_size=32, learning_rate=.001, weight_decay=.01, seed=20261109,
                                  loss='Tool BCE (sqrt imbalance positive weights capped5) +4*active-only SmoothL1(beta0.1) on parameter/100. No pixel L1. Fitted normalization uses training features only.',
                                  decoding='Gate sigmoid>=0.5, tanh value*100 rounded to integer, zero omitted, sepia nonnegative. No image-specific rule or parameter repair.'),
                    selection='Same gate as round4: no severe damage in either development split, natural both-language complete>3/6, controlled>=81/100. Prioritize natural completion, then controlled, paired direct preference, earlier checkpoint. No L1 or training loss selection.',
                    acceptance='Closed. No acceptance input feature extraction or inference. No promotion without replication and final lock.',
                    code_hashes={str((ROOT/f).relative_to(ROOT)): sha(ROOT/f) for f in ('scripts/functional_v5/feature_runtime.py', 'scripts/functional_v4/runtime.py', 'scripts/functional_v1/runtime.py', 'scripts/functional_v1/editor.py', 'scripts/functional_v1/common.py')},
                    budget='Same original16 occupied GPU hours including cache construction, actual inference and trial. GPUs0/2/3 only; GPU1 protected.')
    write(EXP/'feature-protocol.json', protocol)
    write(DATA/'dataset-dependency.json', protocol['training_rows'])
    print('feature protocol frozen')


def run(gpu):
    import torch
    p = read(EXP/'feature-protocol.json')
    for f, h in p['code_hashes'].items():
        assert sha(ROOT/f) == h
    assert sha(PARENT/'adapter_model.safetensors') == p['parent_sha256']
    out = EXP/'features'
    out.mkdir(exist_ok=True)
    with GPUJob(gpu, 'v5-frozen-backbone-feature-cache', max_seconds=5400) as budget:
        from .feature_runtime import processor, load_model, extract_features
        torch.set_num_threads(4)
        backbone, proc = load_model(PARENT), processor()
        assert not any(w.requires_grad for w in backbone.parameters())
        for split, source in p['sources'].items():
            assert sha(ROOT/source['path']) == source['sha256']
            cases = read(ROOT/source['path'])
            rows = []
            for c in cases:
                for language in ('zh', 'en'):
                    file = out/split/f'{c["id"]}-{language}.pt'
                    if file.exists():
                        saved = torch.load(file, map_location='cpu', weights_only=True)
                        assert saved['input_sha256'] == c['input_sha256']
                        assert saved['request'] == c['instructions'][language]
                        assert saved['protocol_sha256'] == sha(EXP/'feature-protocol.json')
                    else:
                        budget.tick()
                        assert sha(ROOT/c['image']) == c['input_sha256']
                        hidden, stats = extract_features(backbone, proc, c, language)
                        saved = dict(case_id=c['id'], language=language, input_sha256=c['input_sha256'],
                                     request=c['instructions'][language], hidden=hidden, stats=stats,
                                     protocol_sha256=sha(EXP/'feature-protocol.json'))
                        file.parent.mkdir(exist_ok=True)
                        torch.save(saved, file)
                    rows.append(dict(case_id=c['id'], language=language, path=str(file.relative_to(ROOT)), sha256=sha(file)))
                    write(out/'status.json', dict(status='running', split=split, completed=len(rows), total=len(cases)*2, at=stamp()))
            write(out/f'{split}-manifest.json', dict(source_sha256=source['sha256'], protocol_sha256=sha(EXP/'feature-protocol.json'), rows=rows))
        write(out/'status.json', dict(status='completed', at=stamp(), acceptance_opened=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'run'])
    ap.add_argument('--gpu', choices=['0', '2', '3'])
    args = ap.parse_args()
    prepare() if args.command == 'prepare' else run(args.gpu)
