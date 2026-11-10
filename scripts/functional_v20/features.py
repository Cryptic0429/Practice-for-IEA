"""Both hidden representations come from one frozen, target-free image/request forward."""
import argparse
from .common import ROOT, EXP, PARENT, PARENT_SHA, protocol, read, write, sha, stamp, GPUJob

def extract(model, p, case, language):
    import torch
    from functional_v1.runtime import encode
    from functional_v15.outcomes import input_row
    from functional_v19.photometry import features
    from .request_span import request_indices
    row = input_row(case, language)
    encoded = encode(p, row)
    indices = request_indices(p, row, case['instructions'][language], encoded)
    moved = {k: v.to('cuda') for k, v in encoded.items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = model.get_base_model().model(**moved, use_cache=False, return_dict=True).last_hidden_state
    last = hidden[0, -1].float().cpu()
    request_mean = hidden[0, indices].float().mean(0).cpu()
    stats = torch.from_numpy(features(ROOT / case['image']))
    assert last.shape == request_mean.shape == (3584,)
    assert all(torch.isfinite(t).all() for t in (last, request_mean, stats))
    return dict(last=last, request_mean=request_mean, stats=stats,
                request_tokens=len(indices), prefix_tokens=encoded['input_ids'].shape[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'old_native', 'native_development', 'decision_development'])
    args = ap.parse_args()
    p = protocol()
    assert read(EXP / 'controller-cpu-preflight.json')['status'] == 'passed'
    assert all(s in p['feature_sources'] for s in args.splits)
    assert not any('acceptance' in s for s in args.splits), 'Acceptance is sealed'
    import torch
    from functional_v15.runtime import load_model, processor
    torch.set_num_threads(4)
    with GPUJob(args.gpu, 'v20-frozen-R15-dual-pooling-cache', max_seconds=1500) as budget:
        model = load_model('language_merger', PARENT)
        assert not any(w.requires_grad for w in model.parameters())
        proc = processor()
        for split in args.splits:
            source = p['feature_sources'][split]
            assert sha(ROOT / source['path']) == source['sha256']
            cases = read(ROOT / source['path'])
            rows = []
            for case in cases:
                assert sha(ROOT / case['image']) == case['input_sha256']
                for language in ('zh', 'en'):
                    budget.tick()
                    path = EXP / 'features' / split / f'{case["id"]}-{language}.pt'
                    if path.exists():
                        saved = torch.load(path, map_location='cpu', weights_only=True)
                        assert saved['input_sha256'] == case['input_sha256'] and saved['request'] == case['instructions'][language]
                        assert saved['protocol_sha256'] == sha(EXP / 'training-protocol.json')
                    else:
                        saved = dict(extract(model, proc, case, language), input_sha256=case['input_sha256'],
                                     request=case['instructions'][language], protocol_sha256=sha(EXP / 'training-protocol.json'))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(saved, path)
                    rows.append(dict(case_id=case['id'], language=language, path=str(path.relative_to(ROOT)), sha256=sha(path)))
                    write(EXP / 'features/status.json', dict(status='running', split=split, completed=len(rows), total=len(cases)*2, at=stamp()))
            write(EXP / 'features' / f'{split}-manifest.json', dict(source_sha256=source['sha256'], rows=rows,
                  protocol_sha256=sha(EXP / 'training-protocol.json'), parent_sha256=PARENT_SHA, labels_in_features=False))
        assert sha(PARENT / 'adapter_model.safetensors') == PARENT_SHA
        write(EXP / 'features/status.json', dict(status='completed', splits=args.splits, acceptance_opened=False, at=stamp()))

if __name__ == '__main__': main()
