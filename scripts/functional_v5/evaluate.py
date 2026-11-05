"""Execute controller predictions from immutable input-only caches, with the same runtime decoder."""
import argparse
import json
from pathlib import Path
import torch
from .common import ROOT, EXP, EVAL, HELDOUT_DATA, read, write, append, sha, stamp, EDITOR_VERSION
from .feature_runtime import load_controller, normalized_features, decode
from functional_v1.common import parse
from functional_v1.editor import edit, load_rgb, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', choices=['vlm', 'vlm-stats'], required=True)
    ap.add_argument('--epoch', choices=[20, 80], type=int, required=True)
    ap.add_argument('--split', choices=['development', 'natural-use'], default='natural-use')
    args = ap.parse_args()
    torch.set_num_threads(4)
    protocol = read(EXP/'feature-protocol.json')
    for file, digest in protocol['code_hashes'].items():
        assert sha(ROOT/file) == digest
    checkpoint = EXP/'runs'/f'controller-{args.route}-seed-20261109'/f'checkpoint-{args.epoch}.pt'
    controller, item = load_controller(checkpoint)
    assert item['protocol_sha256'] == sha(EXP/'feature-protocol.json')
    assert item['parent_sha256'] == protocol['parent_sha256']
    assert item['runtime_sha256'] == sha(ROOT/'scripts/functional_v5/feature_runtime.py')
    source = protocol['sources'][args.split]
    assert sha(ROOT/source['path']) == source['sha256']
    cases = read(ROOT/source['path'])
    manifest = read(EXP/'features'/f'{args.split}-manifest.json')
    assert manifest['source_sha256'] == source['sha256'] and manifest['protocol_sha256'] == item['protocol_sha256']
    cached = {(r['case_id'], r['language']): r for r in manifest['rows']}
    mid = f'{args.route}-e{args.epoch}'
    out = EVAL/args.split/mid
    if (out/'predictions.jsonl').exists():
        raise RuntimeError('Preserve previous controller predictions')
    out.mkdir(parents=True, exist_ok=True)
    write(out/'identity.json', dict(model_id=mid, head_checkpoint=str(checkpoint.relative_to(ROOT)),
         head_sha256=sha(checkpoint), backbone_adapter=protocol['parent'], backbone_adapter_sha256=item['parent_sha256'],
         input_route=args.route, editor=EDITOR_VERSION, runtime_sha256=item['runtime_sha256'],
         feature_protocol_sha256=item['protocol_sha256'], case_sha256=source['sha256'], cpu_cached_prediction=True,
         note='Offline uses cached actual image/request prefill. Trial must recompute the identical feature and use this same normalized_features/decode path.'))
    rows = []
    for case in cases:
        for language in ('zh', 'en'):
            record = dict(case_id=case['id'], group_id=case['group_id'], family=case['family'], action=case['action'],
                          language=language, model_id=mid, instruction=case['instructions'][language],
                          input_image=case['image'], input_sha256=case['input_sha256'], status='failed')
            try:
                cached_row = cached[case['id'], language]
                assert sha(ROOT/cached_row['path']) == cached_row['sha256']
                feat = torch.load(ROOT/cached_row['path'], map_location='cpu', weights_only=True)
                assert feat['input_sha256'] == case['input_sha256'] and feat['request'] == case['instructions'][language]
                assert feat['protocol_sha256'] == item['protocol_sha256']
                x, _, _ = normalized_features(feat['hidden'][None], feat['stats'][None], item['route'], item['mean'], item['std'])
                with torch.inference_mode():
                    gate, value = controller(x)
                parameters = decode(gate[0], value[0])
                response = '```json\n'+json.dumps(parameters, indent=2)+'\n```'
                assert parse(response) == parameters
                output = out/'images'/f'{case["id"]}-{language}.png'
                assert sha(ROOT/case['image']) == case['input_sha256']
                save(output, edit(load_rgb(ROOT/case['image']), parameters))
                record.update(status='executed', parameters=parameters, response=response,
                              output_image=str(output.relative_to(ROOT)), output_sha256=sha(output))
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
            append(out/'predictions.jsonl', record)
            rows.append(record)
    write(out/'status.json', dict(status='completed_predictions', count=len(rows),
                                 executed=sum(r['status'] == 'executed' for r in rows), finished_at=stamp()))
    print(mid, args.split, len(rows), 'CPU cached predictions')


if __name__ == '__main__':
    main()
