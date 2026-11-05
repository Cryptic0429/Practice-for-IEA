"""Frozen functional tasks; failed outputs stay failures and never become hidden no-ops."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
from .common import ROOT, DATA, EXP, EVAL, PARENT, GPUJob, read, write, append, sha, stamp, EDITOR_VERSION
from .editor import load_rgb, edit, save, statistics


def predict_all(model_id, adapter, split, gpu, limit=0):
    cases = read(DATA/f'{split}-cases.json')
    if split == 'acceptance':
        lock = read(EXP/'final-model-lock.json')
        assert lock['acceptance_cases_sha256'] == sha(DATA/'acceptance-cases.json')
        assert model_id in lock['models'], 'Final models must be locked before acceptance inference'
        locked = lock['models'][model_id]
        assert locked['adapter_sha256'] == (sha(Path(adapter)/'adapter_model.safetensors') if adapter else None)
        assert locked['adapter'] == (str(Path(adapter).relative_to(ROOT)) if adapter else None)
        assert lock['editor'] == EDITOR_VERSION
        for file, digest in lock['code_hashes'].items():
            assert sha(ROOT/file)==digest, 'Final inference code changed after locking: '+file
    out = EVAL/split/model_id
    out.mkdir(parents=True, exist_ok=True)
    path = out/'predictions.jsonl'
    old = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done = {(r['case_id'], r['language']) for r in old}
    identity = {'model_id': model_id, 'adapter': str(adapter) if adapter else None,
                'adapter_sha256': sha(Path(adapter)/'adapter_model.safetensors') if adapter else None,
                'case_sha256': sha(DATA/f'{split}-cases.json'), 'editor': EDITOR_VERSION,
                'runtime_sha256': sha(Path(__file__).with_name('runtime.py')),
                'common_sha256': sha(Path(__file__).with_name('common.py')),
                'editor_sha256': sha(Path(__file__).with_name('editor.py')),
                'inputs': 'Original image, original bilingual request and fixed tool specification only.'}
    if (out/'identity.json').exists():
        assert read(out/'identity.json') == identity
    else:
        write(out/'identity.json', identity)
    if len(done) == 2*len(cases):
        return
    with GPUJob(gpu, f'predict-{split}-{model_id}', max_seconds=5400) as budget:
        import torch
        from .runtime import load_model, processor, predict
        torch.set_num_threads(4)
        model, p = load_model(adapter), processor()
        count = 0
        for case in cases:
            for language in ('zh', 'en'):
                if (case['id'], language) in done:
                    continue
                if limit and count >= limit:
                    break
                budget.tick()
                record = {'case_id': case['id'], 'group_id': case['group_id'], 'family': case['family'],
                          'action': case['action'], 'language': language, 'model_id': model_id,
                          'instruction': case['instructions'][language], 'input_sha256': case['input_sha256'],
                          'input_image': case['image'], 'status': 'failed'}
                assert sha(ROOT/case['image']) == case['input_sha256']
                try:
                    response, parameters = predict(model, p, case, language)
                    rgb = load_rgb(ROOT/case['image'])
                    image = edit(rgb, parameters)
                    output = out/'images'/f'{case["id"]}-{language}.png'
                    save(output, image)
                    record.update(status='executed', response=response, parameters=parameters,
                                  output_image=str(output.relative_to(ROOT)), output_sha256=sha(output),
                                  input_output_identical=bool(np.array_equal(rgb, image)),
                                  input_statistics=statistics(rgb), output_statistics=statistics(image),
                                  diagnostic_output_input_L1=float(np.abs(image.astype(np.float32)-rgb).mean()/255))
                except Exception as error:
                    record.update(error=f'{type(error).__name__}: {error}')
                record['completed_at'] = stamp()
                append(path, record)
                old.append(record)
                done.add((case['id'], language))
                count += 1
                write(out/'status.json', {'status': 'running', 'pid': os.getpid(),
                                          'completed': len(done), 'total': 2*len(cases), 'updated_at': stamp()})
            if limit and count >= limit:
                break
        write(out/'status.json', {'status': 'completed_predictions' if len(done) == 2*len(cases) else 'partial',
                                  'completed': len(done), 'total': 2*len(cases),
                                  'executed': sum(r['status']=='executed' for r in old), 'finished_at': stamp()})


def score_all(model_id, split, gpu, limit=0):
    cases = {c['id']: c for c in read(DATA/f'{split}-cases.json')}
    out = EVAL/split/model_id
    records = [json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()]
    path = out/'frozen-base-review.jsonl'
    reviews = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done = {(r['case_id'], r['language']) for r in reviews}
    from .runtime import load_model, processor
    from .review_runtime import judge
    with GPUJob(gpu, f'review-{split}-{model_id}', max_seconds=7200) as budget:
        import torch
        torch.set_num_threads(4)
        model, p = load_model(), processor()
        count = 0
        for record in records:
            key = record['case_id'], record['language']
            if key in done:
                continue
            if limit and count >= limit:
                break
            budget.tick()
            case = cases[record['case_id']]
            if record['status'] != 'executed':
                result = {'completion': 'failed', 'quality': 'uncertain', 'strength': 'uncertain',
                          'evidence': 'No executable edit was produced.'}
            elif record['input_output_identical'] and case['action'] != 'identity':
                result = {'completion': 'failed', 'quality': 'acceptable', 'strength': 'insufficient',
                          'evidence': 'The output is pixel-identical to the input despite a substantive editing request.'}
            else:
                assert sha(ROOT/record['output_image']) == record['output_sha256']
                try:
                    result = judge(model, p, case, ROOT/record['output_image'])
                except Exception as error:
                    result = {'completion': 'uncertain', 'quality': 'uncertain', 'strength': 'uncertain',
                              'evidence': f'Unparseable review: {type(error).__name__}: {error}'}
            item = {'case_id': key[0], 'language': key[1], 'family': record['family'],
                    'output_sha256': record.get('output_sha256'), 'assessment': result,
                    'reviewer': 'frozen original Qwen2.5-VL-7B-Instruct', 'human_review': False}
            append(path, item)
            reviews.append(item)
            done.add(key)
            count += 1
            write(out/'review-status.json', {'status': 'running', 'pid': os.getpid(),
                                            'completed': len(done), 'total': len(records), 'updated_at': stamp()})
        completed = lambda r: r['assessment']['completion']=='completed' and r['assessment']['quality']=='acceptable'
        grouped = collections.defaultdict(list)
        for row in reviews:
            grouped[row['case_id']].append(row)
        summary = {'status': 'completed' if len(done)==100 else 'partial', 'model_id': model_id,
                   'reviewed_language_outputs': len(done), 'source_groups': len(grouped),
                   'fully_completed_both_languages': sum(len(v)==2 and all(completed(r) for r in v) for v in grouped.values()),
                   'major_quality_issues': sum(r['assessment']['quality']=='major_issue' for r in reviews),
                   'per_language': {lang: dict(collections.Counter(r['assessment']['completion'] for r in reviews if r['language']==lang)) for lang in ('zh','en')},
                   'per_family': {family: {'fully_completed_both_languages': sum(len(v)==2 and all(completed(r) for r in v) for v in grouped.values() if v[0]['family']==family),
                                          'cases': sum(v[0]['family']==family for v in grouped.values())} for family in sorted({r['family'] for r in reviews})},
                   'selection_uses_L1': False,
                   'limits': ['AI-assisted preliminary review; direct Codex visual review remains separate.',
                              'One pair of languages shares one source group, not two independent samples.',
                              'Source photos have historical evaluation exposure.']}
        write(out/'functional-summary.json', summary)
        write(out/'review-status.json', {'status': summary['status'], 'completed': len(done), 'total': len(records), 'finished_at': stamp()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['predict','score'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', type=Path)
    ap.add_argument('--split', choices=['development','acceptance'], default='development')
    ap.add_argument('--gpu', choices=['0','2','3'], required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    adapter = PARENT if args.model == 'current' else args.adapter
    if args.command == 'predict':
        predict_all(args.model, adapter, args.split, args.gpu, args.limit)
    else:
        score_all(args.model, args.split, args.gpu, args.limit)


if __name__ == '__main__':
    main()
