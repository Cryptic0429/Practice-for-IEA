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
from .common import ROOT, EXP, EVAL, PARENT, GPUJob, read, write, append, sha, stamp, EDITOR_VERSION
from .common import HELDOUT_DATA as DATA
from functional_v1.editor import load_rgb, edit, save, statistics


def predict_all(model_id, adapter, split, gpu, route, limit=0):
    cases = read(DATA/f'{split}-final-cases.json')
    if split == 'acceptance':
        lock = read(EXP/'final-model-lock.json')
        assert lock['acceptance_cases_sha256'] == sha(DATA/'acceptance-final-cases.json')
        assert model_id in lock['models'], 'Final models must be locked before acceptance inference'
        locked = lock['models'][model_id]
        assert locked['adapter_sha256'] == (sha(Path(adapter)/'adapter_model.safetensors') if adapter else None)
        assert locked['route']==route
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
                'case_sha256': sha(DATA/f'{split}-final-cases.json'), 'editor': EDITOR_VERSION,
                'runtime_sha256': sha(ROOT/'scripts/functional_v4/runtime.py'),
                'base_runtime_sha256': sha(ROOT/'scripts/functional_v1/runtime.py'),
                'input_route': route, 'feature_version': 'image-stats-v1' if route=='measured' else None,
                'common_sha256': sha(ROOT/'scripts/functional_v1/common.py'),
                'editor_sha256': sha(ROOT/'scripts/functional_v1/editor.py'),
                'inputs': 'Current image, original bilingual request, fixed tool prompt/quality note; measured route adds deterministic statistics from the same visual input pixels.'}
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
                    response, parameters = predict(model, p, case, language, route)
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


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--adapter',type=Path)
    ap.add_argument('--split',choices=['development','acceptance','natural-use'],default='development')
    ap.add_argument('--gpu',choices=['0','2','3'],required=True)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--route',choices=['plain','measured'],required=True)
    args=ap.parse_args()
    protocol=read(EXP/'protocol.json')
    assert protocol['heldout_hashes'][args.split]==sha(DATA/f'{args.split}-final-cases.json')
    for file,digest in protocol['code_hashes'].items(): assert sha(ROOT/file)==digest
    predict_all(args.model,args.adapter,args.split,args.gpu,args.route,args.limit)
if __name__=='__main__':main()
