"""Frozen decision runtime evaluation; declines never create result images."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from functional_v1.common import ROOT, read, write, append, sha, stamp, GPUJob
from functional_v1.editor import load_rgb, save
from .runtime import EXP, verify_origin, ROUTES
from .outcomes import input_row, parse, execute, EDITOR_VERSION, SCHEMA_VERSION


KEYS = {'native-development': 'native_development', 'old-native': 'old_native_regressions',
        'decision-development': 'decision_development', 'development': 'controlled_development',
        'native-acceptance': 'native_acceptance', 'acceptance': 'controlled_acceptance'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    parser.add_argument('--splits', nargs='+', choices=list(KEYS), default=['native-development', 'old-native', 'decision-development'])
    args = parser.parse_args()
    protocol = read(EXP / 'training-protocol.json')
    for name, digest in protocol['code_hashes'].items():
        assert sha(ROOT / name) == digest
    evaluation = read(EXP / 'evaluation-protocol.json')
    assert sha(EXP / 'evaluation-protocol.json') == protocol['evaluation_protocol_sha256']
    origin_identity = verify_origin()
    adapter = args.adapter
    if adapter is not None and not adapter.is_absolute():
        adapter = ROOT / adapter
    adapter_hash = sha(adapter / 'adapter_model.safetensors') if adapter else None
    queues = []
    for split in args.splits:
        source = evaluation['case_files'][KEYS[split]]
        path = ROOT / source['path']
        assert sha(path) == source['sha256']
        if 'acceptance' in split:
            lock = read(EXP / 'final-model-lock.json')
            assert lock['models'][args.model]['adapter_sha256'] == adapter_hash
            assert lock['evaluation_protocol_sha256'] == sha(EXP / 'evaluation-protocol.json')
        cases = read(path)
        out = EXP / 'evaluation' / EDITOR_VERSION / split / args.model
        identity = dict(model_id=args.model, route=args.route, origin_identity=origin_identity, adapter=str(adapter.relative_to(ROOT)) if adapter else None,
                        adapter_sha256=adapter_hash, runtime='functional_v19.runtime', schema=SCHEMA_VERSION,
                        editor=EDITOR_VERSION, split=split, cases_sha256=sha(path),
                        input_policy='Only original input photograph, one original-language request and frozen tools. Current-image RGB statistics are deterministically computed by the frozen feature function and added to visual tokens. No target, reference, reviewer notes, case ID or expected action enters the model.',
                        code_hashes=protocol['code_hashes'])
        if (out / 'identity.json').exists():
            assert read(out / 'identity.json') == identity
        else:
            write(out / 'identity.json', identity)
        rows = [json.loads(line) for line in (out / 'predictions.jsonl').read_text().splitlines()] if (out / 'predictions.jsonl').exists() else []
        done = {(r['case_id'], r['language']) for r in rows}
        assert len(done) == len(rows)
        if len(done) < len(cases) * 2:
            queues.append((split, cases, out, rows, done))
    if not queues:
        return
    with GPUJob(args.gpu, 'v19-evaluate-' + args.model, max_seconds=1800) as budget:
        with budget.ledger() as value:
            next(j for j in value['jobs'] if j['id'] == budget.id).update(editor=EDITOR_VERSION, schema=SCHEMA_VERSION, runtime='functional_v19.runtime')
        import torch
        from .runtime import processor, load_model, predict
        torch.set_num_threads(4)
        model = load_model(args.route, adapter)
        p = processor()
        for split, cases, out, rows, done in queues:
            for c in cases:
                assert sha(ROOT / c['image']) == c['input_sha256']
                for language in ('zh', 'en'):
                    if (c['id'], language) in done:
                        continue
                    budget.tick()
                    row = dict(case_id=c['id'], group_id=c['group_id'], family=c['family'], domain=c['domain'],
                               language=language, instruction=c['instructions'][language], model_id=args.model,
                               input_image=c['image'], input_sha256=c['input_sha256'], status='failed',
                               output_image=None, output_sha256=None)
                    try:
                        response, decision = predict(model, p, c, language)
                        row['response'] = response
                        result = execute(load_rgb(ROOT / c['image']), decision)
                        row.update(decision=decision, parameters=result['parameters'], execution_status=result['status'])
                        if result['status'] == 'declined':
                            assert result['result'] is None
                            row.update(status='declined', reason_code=result['reason_code'])
                        else:
                            path = out / 'images' / f'{c["id"]}-{language}.png'
                            save(path, result['result'])
                            row.update(status='executed', output_image=str(path.relative_to(ROOT)), output_sha256=sha(path),
                                       input_output_identical=bool(np.array_equal(load_rgb(ROOT / c['image']), result['result'])))
                    except Exception as error:
                        row['error'] = f'{type(error).__name__}: {error}'
                    row['at'] = stamp()
                    append(out / 'predictions.jsonl', row)
                    rows.append(row)
                    done.add((c['id'], language))
                    write(out / 'status.json', dict(status='running', pid=os.getpid(), completed=len(done), total=len(cases) * 2, at=stamp()))
            write(out / 'status.json', dict(status='completed_predictions', completed=len(rows), total=len(cases) * 2,
                                            status_counts={s: sum(r['status'] == s for r in rows) for s in ('executed', 'declined', 'failed')}, at=stamp()))


if __name__ == '__main__':
    main()
