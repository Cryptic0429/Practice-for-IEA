"""CPU replay isolates new editor from training and new-prompt inference."""
import json

import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, save, edit as old_edit
from functional_v6.evaluate import EVAL as OLD_EVAL, case_file as old_cases
from functional_v6 import visual_review as established
from .visual_review import EVAL, EXP, configure, case_file
from .editor_soft import edit


def main():
    lock = EXP / 'editor-only-replay-protocol.json'
    assert not lock.exists(), 'Preserve previous replay protocol'
    sources = []
    for split in ('native-development', 'old-native'):
        assert sha(case_file(split)) == sha(old_cases(split))
        for model in ('current-default', 'continued-e4'):
            source = OLD_EVAL / split / model / 'predictions.jsonl'
            sources.append(dict(split=split, source_model=model, predictions=str(source.relative_to(ROOT)),
                                predictions_sha256=sha(source)))
    write(lock, dict(at=stamp(), GPU_hours=0, sources=sources,
        editor_sha256=sha(ROOT / 'scripts/functional_v7/editor_soft.py'),
        code_sha256=sha(__file__), inference_performed=False, training_performed=False,
        purpose='Replay previously frozen original-pipeline parameters through the new editor. Separate editor contribution from new-prompt inference and training; not a model or checkpoint candidate.',
        review='Anonymous direct AI review with exact input/request/result observations reused only when hashes match. No L1 selection. Development only, acceptance remains closed.',
        declared_after_round7_selection=True, modifies_round7_selection=False))
    counts = {}
    for split in ('native-development', 'old-native'):
        cases = {c['id']: c for c in read(case_file(split))}
        ids = []
        for old_model in ('current-default', 'continued-e4'):
            model = 'editor-only-' + ('current-default' if old_model == 'current-default' else 'round6-parent')
            ids.append(model)
            source = OLD_EVAL / split / old_model
            destination = EVAL / split / model
            assert not destination.exists()
            destination.mkdir(parents=True)
            identity = dict(model_id=model, interpretation='CPU renderer diagnostic only; not new model inference',
                            inference_performed=False, source_identity=read(source / 'identity.json'),
                            source_predictions_sha256=sha(source / 'predictions.jsonl'),
                            editor='functional-editor-v7-soft-rgb-candidate',
                            editor_sha256=sha(ROOT / 'scripts/functional_v7/editor_soft.py'),
                            replay_protocol_sha256=sha(lock), at=stamp())
            write(destination / 'identity.json', identity)
            outputs = []
            for previous in map(json.loads, (source / 'predictions.jsonl').read_text().splitlines()):
                case = cases[previous['case_id']]
                row = {**previous, 'model_id': model, 'identity': identity, 'inference_performed': False,
                       'decision': None, 'source_status': previous['status'], 'output_image': None, 'output_sha256': None}
                if previous['status'] == 'executed':
                    before = load_rgb(ROOT / case['image'])
                    assert sha(ROOT / case['image']) == case['input_sha256']
                    assert np.array_equal(old_edit(before, previous['parameters']), load_rgb(ROOT / previous['output_image']))
                    after = edit(before, previous['parameters'], 'bounded-fixed')
                    output = destination / 'images' / f'{case["id"]}-{previous["language"]}.png'
                    save(output, after)
                    row.update(output_image=str(output.relative_to(ROOT)), output_sha256=sha(output),
                               execution_status='unchanged' if np.array_equal(before, after) else 'edited')
                outputs.append(row)
            (destination / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in outputs))
            counts[split + '/' + model] = len(outputs)
        configure()
        established.prepare(split, ids, 'editor-only-frozen-parameters')
    print(json.dumps(dict(language_records=counts, GPU_hours=0, inference_performed=False)))


if __name__ == '__main__':
    main()
