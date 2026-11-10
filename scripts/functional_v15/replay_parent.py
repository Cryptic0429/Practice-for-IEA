"""Frozen-parent parameter replay isolates editor effects with zero GPU forwards."""
import argparse, copy, os
import numpy as np
from functional_v1.common import ROOT, read, write, append, sha, stamp
from .editor import load_rgb, save
from .outcomes import execute, parse, EDITOR_VERSION, SCHEMA_VERSION

EXP = ROOT / 'experiments/functional-v15-20261106'
PARENT_SHA = '12243ea5f08ab9e2e75b082fd6e6a1e6e8b1efbd7eb4e102be3563ba084c4a56'
SPLITS = {'native-development':'native_development', 'old-native':'old_native_regressions',
          'decision-development':'decision_development', 'development':'controlled_development'}
PROTOCOL = EXP / 'editor-only-protocol.json'


def freeze():
    assert not PROTOCOL.exists()
    previous = ROOT / 'experiments/functional-v13-20261106'
    cases = read(previous / 'evaluation-protocol.json')['case_files']
    sources = {}
    for split, key in SPLITS.items():
        folder = previous / 'evaluation/functional-editor-v11-negative-value-gain-candidate' / split / 'parent-frozen'
        identity = read(folder / 'identity.json')
        assert identity['adapter_sha256'] == PARENT_SHA and identity['cases_sha256'] == cases[key]['sha256']
        assert sha(ROOT / cases[key]['path']) == cases[key]['sha256']
        sources[split] = dict(identity=str((folder/'identity.json').relative_to(ROOT)),
            identity_sha256=sha(folder/'identity.json'), predictions=str((folder/'predictions.jsonl').relative_to(ROOT)),
            predictions_sha256=sha(folder/'predictions.jsonl'), cases=cases[key])
    files = ['scripts/functional_v15/replay_parent.py', 'scripts/functional_v15/editor.py',
             'scripts/functional_v15/outcomes.py', 'scripts/functional_v11/editor.py', 'scripts/functional_v11/outcomes.py']
    write(PROTOCOL, dict(at=stamp(), status='frozen_before_editor_only_outputs', sources=sources,
        parent_adapter_sha256=PARENT_SHA, editor=EDITOR_VERSION, schema=SCHEMA_VERSION,
        code_hashes={p:sha(ROOT/p) for p in files}, GPU_hours=0, model_forward_calls=0,
        policy='Execute exactly saved parent decisions; no parameter/prompt/checkpoint changes. '
               'Compare actual pixels and directly review changed images; exact unchanged pixels may inherit old AI reviews. '
               'All development sets have been used adaptively before; acceptance remains unopened. '
               'Any measured change is editor contribution only, never new learning evidence.'))
    print('Frozen CPU editor-only protocol for', list(sources))


def replay():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    protocol = read(PROTOCOL)
    assert all(sha(ROOT/p) == h for p,h in protocol['code_hashes'].items())
    import json
    for split, source in protocol['sources'].items():
        assert sha(ROOT/source['identity']) == source['identity_sha256']
        assert sha(ROOT/source['predictions']) == source['predictions_sha256']
        assert sha(ROOT/source['cases']['path']) == source['cases']['sha256']
        cases = {c['id']:c for c in read(ROOT/source['cases']['path'])}
        old_rows = [json.loads(l) for l in (ROOT/source['predictions']).read_text().splitlines()]
        assert len(old_rows) == 2 * len(cases)
        out = EXP / 'editor-only' / split
        assert not out.exists()
        out.mkdir(parents=True)
        changed = []
        for old in old_rows:
            c = cases[old['case_id']]; lang = old['language']
            assert sha(ROOT/c['image']) == c['input_sha256'] == old['input_sha256']
            assert c['instructions'][lang] == old['instruction'] and c['group_id'] == old['group_id']
            row = copy.deepcopy(old)
            row.update(at=stamp(), model_id='parent-editor-only-replay', editor=EDITOR_VERSION,
                output_image=None, output_sha256=None, original_prediction=old,
                evaluation_kind='CPU_exact_frozen_parent_decisions_no_model_forward', pixels_changed_by_editor=False,
                inference_performed=False, reuse=None)
            if old['status'] == 'failed':
                try: parse(old['response'])
                except (ValueError, TypeError): pass
                else: raise ValueError('Previously valid decision failed execution; inspect separately')
            else:
                assert parse(old['response']) == old['decision']
                result = execute(load_rgb(ROOT/c['image']), old['decision'])
                assert result['status'] == old['execution_status']
                if result['status'] == 'declined':
                    assert result['result'] is None and old['output_image'] is None
                else:
                    original = ROOT/old['output_image']; assert sha(original) == old['output_sha256']
                    same = np.array_equal(result['result'], load_rgb(original))
                    path = out / 'images' / f'{c["id"]}-{lang}.png'
                    path.parent.mkdir(exist_ok=True)
                    if same: os.link(original, path)
                    else: save(path, result['result'])
                    row.update(output_image=str(path.relative_to(ROOT)), output_sha256=sha(path),
                        input_output_identical=bool(np.array_equal(result['result'],load_rgb(ROOT/c['image']))),
                        pixels_changed_by_editor=not same)
                    if not same: changed.append(dict(case_id=c['id'], language=lang, output_image=row['output_image'],
                        output_sha256=row['output_sha256'], previous_image=old['output_image'], previous_sha256=old['output_sha256']))
            append(out/'predictions.jsonl', row)
        write(out/'identity.json', dict(at=stamp(), model_id='parent-editor-only-replay', parent_adapter_sha256=PARENT_SHA,
            editor=EDITOR_VERSION, schema=SCHEMA_VERSION, source=source, protocol_sha256=sha(PROTOCOL),
            model_forward_calls=0, GPU_hours=0, output_rows=len(old_rows), changed_rows=len(changed),
            kind='CPU_editor_contribution_only_not_new_model_inference'))
        write(out/'status.json', dict(status='completed_predictions', at=stamp(), rows=len(old_rows),
            changed_outputs=changed, GPU_hours=0, model_forward_calls=0))
        print(split, len(old_rows), 'CPU replays; changed',len(changed), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true'); args=parser.parse_args()
    freeze() if args.freeze else replay()
