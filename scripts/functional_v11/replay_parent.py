"""CPU editor-only control from exact frozen R10 model decisions; never a forward."""
import copy
import json
import os
import numpy as np
from functional_v1.common import ROOT,read,write,append,sha,stamp
from functional_v1.editor import load_rgb,save
from .runtime import EXP,PARENT_SHA
from .outcomes import execute,parse,EDITOR_VERSION,SCHEMA_VERSION
from .evaluate import KEYS


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert read(EXP/'training-protocol.json')['parent_sha256']==PARENT_SHA
    old_exp=ROOT/'experiments/functional-v10-20261106'
    old_editor='functional-editor-v7-soft-contrast'
    # Resolve the actual frozen editor name from the previous protocol.
    old_editor=read(old_exp/'training-protocol.json')['editor']
    for split in ('native-development','old-native','decision-development'):
        source=old_exp/'evaluation'/old_editor/split/'language-merger-e2'
        old_identity=read(source/'identity.json')
        assert old_identity['adapter_sha256']==PARENT_SHA
        case_info=read(EXP/'evaluation-protocol.json')['case_files'][KEYS[split]]
        assert sha(ROOT/case_info['path'])==case_info['sha256']
        cases={c['id']:c for c in read(ROOT/case_info['path'])}
        out=EXP/'evaluation'/EDITOR_VERSION/split/'parent-editor-replay'
        assert not out.exists()
        rows=[json.loads(line) for line in (source/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==2*len(cases)
        count=0
        for old in rows:
            c=cases[old['case_id']];assert sha(ROOT/c['image'])==c['input_sha256']==old['input_sha256']
            r=copy.deepcopy(old)
            r.update(model_id='parent-editor-replay',at=stamp(),original_prediction=old,
                evaluation_kind='CPU_only_frozen_parameter_editor_replay_not_new_model_inference',
                output_image=None,output_sha256=None)
            if old['status']=='failed':
                try:parse(old['response'])
                except (ValueError,TypeError):pass
                else:raise AssertionError('Unexpected previously failed valid decision; examine execution failure separately')
            else:
                assert parse(old['response'])==old['decision']
                result=execute(load_rgb(ROOT/c['image']),old['decision'])
                if result['status']=='declined':assert result['result'] is None
                else:
                    path=out/'images'/f'{c["id"]}-{old["language"]}.png'
                    save(path,result['result'])
                    r.update(output_image=str(path.relative_to(ROOT)),output_sha256=sha(path),
                        input_output_identical=bool(np.array_equal(result['result'],load_rgb(ROOT/c['image']))))
                    count+=int(not np.array_equal(result['result'],load_rgb(ROOT/old['output_image'])))
            append(out/'predictions.jsonl',r)
        write(out/'identity.json',dict(model_id='parent-editor-replay',adapter_sha256=PARENT_SHA,
            source_identity=old_identity,source_predictions_sha256=sha(source/'predictions.jsonl'),
            editor=EDITOR_VERSION,schema=SCHEMA_VERSION,cases_sha256=case_info['sha256'],split=split,
            model_forward_calls=0,GPU_hours=0,kind='CPU_frozen_parameter_editor_replay',
            limitation='Measures editor contribution only; old raw model decisions remain exactly the same.'))
        write(out/'status.json',dict(status='completed_predictions',completed=len(rows),total=len(rows),
            editor_changed_outputs=count,at=stamp(),GPU_hours=0,model_forward_calls=0))
        print(split, len(rows), 'CPU replays, changed images',count)


if __name__=='__main__':main()
