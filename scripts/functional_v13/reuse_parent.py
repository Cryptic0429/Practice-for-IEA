"""Exact frozen R11 predictions under the identical prompt/parser/editor."""
import copy
import json
import os
from functional_v1.common import ROOT,read,write,append,sha,stamp
from .runtime import EXP,PARENT_SHA
from .evaluate import KEYS
from functional_v11.outcomes import EDITOR_VERSION,parse


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    protocol=read(EXP/'training-protocol.json')
    previous=read(ROOT/'experiments/functional-v11-20261106/training-protocol.json')
    assert protocol['editor']==previous['editor']==EDITOR_VERSION
    assert protocol['schema']==previous['schema'] and protocol['system_prompt_sha256']==previous['system_prompt_sha256']
    for split in ('native-development','old-native','decision-development'):
        old=ROOT/'experiments/functional-v11-20261106/evaluation'/EDITOR_VERSION/split/'language-merger-e1'
        identity=read(old/'identity.json');assert identity['adapter_sha256']==PARENT_SHA
        source=read(EXP/'evaluation-protocol.json')['case_files'][KEYS[split]]
        assert identity['cases_sha256']==source['sha256']==sha(ROOT/source['path'])
        cases={c['id']:c for c in read(ROOT/source['path'])}
        rows=[json.loads(x) for x in (old/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==2*len(cases)
        out=EXP/'evaluation'/EDITOR_VERSION/split/'parent-frozen'
        assert not out.exists()
        for r in rows:
            c=cases[r['case_id']]
            assert r['input_sha256']==c['input_sha256']==sha(ROOT/c['image'])
            assert r['instruction']==c['instructions'][r['language']]
            if r.get('output_image'):assert sha(ROOT/r['output_image'])==r['output_sha256']
            if r.get('decision'):assert parse(r['response'])==r['decision']
            n=copy.deepcopy(r);n.update(model_id='parent-frozen',at=stamp(),
                reuse=dict(kind='exact_frozen_predictions_no_new_forward_or_pixel_change',
                    predictions=str((old/'predictions.jsonl').relative_to(ROOT)),sha256=sha(old/'predictions.jsonl')))
            append(out/'predictions.jsonl',n)
        write(out/'identity.json',dict(model_id='parent-frozen',adapter_sha256=PARENT_SHA,source_identity=identity,
            source_predictions_sha256=sha(old/'predictions.jsonl'),editor=EDITOR_VERSION,schema=protocol['schema'],
            cases_sha256=source['sha256'],kind='exact_frozen_predictions_reused_no_new_inference',model_forward_calls=0,GPU_hours=0))
        write(out/'status.json',dict(status='completed_predictions',completed=len(rows),total=len(rows),model_forward_calls=0,GPU_hours=0))
        print('Reused exact R11',split,len(rows))


if __name__=='__main__':main()
