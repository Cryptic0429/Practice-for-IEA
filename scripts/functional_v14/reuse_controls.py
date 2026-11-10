"""Copy frozen R13 controls with zero new forward calls and exact source hashes."""
import copy,json,os
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v11.outcomes import EDITOR_VERSION,parse
from .runtime import EXP
from .evaluate import KEYS


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    target=EXP/'control-reuse.json';assert not target.exists()
    old=ROOT/'experiments/functional-v13-20261106/evaluation'/EDITOR_VERSION
    protocol=read(EXP/'training-protocol.json');records=[]
    for mid,source_mid in [('parent-frozen','parent-frozen'),('matched-narrow-e1','language-merger-e1')]:
        for split in ('native-development','old-native','decision-development','development'):
            source=old/split/source_mid;identity=read(source/'identity.json')
            case_record=read(EXP/'evaluation-protocol.json')['case_files'][KEYS[split]]
            assert identity['cases_sha256']==case_record['sha256']==sha(ROOT/case_record['path'])
            cases={c['id']:c for c in read(ROOT/case_record['path'])}
            rows=list(map(json.loads,(source/'predictions.jsonl').read_text().splitlines()))
            assert len(rows)==2*len(cases)
            out=EXP/'evaluation'/EDITOR_VERSION/split/mid;assert not out.exists();out.mkdir(parents=True)
            for row in rows:
                case=cases[row['case_id']];assert row['input_sha256']==case['input_sha256']==sha(ROOT/case['image'])
                assert row['instruction']==case['instructions'][row['language']]
                if row.get('output_image'):assert sha(ROOT/row['output_image'])==row['output_sha256']
                if row.get('decision'):assert parse(row['response'])==row['decision']
                row.update(model_id=mid,reuse=dict(predictions=str((source/'predictions.jsonl').relative_to(ROOT)),sha256=sha(source/'predictions.jsonl'),model_forward_calls=0,kind='exact_frozen_predictions_no_new_forward_or_pixel_change'))
            (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            write(out/'identity.json',dict(model_id=mid,adapter_sha256=identity['adapter_sha256'],source_identity=identity,source_predictions_sha256=sha(source/'predictions.jsonl'),
                cases_sha256=case_record['sha256'],editor=protocol['editor'],schema=protocol['schema'],kind='exact_frozen_predictions_reused_no_new_inference',model_forward_calls=0,GPU_hours=0))
            for name in ('direct-functional-summary.json','direct-review.json','decision-summary.json'):
                if not (source/name).exists():continue
                value=copy.deepcopy(read(source/name))
                if isinstance(value,dict):value.update(model_id=mid,inference_performed=False,runtime='Exact frozen R13 predictions and pixels reused; no new inference',source_file=str((source/name).relative_to(ROOT)),source_sha256=sha(source/name))
                write(out/name,value)
            write(out/'status.json',dict(status='completed_predictions',completed=len(rows),total=len(rows),model_forward_calls=0,GPU_hours=0))
            records.append(dict(split=split,model=mid,source_model=source_mid,language_predictions=len(rows),source_predictions_sha256=sha(source/'predictions.jsonl')))
    write(target,dict(at=stamp(),records=records,GPU_hours=0,model_forward_calls=0,policy='Exact R13 decisions, pixels and final AI assessments reused with source hashes; not new inference, new visual review or user approval.'))
    print('Reused parent and matched narrow controls; zero new GPU forward calls.')


if __name__=='__main__':main()
