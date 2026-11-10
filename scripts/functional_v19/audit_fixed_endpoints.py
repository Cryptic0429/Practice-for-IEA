"""TRAIN-only diagnostic of fixed final checkpoints, never checkpoint selection."""
import argparse
import gc
import json
import os
from functional_v1.common import ROOT,read,write,append,sha,stamp,GPUJob

EXP=ROOT/'experiments/functional-v19-20261107'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',choices=['0','2','3'],required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu
    assert read(EXP/'development-selection.json')['status']=='failed_practical_gate'
    protocol=read(EXP/'fixed-endpoint-protocol.json')
    for path,digest in protocol['code_hashes'].items():assert sha(ROOT/path)==digest
    selected=set(protocol['explicit_visual_preference_case_ids'])
    cases=[case for pair in protocol['pairs'] for case in pair['cases']]
    assert len(cases)==24 and len(selected)==19
    for case in cases:assert sha(ROOT/case['image'])==case['input_sha256']
    for spec in protocol['models'].values():
        assert sha(ROOT/spec['adapter']/'adapter_model.safetensors')==spec['sha256']
        run=ROOT/spec['run']
        assert read(run/'status.json')['status']=='completed' and read(run/'status.json')['step']==1002
        assert read(ROOT/spec['adapter']/'trainer_state.json')['global_step']==1002
    import torch
    from PIL import Image
    from .runtime import load_model,processor,build_batch,predict
    from .bridge import set_features
    from .photometry import features
    from functional_v15.editor import load_rgb
    from functional_v15.outcomes import execute,EDITOR_VERSION,SCHEMA_VERSION
    from functional_v18.objective import sequence_logp
    torch.set_num_threads(4)
    def score(model,p,case,language,decision):
        set_features(model,features(ROOT/case['image']))
        batch={k:v.to('cuda') for k,v in build_batch(p,case,language,decision).items()}
        labels=batch.pop('labels')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(**batch,use_cache=False)
            value,count=sequence_logp(result.logits,labels)
        assert torch.isfinite(value)
        return dict(sum_logp=float(value),tokens=count,mean_logp=float(value/count))
    completed={}
    with GPUJob(args.gpu,'v19-fixed-final-TRAIN-specificity-audit',max_seconds=600) as budget:
        with budget.ledger() as ledger:
            next(j for j in ledger['jobs'] if j['id']==budget.id).update(editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v19.runtime',training_family='functional_v19')
        for route,spec in protocol['models'].items():
            out=EXP/'fixed-endpoint-diagnostic'/route
            out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
            model=load_model('photometric_bridge',adapter=ROOT/spec['adapter']);p=processor()
            assert all(not t.requires_grad for t in model.parameters())
            comparisons=[];predictions=[]
            for pair in protocol['pairs']:
                for index,case in enumerate(pair['cases']):
                    if case['id'] in selected:
                        for language in ('zh','en'):
                            own=score(model,p,case,language,case['decision'])
                            crossed=score(model,p,case,language,pair['cases'][1-index]['decision'])
                            comparisons.append(dict(case_id=case['id'],language=language,own=own,crossed=crossed,
                                own_minus_crossed_mean_logp=own['mean_logp']-crossed['mean_logp']))
                            budget.tick()
                    row=dict(case_id=case['id'],pair_id=pair['id'],language='en',input_image=case['image'],
                        input_sha256=case['input_sha256'],status='failed',output_image=None,output_sha256=None)
                    try:
                        response,decision=predict(model,p,case,'en')
                        value=execute(load_rgb(ROOT/case['image']),decision)
                        row.update(response=response,decision=decision,status=value['status'])
                        if value['result'] is not None:
                            path=out/'images'/(case['id']+'-en.png');Image.fromarray(value['result']).save(path,compress_level=1)
                            row.update(output_image=str(path.relative_to(ROOT)),output_sha256=sha(path))
                    except (ValueError,KeyError) as error:
                        row['error']=f'{type(error).__name__}: {error}'
                    append(out/'predictions.jsonl',row);predictions.append(row)
                    write(out/'teacher-forcing.json',comparisons)
                    budget.tick()
            assert len(comparisons)==38 and len(predictions)==24
            by_id={r['case_id']:r for r in predictions}
            equal=sum(by_id[pair['cases'][0]['id']].get('decision') is not None and
                by_id[pair['cases'][0]['id']].get('decision')==by_id[pair['cases'][1]['id']].get('decision') for pair in protocol['pairs'])
            summary=dict(at=stamp(),status='fixed_final_TRAIN_diagnostic_completed_not_functional_score',route=route,
                adapter=spec['adapter'],adapter_sha256=spec['sha256'],checkpoint_step=1002,
                scored_visual_preference_language_rows=38,
                own_preferred_by_mean_logp=sum(r['own_minus_crossed_mean_logp']>0 for r in comparisons),
                parent_own_preferred_by_mean_logp=28,English_generated_inputs=24,English_equal_decision_pairs=equal,
                parent_English_equal_decision_pairs=8,comparisons=comparisons,
                prediction_status_counts={status:sum(r['status']==status for r in predictions) for status in sorted({r['status'] for r in predictions})},
                English_declined_inputs=sum(r.get('decision',{}).get('action')=='decline' for r in predictions),
                generation_is_input_only=True,optimizer_steps=0,new_visual_grades=0,training_admission=False,
                checkpoint_selection_use=False,development_inputs_used=False,acceptance_inputs_used=False,
                limitation='Seen TRAIN sources and targets only. Parameter likelihood and unequal decisions do not establish editing quality.5 ambiguous cases are generated but excluded from preference ranking counts. No output automatically becomes a negative label.')
            assert sha(ROOT/spec['adapter']/'adapter_model.safetensors')==spec['sha256']
            write(out/'summary.json',summary);completed[route]={k:v for k,v in summary.items() if k!='comparisons'}
            del model;gc.collect();torch.cuda.empty_cache()
    write(EXP/'fixed-endpoint-diagnostic/result.json',dict(at=stamp(),models=completed,
        protocol_sha256=sha(EXP/'fixed-endpoint-protocol.json'),functional_selection_changed=False))


if __name__=='__main__':main()
