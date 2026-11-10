"""New CPU head plus exact input-bound archived parent semantics; actual new edits."""
import argparse,json
import numpy as np
import torch
from .common import ROOT,EXP,POLICIES,protocol,read,write,sha,append,stamp
from .head import load_head,predict_cached
from functional_v15.outcomes import execute,EDITOR_VERSION,SCHEMA_VERSION
from functional_v1.editor import load_rgb,save

KEYS={'old-native':'old_native','native-development':'native_development','decision-development':'decision_development','development':'controlled_development'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--head',required=True)
    ap.add_argument('--policy',choices=POLICIES,required=True);ap.add_argument('--splits',nargs='+',choices=KEYS,default=['old-native'])
    args=ap.parse_args();p=protocol();torch.set_num_threads(4)
    net,item=load_head(ROOT/args.head);assert item['protocol_sha256']==sha(EXP/'training-protocol.json')
    semantic_source=p['parent_semantics'];assert sha(ROOT/semantic_source['path'])==semantic_source['sha256']
    semantics=read(ROOT/semantic_source['path'])['records'];textures=read(EXP/'texture-features.json')
    assert textures['implementation_sha256']==sha(ROOT/'scripts/functional_v21/texture.py')
    for split in args.splits:
        internal=KEYS[split];source=p['sources'][internal]
        assert sha(ROOT/source['path'])==source['sha256'];cases=read(ROOT/source['path'])
        manifest_source=p['hidden_manifests'].get(internal)
        expected_feature_protocol=p['hidden_protocol_sha256']
        split_textures=textures
        if manifest_source is None:
            assert internal=='controlled_development'
            binding=read(EXP/'controlled-feature-binding.json')
            assert binding['training_protocol_sha256']==sha(EXP/'training-protocol.json')
            assert binding['source_cases_sha256']==source['sha256'] and args.model in binding['stage1_eligible_models']
            gate=read(EXP/'stage1-gate.json')
            assert sha(EXP/'stage1-gate.json')==binding['stage1_gate_sha256'] and args.model in gate['eligible']
            manifest_source=binding['manifest'];expected_feature_protocol=sha(EXP/'training-protocol.json')
            texture_source=binding['texture']
            assert sha(ROOT/texture_source['path'])==texture_source['sha256']
            split_textures=read(ROOT/texture_source['path'])
            assert split_textures['implementation_sha256']==sha(ROOT/'scripts/functional_v21/texture.py')
        assert sha(ROOT/manifest_source['path'])==manifest_source['sha256'];manifest=read(ROOT/manifest_source['path'])
        cached={(r['case_id'],r['language']):r for r in manifest['rows']}
        parents={(r['case_id'],r['language']):r for r in semantics[internal]}
        out=EXP/'evaluation'/EDITOR_VERSION/split/args.model;out.mkdir(parents=True,exist_ok=False)
        write(out/'identity.json',dict(model_id=args.model,policy=args.policy,head=args.head,head_sha256=sha(ROOT/args.head),
            parent_sha256=p['parent_sha256'],parent_semantics_sha256=semantic_source['sha256'],parent_LM_reinferred=False,
            new_CPU_head_inference=True,cases_sha256=source['sha256'],runtime='functional_v21.runtime',editor=EDITOR_VERSION,schema=SCHEMA_VERSION,
            feature_manifest_sha256=manifest_source['sha256'],texture_implementation_sha256=split_textures['implementation_sha256'],
            feature_protocol_sha256=expected_feature_protocol,protocol_sha256=item['protocol_sha256']))
        rows=[]
        for case in cases:
            assert sha(ROOT/case['image'])==case['input_sha256']
            for language in ('zh','en'):
                row=dict(case_id=case['id'],group_id=case['group_id'],family=case['family'],domain=case['domain'],language=language,
                    instruction=case['instructions'][language],model_id=args.model,input_image=case['image'],input_sha256=case['input_sha256'],
                    status='failed',output_image=None,output_sha256=None)
                try:
                    prior=parents[case['id'],language]
                    assert prior['input_sha256']==case['input_sha256'] and prior['instruction']==case['instructions'][language]
                    if prior['status']=='failed':raise ValueError('parent_model_failed:'+str(prior['parent_error']))
                    f=cached[case['id'],language];assert sha(ROOT/f['path'])==f['sha256']
                    feature=torch.load(ROOT/f['path'],map_location='cpu',weights_only=True)
                    assert feature['protocol_sha256']==expected_feature_protocol
                    assert feature['input_sha256']==case['input_sha256'] and feature['request']==case['instructions'][language]
                    tex=split_textures['images'][case['image']];assert tex['input_sha256']==case['input_sha256']
                    response,decision=predict_cached(net,item,feature,torch.tensor(tex['values'],dtype=torch.float32),prior['parent_decision'],args.policy)
                    result=execute(load_rgb(ROOT/case['image']),decision)
                    row.update(response=response,decision=decision,parent_decision=prior['parent_decision'],parameters=result['parameters'],execution_status=result['status'])
                    if result['status']=='declined':row.update(status='declined',reason_code=result['reason_code']);assert result['result'] is None
                    else:
                        path=out/'images'/f'{case["id"]}-{language}.png';save(path,result['result'])
                        row.update(status='executed',output_image=str(path.relative_to(ROOT)),output_sha256=sha(path),
                            input_output_identical=bool(np.array_equal(load_rgb(ROOT/case['image']),result['result'])))
                except Exception as error:row['error']=f'{type(error).__name__}: {error}'
                row['at']=stamp();append(out/'predictions.jsonl',row);rows.append(row)
        write(out/'status.json',dict(at=stamp(),status='completed_predictions',completed=len(rows),total=2*len(cases),
            status_counts={s:sum(r['status']==s for r in rows) for s in ('executed','declined','failed')}))
    assert not torch.cuda.is_initialized()

if __name__=='__main__':main()
