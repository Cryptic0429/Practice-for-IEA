"""Inspect full3k replica evidence and freeze all final-test model identities, or refuse."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new_json(p,data):
    with p.open('x') as f:f.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
    assert not (OUT/'test-opened.json').exists() and not (OUT/'final-models-lock.json').exists()
    protocolfile=OUT/'final-protocol.json';protocol=json.loads(protocolfile.read_text());assert protocol['status']=='frozen_before_test_inference'
    for path,digest in protocol['code_hashes'].items():assert sha(ROOT/path)==digest,path
    supplement=OUT/'final-analysis-method-supplement.json';method=json.loads(supplement.read_text());assert method['analysis_code_sha256']==sha(ROOT/'scripts/goal_v2_analyze_final.py')
    selectionfile=OUT/'3k-export-selection.json';selection=json.loads(selectionfile.read_text());assert selection['status']=='frozen_before_replica_validation'
    recipefile=OUT/'development-training-lock-3k-v2.json';recipe=json.loads(recipefile.read_text());assert recipe['status']=='frozen_training_recipe'
    statsfile=OUT/'validation-analysis/summary.json';stats=json.loads(statsfile.read_text());baseline=stats['models']['B'];assert stats['best_original_validation_model']=='B'
    val=json.loads((ROOT/'data/goal-v2-20261103/validation.json').read_text());ids=[r['id'] for r in val];assert len(ids)==len(set(ids))==200
    bpred={r['id']:r for r in (json.loads(l) for l in (OUT/'evaluation/validation-B/predictions.jsonl').open())};base_l1=np.array([bpred[i]['scoring']['edited_metrics']['pixel_L1'] for i in ids]);ix=np.random.default_rng(20261103).integers(0,200,(10000,200))
    challenge_source=ROOT/'data/goal-v2-20261103/instruction-challenge-v1.json';challenge_ids={r['id'] for r in json.loads(challenge_source.read_text())};challenge_baseline=json.loads((OUT/'instruction-challenge/B-original-format/summary.json').read_text())
    candidates={};proof={};reasons=[];seed_list=[20261103,20261104,20261105]
    for seed in seed_list:
        if seed==20261103:
            run=OUT/'json-calibrated3k-long1024-lr1e-4-s20261103';name=selection['primary_evaluation_name'];adapter=Path(selection['primary_adapter'])
        else:
            run=OUT/f'json3k-replica1024-lr1e-4-s{seed}';state=json.loads((OUT/f'worker-3k-replication-{seed}.json').read_text());assert state['status']=='completed';name=state['evaluation_name'];adapter=Path(state['adapter'])
            assert state['export_selection_sha256']==sha(selectionfile)
        config=json.loads((run/'run-config.json').read_text());state=json.loads((run/'trainer_state.json').read_text());training=json.loads((run/'training-input.json').read_text())
        assert config['seed']==config['data_seed']==seed and config['max_steps']==1024 and config['learning_rate']==1e-4 and config['warmup_ratio']==.05
        assert config['per_device_train_batch_size']==1 and config['gradient_accumulation_steps']==4 and state['global_step']==1024 and abs(state['epoch']-4096/3000)<1e-6
        assert training['sha256']==recipe['training_sha256']
        audit=json.loads((run/'checkpoint-1024-audit.json').read_text());assert audit['passed'] and audit['step']==1024 and audit['adapter_tensors']==audit['optimizer_parameter_states']==112
        weights=load_file(str(adapter/'adapter_model.safetensors'));assert len(weights)==112 and all(torch.isfinite(v).all() for v in weights.values())
        if selection['variant']=='swa':
            left=load_file(str(run/'checkpoint-512/adapter_model.safetensors'));right=load_file(str(run/'checkpoint-1024/adapter_model.safetensors'));assert weights.keys()==left.keys()==right.keys()
            assert all(torch.equal(weights[k],(left[k].float()+right[k].float())*.5) for k in weights)
            averaging=json.loads((adapter/'averaging.json').read_text());assert averaging['weights']==[.5,.5] and averaging['adapter_sha256']==sha(adapter/'adapter_model.safetensors')
        directory=OUT/'evaluation'/f'validation-{name}';assert json.loads((directory/'status.json').read_text())['status']=='completed'
        identity=json.loads((directory/'run.json').read_text())['identity'];assert identity['adapter_sha256']==sha(adapter/'adapter_model.safetensors')
        if seed!=20261103:assert datetime.datetime.fromisoformat(json.loads((directory/'run.json').read_text())['started_at'])>=datetime.datetime.fromisoformat(selection['frozen_at'])
        records=[json.loads(l) for l in (directory/'predictions.jsonl').open()];assert [r['id'] for r in records]==ids and len(records)==200
        m=stats['models'][name];comparison=stats['comparisons'][name];assert m['source_sha256']==sha(directory/'predictions.jsonl') and comparison['reference']=='B'
        diagnostics=json.loads((OUT/'validation-analysis'/f'{name}-direction-features.json').read_text());assert diagnostics['source_sha256']==m['source_sha256'];byid={r['id']:r for r in diagnostics['rows']}
        l1=np.array([r['scoring']['edited_metrics']['pixel_L1'] for r in records]);delta=base_l1-l1;adjusted=delta.copy()
        no_op=np.array([byid[r['id']]['near_identity'] or r['empty_parameters'] or r['all_zero_parameters'] or not r['scoring']['editor_success'] for r in records]);adjusted[no_op]=np.minimum(adjusted[no_op],0.)
        adjusted_relative=float(adjusted.mean()/base_l1.mean());adjusted_lower=float(np.quantile(adjusted[ix].mean(1),.025))
        checks={'point_target':comparison['relative_mean_gain']>=.03,'positive_interval':comparison['ci95_gain'][0]>0,'execution_maintained':m['executable']>=baseline['executable'],
                'direction_maintained':m['all_assessed_directions_aligned']>=baseline['all_assessed_directions_aligned'],'no_noop_gain_explanation':adjusted_relative>=.03 and adjusted_lower>0}
        challenge_dir=OUT/'instruction-challenge'/name;assert json.loads((challenge_dir/'status.json').read_text())['status']=='completed'
        ci=json.loads((challenge_dir/'identity.json').read_text());assert ci['adapter_sha256']==identity['adapter_sha256'] and ci['challenge_sha256']==sha(challenge_source)
        cs=json.loads((challenge_dir/'summary.json').read_text());cr=[json.loads(l) for l in (challenge_dir/'predictions.jsonl').open()];assert len(cr)==48 and {r['id'] for r in cr}==challenge_ids and cs['source_groups']==8
        checks['balanced_instruction_maintained']=all(cs[k]>=challenge_baseline[k] for k in ['executable','aligned_requests','strict_pairs'])
        key=f'Seed{seed}';candidates[key]={'path':str(adapter),'adapter_sha256':identity['adapter_sha256'],'config_sha256':sha(adapter/'adapter_config.json'),'format':'json','role':'candidate','seed':seed,'validation_name':name,'validation_l1':m['mean_l1']}
        proof[key]={'checks':checks,'validation_metrics':m,'comparison':comparison,'no_positive_noop_credit_relative_gain':adjusted_relative,'no_positive_noop_credit_ci95_gain_lower':adjusted_lower,'balanced_challenge':{k:cs[k] for k in ['executable','aligned_requests','strict_pairs','ordered_pairs']},'validation_predictions_sha256':m['source_sha256'],'checkpoint_audit_sha256':sha(run/'checkpoint-1024-audit.json'),'challenge_predictions_sha256':sha(challenge_dir/'predictions.jsonl')}
        reasons.extend(f'{key}:{k}' for k,v in checks.items() if not v)
    assert len({v['adapter_sha256'] for v in candidates.values()})==3
    gate={'passed':not reasons,'family':'3k-v2','variant':selection['variant'],'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_seeds':seed_list,'proof':proof,'failed_checks':reasons,'test_opened':False,'scope':'Development gate only; final independent test, broader visual review and archive verification remain required.'}
    gatefile=OUT/'multiseed-development-gate.json';new_json(gatefile,gate)
    if reasons:print(json.dumps({'development_gate_passed':False,'failed_checks':reasons,'test_not_opened':True},indent=2));return
    original=json.loads((OUT/'original-baselines-lock.json').read_text());models={}
    for name,entry in original['models'].items():
        adapter=Path(entry['path']);assert sha(adapter/'adapter_model.safetensors')==entry['adapter_sha256'] and sha(adapter/'adapter_config.json')==entry['config_sha256']
        models[name]={'path':str(adapter),'adapter_sha256':entry['adapter_sha256'],'config_sha256':entry['config_sha256'],'format':'original','role':'baseline'}
    models.update(candidates);data=json.loads((ROOT/'data/goal-v2-20261103/manifest.json').read_text());assert data['files']['test_sealed']['sha256']==protocol['test_source']['sha256']
    training_rows=json.loads((ROOT/'data/goal-v2-20261103/train3k-contrast-calibrated.json').read_text());test_rows=json.loads(Path(protocol['test_source']['path']).read_text());assert not {r['group_id'] for r in training_rows}&{r['group_id'] for r in test_rows}
    current_base=OUT/'base-source-current-audit.json';base_audit=json.loads(current_base.read_text());assert base_audit['passed'] and len(base_audit['files'])==16
    audited_at=datetime.datetime.fromisoformat(base_audit['checked_at']).timestamp()
    for entry in base_audit['files']:
        file=Path(entry['path']);assert file.stat().st_size==entry['bytes'] and file.stat().st_mtime<=audited_at
    codes=dict(protocol['code_hashes']);codes.update({p:sha(ROOT/p) for p in ['scripts/goal_v2_analyze_final.py','scripts/goal_v2_lock_final_models.py','scripts/goal_v2_average_adapter.py','scripts/goal_v2_final_worker.py']})
    locked={'status':'frozen_for_final_test','locked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'family':'3k-v2','variant':selection['variant'],'models':models,'baseline_keys':['A','B','SFT','GRPO'],'candidate_keys':list(candidates),'training_seeds':seed_list,
            'best_by_validation':min(candidates,key=lambda k:candidates[k]['validation_l1']),'selection_rule':'Minimum full200 validation L1 among the three predeclared same-recipe seeds; never replace from final test results.',
            'final_protocol_sha256':sha(protocolfile),'development_gate_sha256':sha(gatefile),'export_selection_sha256':sha(selectionfile),'training_recipe_sha256':sha(recipefile),'base_source_audit_sha256':sha(current_base),'analysis_supplement_sha256':sha(supplement),'test_source_sha256':protocol['test_source']['sha256'],'code_hashes':codes,
            'test_policy':'All model identities and protocol now fixed. Evaluate all71 with every original baseline and all3 seeds. No further test-driven tuning; inconclusive/failed result consumes this heldout for future confirmation.'}
    new_json(OUT/'final-models-lock.json',locked);print(json.dumps({'development_gate_passed':True,'final_model_lock_created':True,'candidate_keys':locked['candidate_keys'],'best_by_validation':locked['best_by_validation'],'test_inference_performed':False},indent=2))
if __name__=='__main__':main()
