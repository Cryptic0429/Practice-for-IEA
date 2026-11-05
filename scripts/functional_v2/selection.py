"""Freeze development-only choices; never promote a checkpoint with new major damage."""
import json
import shutil
from collections import Counter, defaultdict
from .common import ROOT, DATA, EXP, EVAL, PARENT, read, write, sha, stamp


def main():
    destination=EXP/'development-selection.json'
    if destination.exists():
        raise RuntimeError('Selection already frozen; do not silently reselect')
    baseline=read(EVAL/'development/parent-v1/direct-functional-summary.json')
    summaries={f'{route}-e{epoch}':read(EVAL/'development'/f'{route}-e{epoch}'/'direct-functional-summary.json') for route in ('curriculum','control') for epoch in range(1,5)}
    assert all(s['status']=='completed' for s in summaries.values())
    tie=read(EXP/'review/development/control-tie/observations.json')
    assert len(tie)==9 and all(sha(ROOT/r['image'])==r['sha256'] for r in tie)
    assert Counter(r['preferred_model'] for r in tie)=={'control-e2':5,'control-e4':4}
    chosen={'curriculum':'curriculum-e3','control':'control-e2'}
    result={'status':'development_complete_practical_goal_not_achieved','frozen_at':stamp(),
      'selection_policy':read(EXP/'protocol.json')['selection'], 'acceptance_status':'closed_not_evaluated',
      'acceptance_final_cases_sha256':sha(DATA/'acceptance-final-cases.json'),
      'training_data_sha256':sha(DATA/'train-accepted.json'), 'control_data_sha256':sha(DATA/'control-accepted.json'),
      'primary_unit':'100 bilingual semantic tasks nested in 20 photo groups, 10 portrait/10 landscape',
      'baseline':baseline, 'all_development_summaries':summaries,'models':{},
      'trial_default_changed':False,'seed_replication_started':False,
      'promotion_decision':'Rejected: every curriculum checkpoint introduces at least one major-quality case relative to the zero-major baseline. No seed replication or acceptance opening for promotion.',
      'tie_review':'Control epochs 2/4 tie at 34 complete, zero major. Nine different language outputs directly compared with identity hidden; e2 preferred 5, e4 4. Seven photo groups: e2 wins3/e4 wins2/tie2. Same verdict under earlier-epoch fallback.',
      'review_mode':'AI-assisted direct Codex visual review, not user validation; no L1 in selection'}
    for route,model_id in chosen.items():
        epoch=int(model_id[-1]); run=EXP/'runs'/f'{route}-seed-20261106';source=run/f'checkpoint-{epoch*180}'
        dest=ROOT/'models/from-functional-v1'/f'sft-{route}'/'seed-01'
        if dest.exists():raise RuntimeError('Refuse to overwrite registered model')
        dest.mkdir(parents=True)
        for name in ('adapter_config.json','adapter_model.safetensors'):
            shutil.copy2(source/name,dest/name)
            assert sha(source/name)==sha(dest/name)
        metadata={'model_id':f'functional-v2-{route}-20261106','evaluation_id':model_id,
          'adapter':str(dest.relative_to(ROOT)), 'adapter_sha256':sha(dest/'adapter_model.safetensors'),
          'parent':str(PARENT.relative_to(ROOT)),'parent_sha256':sha(PARENT/'adapter_model.safetensors'),
          'load_method':'Original Qwen2.5-VL-7B base plus this ONE adapter; do not stack the parent adapter',
          'checkpoint':str(source.relative_to(ROOT)), 'epoch':epoch,'step':epoch*180,
          'dataset_sha256':sha(DATA/('train-accepted.json' if route=='curriculum' else 'control-accepted.json')),
          'development_summary':summaries[model_id], 'acceptance':'not opened', 'user_feedback':'none',
          'promoted':False,'frozen_at':stamp()}
        write(dest/'metadata.json',metadata)
        result['models'][metadata['model_id']]=metadata
    write(destination,result)
    print(json.dumps({'status':result['status'],'models':{k:v['adapter_sha256'] for k,v in result['models'].items()}}))

if __name__=='__main__':main()
