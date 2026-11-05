"""Close the failed correction pilot and register diagnostic checkpoints, without promotion."""
import shutil
from .common import ROOT,DATA,EXP,EVAL,PARENT,read,write,sha,stamp


def main():
    path=EXP/'development-selection.json'
    if path.exists():raise RuntimeError('Already frozen')
    scores={split:{f'{route}-e{e}':read(EVAL/split/f'{route}-e{e}'/'direct-functional-summary.json') for route in ('dpo','sft') for e in (1,2)} for split in ('development','natural-use')}
    assert all(s['status']=='completed' for by_model in scores.values() for s in by_model.values())
    qualified=[m for m,s in scores['development'].items() if s['major_quality_issues']==0 and s['fully_completed_both_languages']>=81 and scores['natural-use'][m]['major_quality_issues']==0 and scores['natural-use'][m]['fully_completed_both_languages']>3]
    assert not qualified,'Pilot promotion needs a distinct final-lock workflow'
    result={'status':'correction_pilot_failed_practical_goal_not_achieved','frozen_at':stamp(),'development_results':scores['development'],'natural_results':scores['natural-use'],'models':{},'promoted':False,'acceptance_status':'closed_not_evaluated','seed_replication_started':False,'trial_default_changed':False,'reason':'Neither route improves the six original-request developmental probes, and all checkpoints retain a major exposure failure. A small 37-preference/48-update experiment cannot establish general ineffectiveness of DPO.'}
    for route in ('dpo','sft'):
        model=f'{route}-e1';source=EXP/'runs'/f'{route}-seed-20261107'/'checkpoint-24';dest=ROOT/'models/from-functional-v2'/f'{route}-correction'/'seed-01'
        dest.mkdir(parents=True,exist_ok=False)
        for filename in ('adapter_config.json','adapter_model.safetensors'):shutil.copy2(source/filename,dest/filename)
        item={'model_id':f'functional-v3-{route}-20261107','evaluation_id':model,'adapter':str(dest.relative_to(ROOT)),
          'adapter_sha256':sha(dest/'adapter_model.safetensors'),'parent':str(PARENT.relative_to(ROOT)),'parent_sha256':sha(PARENT/'adapter_model.safetensors'),
          'checkpoint':str(source.relative_to(ROOT)),'epoch':1,'step':24,'dataset_sha256':sha(DATA/'training-pool.json'),
          'diagnostic_choice':'Epoch1 dominates epoch2 on controlled completion with same natural count and major damage; neither qualifies for promotion.',
          'load_method':'Original Qwen base plus this one adapter; do not stack parent adapters.',
          'status':'diagnostic_trial_only_not_promoted','development_summary':scores['development'][model],
          'natural_summary':scores['natural-use'][model],'human_feedback':False,'frozen_at':stamp()}
        write(dest/'metadata.json',item);result['models'][item['model_id']]=item
    write(path,result)
    print(result['status'])
if __name__=='__main__':main()
