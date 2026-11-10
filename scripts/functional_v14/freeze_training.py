"""Freeze one late-vision hypothesis with exact R13 data and unchanged functional gates."""
import copy,hashlib
from functional_v1.common import ROOT,read,write,sha,stamp,GPUJob,EXP as BUDGET_EXP
from functional_v11.outcomes import system_prompt,EDITOR_VERSION,SCHEMA_VERSION
from .runtime import EXP,DATA,PARENT,PARENT_SHA,ROUTES,verify_origin


def main():
    assert not (EXP/'training-protocol.json').exists()
    old=ROOT/'experiments/functional-v13-20261106';p13=read(old/'training-protocol.json')
    ledger=read(BUDGET_EXP/'resources.json');used=GPUJob.consumed(ledger);cap=ledger['limit_gpu_hours']
    assert cap==24 and cap-used>5.5
    for path,h in p13['code_hashes'].items():assert sha(ROOT/path)==h
    for name,key in [('train-accepted.json','dataset_sha256'),('cases.json','cases_sha256')]:assert sha(DATA/name)==p13[key]
    assert read(EXP/'shared-data-audit.json')['status']=='passed'
    transfer=read(EXP/'runtime-transfer-audit.json');assert transfer['results'][0]['trainable_tensors']==156
    assert transfer['results'][0]['trainable_parameters']==3371648
    assert transfer['parent_sha256']==PARENT_SHA
    assert read(EXP/'initial-function-parity.json')['status']=='passed'
    architecture=read(EXP/'installed-vision-architecture.json')
    assert architecture['vision_config']['depth']==32 and len(architecture['last4_linear_modules'])==20
    evaluation=copy.deepcopy(read(old/'evaluation-protocol.json'))
    for item in evaluation['case_files'].values():assert sha(ROOT/item['path'])==item['sha256']
    evaluation.update(at=stamp(),status='round14_cases_gates_and_visual_hypothesis_frozen_before_optimization',
        comparisons=['Exact R11 parent and R13 narrow checkpoints reused with explicit source identity; no new inference claimed.',
            'Same R11 parent,925/1850 R13 data, seed,1850-row order,2epochs,LR5e-5,tools,prompt,pixels; add LoRA only in visual blocks28..31.',
            'R13 narrow/wide results are existing controls, not newly retuned or rerun candidates.'],
        selection=dict(checkpoint='Epoch1 or2 of visual_late4,seed20261127.',
            eligibility='Unchanged: native>=16/20,old>=4/6,controls20/20,zero marked major;then controlled>=81/100 and zero major. No feasible-task refusal counted complete.',
            rank=['More bilingual complete+acceptable native cases','Fewer severe side effects','Better anonymous paired visual outcome if needed','Earlier checkpoint']),
        resource=dict(shared_gpu_hour_cap=24,allowed_gpu_indices=[0,2,3],gpu1_original_service_preserved=True,ledger=str((BUDGET_EXP/'resources.json').relative_to(ROOT))),
        limitations=['Same repeatedly used development inputs, adaptive comparison; not an independent test.',
            'All925 labels inherit R13 AI evidence; zero new photographs, zero new independent review.',
            'Late-visual adaptation is a hypothesis; official PEFT support and initial parity do not imply quality gain.'],
        seed_replication='Only if all development gates pass without new major damage;seed20261128 on selected route, then freeze before acceptance.',
        new_training_started=False,actual_user_feedback=0)
    write(EXP/'evaluation-protocol.json',evaluation)
    settings=copy.deepcopy(p13['training']);settings.update(trainable_tensors={'visual_late4':156},new_language_B_zero=False,
        new_visual_block_B_zero=True,visual_block_indices=[28,29,30,31],matched_control_seed=20261127)
    files=set(p13['code_hashes']);files.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts/functional_v14').glob('*.py'))
    cache=read(EXP/'batch-cache.json')
    write(EXP/'training-protocol.json',dict(at=stamp(),status='actual_data_runtime_optimizer_and_functional_gates_frozen_before_training',
        hypothesis='Adapt only final4 visual blocks in addition to inherited q/v+merger; test image-dependent direction/strength and quality against exact matched R13 controls. Not a new editor,prompt,resolution or dataset.',
        routes=list(ROUTES),origin='base',origin_identity=verify_origin(),parent_adapter=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        parent_status='R11 failed practical candidate;15/20 native,4/6 old,20/20 controls,77/100 controlled with1 major.',
        training=settings,dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),admission_sha256=sha(DATA/'admission.json'),
        input_audit_path=p13['input_audit_path'],input_audit_sha256=p13['input_audit_sha256'],target_replay_path=p13['target_replay_path'],target_replay_sha256=p13['target_replay_sha256'],
        batch_cache_path=cache['cache_path'],batch_cache_sha256=cache['cache_sha256'],evaluation_protocol_sha256=sha(EXP/'evaluation-protocol.json'),
        code_hashes={p:sha(ROOT/p) for p in sorted(files)},editor=EDITOR_VERSION,schema=SCHEMA_VERSION,runtime='functional_v14.runtime',
        system_prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),source_counts=read(DATA/'admission.json'),
        shared_data_audit_sha256=sha(EXP/'shared-data-audit.json'),architecture_sha256=sha(EXP/'installed-vision-architecture.json'),
        resource=dict(shared_cap_GPU_hours=24,consumed_at_freeze=used,remaining_at_freeze=cap-used,allowed_GPUs=[0,2,3],preserve_GPU1=True,
            main_training_max_GPU_hours=2,development_max_GPU_hours=.6,conditional_replication_reserve=2,acceptance_and_trial_reserve=.7,
            on_limit='Save and stop at actual cumulative23.9 threshold/24 hard cap; never change GPU1.')))
    print('Frozen:925/1850 same data,156tensors,visual28..31,2epochs,unchanged gates;GPUh remaining',cap-used)


if __name__=='__main__':main()
