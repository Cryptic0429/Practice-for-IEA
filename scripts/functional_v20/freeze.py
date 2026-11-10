"""Freeze a limited pilot before features, CPU training or new functional outputs."""
import shutil
from .common import ROOT, EXP, DATA, PARENT, PARENT_SHA, read, write, sha, stamp, GPUJob

def main():
    assert not (EXP/'training-protocol.json').exists()
    prior=ROOT/'experiments/functional-v19-20261107'
    old=read(prior/'training-protocol.json')
    evaluation=read(prior/'evaluation-protocol.json')
    src=ROOT/'data/training/functional-v19-20261107'
    DATA.mkdir(parents=True,exist_ok=True)
    for name in ('cases.json','train-accepted.json','training-pool.json'):
        shutil.copy2(src/name,DATA/name)
    assert sha(DATA/'cases.json')==old['cases_sha256']
    assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
    cases=read(DATA/'cases.json')
    train_groups={c['group_id'] for c in cases}
    for key,source in evaluation['case_files'].items():
        assert sha(ROOT/source['path'])==source['sha256']
        assert not train_groups & {c['group_id'] for c in read(ROOT/source['path'])},key
    sources={'train':dict(path=str((DATA/'cases.json').relative_to(ROOT)),sha256=sha(DATA/'cases.json'))}
    for dest,key in [('old_native','old_native_regressions'),('native_development','native_development'),
                     ('decision_development','decision_development'),('controlled_development','controlled_development')]:
        sources[dest]=evaluation['case_files'][key]
    ledger=read(ROOT/'experiments/functional-v1-20261104/resources.json')
    used=GPUJob.consumed(ledger)
    assert used<23.2
    code_hashes=dict(old['code_hashes'])
    for path in (ROOT/'scripts/functional_v20').glob('*.py'):
        code_hashes[str(path.relative_to(ROOT))]=sha(path)
    settings=dict(epochs=80,checkpoints=[20,80],seed=20261203,replicate_seed=20261204,
                  batch_semantics=16,paired_language_inputs=32,learning_rate=.001,weight_decay=.01,
                  positive_presentations_per_epoch=1850,steps_per_epoch=58,
                  objective='4-action CE + per-tool3-direction weighted CE (TRAIN sqrt class weights cap5) +4 active unsigned SmoothL1(beta.1) +.25 paired ZH/EN JS(action/tool) and active magnitude MSE.',
                  decoder='Argmax action and off/negative/positive per tool; sigmoid magnitude*100 integer rounding; zero omitted. Sepia negative masked. Empty edit is failure, never keep/decline fallback.',
                  normalization='Per-record hidden RMS then TRAIN-only mean/std floor.1 on hidden+16 current-input RGB stats.')
    value=dict(at=stamp(),status='frozen_before_7B_features_or_head_training',routes=['last','request_mean'],
         hypothesis='Separate action/direction from unsigned strength and regularize bilingual consistency on stronger frozen R15 features. Compare last prefix vs literal request-token mean from the same forward. Pilot does not isolate all differences versus failed R5.',
         parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,parent_identity='Actual R15e1, path named by earlier starting point; frozen116 LoRA tensors.',
         feature_sources=sources,training=settings,architecture='Frozen original NF4 Qwen+frozen R15e1;3584 hidden +16 RGB ->128GELU/dropout.1 ->4actions+9x3directions+9unsignedmagnitudes. CPU heads only.',
         input_policy='Unchanged R15 literal image/request/tools prefix. One current photo only. Both poolings from one target-free causal forward; no reference,label,caseID,expected action,task family or review notes encoded. 16RGB stats same frozen R19 function.',
         data=dict(semantic_cases=925,language_inputs=1850,original_groups=240,portrait_groups=120,landscape_groups=120,
                   edit=780,keep=80,decline=65,original_English=154,AI_authored=771,new_photos=0,new_requests=0,new_positive_labels=0,
                   resampling='Each semantic case once per epoch, adjacent ZH/EN; no preference repeat.'),
         old_R5_difference='R5 used R2 parent,370/740/60groups with10native requests,9gates+signedtanh andno explicit outcome classes. R20 has925/1850/240 with154native,4actions,3direction classes,unsigned strength andpaired consistency; last-vs-request mean is matched within R20.',
         evaluation=dict(first_screen='All4 fixed checkpoints on old6. Require both-language completion>=4/6 andzero major before native20+decision20.',
             stage1='Native>=16/20,old>=4/6,controls20/20,zero major. Only qualifying checkpoints may run controlled100; require>=81 andzero major.',
             selection='Completed both languages with acceptable quality; native, controlled, blindpaired comparison, earlier checkpoint. No L1/loss/parameter error selection.',
             reviews='AI-assisted direct Codex review; blind model identity, full-size supplement where needed; preserve identical-pixel historical disputes as uncertain.',
             acceptance='Closed until selected route/checkpoint locked and replicate seed passes same gates; no acceptance-based edits.'),
         resources=dict(cap_GPU_hours=24,occupied_at_freeze=used,remaining=24-used,allowed_GPU=[0,2,3],GPU1_preserved=True,
             initial_cache_max_seconds=1500,reserve='Keep at least .25GPUh for online trial/verification; do not start extra GPU work if reserve endangered.',
             head_training='CPU only; fixed80epochs x2routes. Feature extraction, actual online inference and all failures charged to shared ledger.'),
         previous_evaluation_protocol=dict(path=str((prior/'evaluation-protocol.json').relative_to(ROOT)),sha256=sha(prior/'evaluation-protocol.json')),
         code_hashes=code_hashes,L1_used_for_selection=False,independent_user_feedback=0)
    write(EXP/'training-protocol.json',value)
    write(DATA/'source-manifest.json',dict(at=stamp(),source_round='R19 unchanged reviewed labels',
        files={n:sha(DATA/n) for n in ('cases.json','train-accepted.json','training-pool.json')},data=value['data'],new_visual_admission=False))
    print('Frozen R20; CPU contracts next, then optional GPU cache. Remaining',24-used)

if __name__=='__main__':main()
