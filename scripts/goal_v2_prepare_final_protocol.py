"""Freeze final-test methods and30 review IDs without inference or image scoring."""
import datetime,hashlib,json,random
from pathlib import Path
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_new(p,d):
    with p.open('x') as f:f.write(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
    assert not (OUT/'test-opened.json').exists() and not (OUT/'final-evaluation').exists()
    base=ROOT/'configs/evaluation-protocol-v1.json';baseline=OUT/'original-baselines-lock.json';acceptance=OUT/'acceptance-protocol.json'
    manifest=ROOT/'data/goal-v2-20261103/manifest.json';data=json.loads(manifest.read_text());entry=data['files']['test_sealed'];source=Path(entry['path']);assert sha(source)==entry['sha256']
    rows=json.loads(source.read_text());assert len(rows)==len({r['group_id'] for r in rows})==71
    chosen=random.Random(20261103).sample(list(range(71)),30);model_keys=['A','B','SFT','GRPO','Seed20261103','Seed20261104','Seed20261105']
    key={};cases=[];rng=random.Random(20261104)
    for number,index in enumerate(chosen,1):
        row=rows[index];order=model_keys.copy();rng.shuffle(order);key[str(number)]={f'Output{i+1}':name for i,name in enumerate(order)}
        cases.append({'case':number,'id':row['id'],'group_id':row['group_id']})
    visual=OUT/'final-visual-plan.json';decoding=OUT/'final-visual-key.private.json'
    save_new(decoding,{'case_mapping':key,'rule':'Do not decode until observations on all30 cases are saved.'})
    save_new(visual,{'selection':'Uniform seeded30 of71 source groups, fixed before predictions; no quality selection','seed':20261103,'cases':cases,'key_sha256':sha(decoding),'panels':['Input','Reference',*[f'Output{i+1}' for i in range(7)]],
                     'rubric':{'direction':'correct / partial / opposite / unassessable','strength':'adequate / excessive / insufficient / unassessable','unrelated_changes':'none / minor / major','local_requests':'explicitly note unsupported/local changes rather than global proxy pass','reference_match':'descriptive only; reference does not uniquely define aesthetics'},
                     'display':'Full frames, equal aspect-preserving letterboxes; no colour adjustment or crop','reviewer':'AI-assisted visual review; not a human user study or formal double-blind study'})
    paths=['scripts/goal_v2_final_evaluate.py','scripts/stage2_runtime.py','scripts/evaluation_core.py','scripts/goal_v2_json_runtime.py','scripts/goal_v2_analyze_validation.py','repo/src/image_editor.py','repo/src/utils.py','repo/src/config.py','configs/evaluation-protocol-v1.json']
    frozen={'status':'frozen_before_test_inference','frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_source':entry,'test_manifest_sha256':sha(manifest),'base_protocol_sha256':sha(base),'acceptance_protocol_sha256':sha(acceptance),'original_baselines_lock_sha256':sha(baseline),
            'code_hashes':{p:sha(ROOT/p) for p in paths},'models':{'baseline_keys':model_keys[:4],'candidate_keys':model_keys[4:],'baseline_format':'original','candidate_format':'json-only-schema-v1','generation':{'do_sample':False,'max_new_tokens':512,'repetition_penalty':1.,'num_return_sequences':1}},
            'primary':'Each of3 locked independent training seeds must improve full71 paired mean L1 by at least3% against every original frozen baseline; identify strongest baseline conservatively. All rows and failures retained.',
            'uncertainty':{'draws':10000,'seed':20261103,'unit':'source group','seed_mean':'Resample identical71 groups jointly across all3 seeds; do not pool213 seed-by-image rows as independent','report':'Per-seed paired intervals and seed-level range; inconclusive results are not proof of goal success'},
            'guardrails':{'execution':'Maintain strongest original baseline execution; report failures and common-executable comparison','nonoperation':'Report empty/zero, pixel identity, near identity<1/255, edit-L1 distribution and substantive-edit union comparison; do not manufacture gains via fallback or no-op','instruction':'Frozen global brightness/warmth/saturation proxies, unassessable requests excluded from pass counts; per-seed compared with strongest baseline; broader anonymized visual review','NoEdit':'Always report unedited original L1'},
            'visual_plan_sha256':sha(visual),'future_test_policy':'No test-driven model/recipe selection. If unsuccessful or inconclusive, inspected71 are regression data for future work, requiring a new independent confirmation set.','pending':'Final model identities and development gate must be frozen before this evaluator can run. No inference performed by preparation.'}
    save_new(OUT/'final-protocol.json',frozen);print(json.dumps({'status':frozen['status'],'groups':71,'visual_cases':30,'test_images_read_or_scored':False,'final_protocol_sha256':sha(OUT/'final-protocol.json')},indent=2))
if __name__=='__main__':main()
