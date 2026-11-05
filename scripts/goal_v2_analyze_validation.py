"""Paired development statistics and conservative image-direction proxies; validation only."""
import fcntl,hashlib,json,re
from pathlib import Path
import cv2
import numpy as np
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
ORIGINALS=['A','B','SFT','GRPO']
THRESHOLD=.005
PATTERNS={
 'brightness':([r'\bbrighten\b',r'\bbrighter\b',r'\blighten\b',r'increase (?:the )?brightness'],
               [r'\bdarken\b',r'\bdarker\b',r'decrease (?:the )?brightness',r'reduce (?:the )?brightness']),
 'warmth':([r'\bwarmer\b',r'add (?:some )?warmth',r'increase (?:the )?warmth',r'\bwarm tone'],
           [r'\bcooler\b',r'\bcool tone',r'reduce (?:the )?warmth']),
 'saturation':([r'increase (?:the )?saturation',r'\bmore vibrant\b',r'\bmore saturated\b'],
               [r'decrease (?:the )?saturation',r'reduce (?:the )?saturation',r'\bless saturated\b',r'\bdesaturate\b'])}
def directions(text):
    text=text.lower()
    if re.search(r"\b(not|don't|without|avoid|unchanged|keep|face|sky|background|bottom|top|person|hair|eyes|skin|foreground|line)\b",text):return {}
    result={}
    for feature,(positive,negative) in PATTERNS.items():
        pos=any(re.search(p,text) for p in positive);neg=any(re.search(p,text) for p in negative)
        if pos!=neg:result[feature]=1 if pos else -1
    return result
def features(image):
    b,g,r=image.astype(np.float32).mean((0,1))/255.
    saturation=float(cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1].mean()/255.)
    return {'brightness':float(.114*b+.587*g+.299*r),'warmth':float(r-b),'saturation':saturation}
def main():
    cv2.setNumThreads(2)
    val=json.loads((ROOT/'data/goal-v2-20261103/validation.json').read_text());order=[r['id'] for r in val]
    assert len(order)==len(set(order))==200 and len({r['group_id'] for r in val})==200
    lookup={r['id']:r for r in val};analysis=OUT/'validation-analysis';analysis.mkdir(exist_ok=True)
    available={};summaries={};indices=np.random.default_rng(20261103).integers(0,200,size=(10000,200))
    for directory in sorted((OUT/'evaluation').glob('validation-*')):
        state=directory/'status.json'
        if not state.exists() or json.loads(state.read_text())['status']!='completed':continue
        source=directory/'predictions.jsonl';digest=hashlib.sha256(source.read_bytes()).hexdigest();name=directory.name.removeprefix('validation-')
        records=[json.loads(l) for l in source.open()];byid={r['id']:r for r in records}
        assert len(records)==len(byid)==200 and set(byid)==set(order)
        records=[byid[i] for i in order]
        run_identity=json.loads((directory/'run.json').read_text())['identity']
        l1=np.array([r['scoring']['edited_metrics']['pixel_L1'] for r in records]);assert np.isfinite(l1).all()
        available[name]=l1
        cache=analysis/f'{name}-direction-features.json'
        if cache.exists():
            saved=json.loads(cache.read_text());assert saved['source_sha256']==digest
        else:
            rows=[]
            for rec in records:
                row=lookup[rec['id']];original=cv2.imread(row['images'][0]);edited=cv2.imread(rec['scoring']['edited_image_path'])
                assert original is not None and edited is not None and original.shape==edited.shape
                change=float(np.abs(original.astype(np.float32)-edited.astype(np.float32)).mean()/255.)
                wanted=directions(row['extra_info']['instruction']);f0=features(original);f1=features(edited)
                checks={k:{'expected_sign':sign,'delta':f1[k]-f0[k],
                    'classification':'same_direction' if sign*(f1[k]-f0[k])>THRESHOLD else ('opposite_direction' if sign*(f1[k]-f0[k])<-THRESHOLD else 'no_clear_change')}
                    for k,sign in wanted.items()}
                rows.append({'id':row['id'],'group_id':row['group_id'],'edit_l1':change,'pixel_identical':bool(np.array_equal(original,edited)),
                             'near_identity':change<1/255,'direction_checks':checks})
            saved={'source_sha256':digest,'threshold':THRESHOLD,'rows':rows,
                   'limits':'Global direction proxies cannot establish full instruction following, strength adequacy, local changes or aesthetics.'}
            cache.write_text(json.dumps(saved,indent=2)+'\n')
        rows=saved['rows'];assessed=[r for r in rows if r['direction_checks']]
        aligned=sum(all(c['classification']=='same_direction' for c in r['direction_checks'].values()) for r in assessed)
        summaries[name]={'n':200,'mean_l1':float(l1.mean()),'NoEdit_l1':float(np.mean([r['scoring']['baseline_metrics']['pixel_L1'] for r in records])),
                         'executable':sum(r['scoring']['editor_success'] for r in records),'fallback':sum(r['scoring']['fallback_used'] for r in records),
                         'empty':sum(r.get('empty_parameters',False) for r in records),'zero':sum(r.get('all_zero_parameters',False) for r in records),
                         'pixel_identical':sum(r['pixel_identical'] for r in rows),'near_identity':sum(r['near_identity'] for r in rows),
                         'mean_input_output_l1':float(np.mean([r['edit_l1'] for r in rows])),
                         'direction_assessable':len(assessed),'all_assessed_directions_aligned':aligned,'source_sha256':digest,
                         'json_prompt_signature':run_identity.get('goal_v2_json_runtime.py')}
    ready=[n for n in ORIGINALS if n in available]
    best=min(ready,key=lambda n:available[n].mean()) if ready else None
    comparisons={}
    if best:
        baseline=available[best]
        for name,l1 in available.items():
            delta=baseline-l1;boot=delta[indices].mean(1);relative=boot/baseline[indices].mean(1)
            comparisons[name]={'reference':best,'paired_mean_gain':float(delta.mean()),'relative_mean_gain':float(delta.mean()/baseline.mean()),
                               'ci95_gain':np.quantile(boot,[.025,.975]).tolist(),'ci95_relative':np.quantile(relative,[.025,.975]).tolist(),
                               'improved':int((delta>1e-6).sum()),'worsened':int((delta<-1e-6).sum()),'ties':int((np.abs(delta)<=1e-6).sum())}
    matched={}
    for candidate_name in sorted(n for n in available if summaries[n]['json_prompt_signature'] and n not in ['A-json-schema','B-json-schema']):
        for control in ['A-json-schema','B-json-schema']:
            if control not in available:continue
            if summaries[candidate_name]['json_prompt_signature']!=summaries[control]['json_prompt_signature']:continue
            baseline=available[control];delta=baseline-available[candidate_name]
            boot=delta[indices].mean(1);relative=boot/baseline[indices].mean(1)
            matched[candidate_name+'__vs__'+control]={'candidate':candidate_name,'control':control,'paired_mean_gain':float(delta.mean()),
                'relative_mean_gain':float(delta.mean()/baseline.mean()),'ci95_gain':np.quantile(boot,[.025,.975]).tolist(),
                'ci95_relative':np.quantile(relative,[.025,.975]).tolist(),
                'candidate_executable':summaries[candidate_name]['executable'],'control_executable':summaries[control]['executable'],
                'candidate_direction_aligned':summaries[candidate_name]['all_assessed_directions_aligned'],
                'control_direction_aligned':summaries[control]['all_assessed_directions_aligned']}
    required=set(ORIGINALS+['full-lr1e-5-s20261103','parameters-lr1e-5-s20261103','parameters-lr3e-5-s20261103'])
    report={'status':'all_initial_models_complete' if required.issubset(available) else 'partial_completed_models_only','models':summaries,
            'best_original_validation_model':best,'comparisons':comparisons,'matched_json_prompt_comparisons':matched,
            'limits':['Development/validation results, with model selection and uncorrected comparisons; not confirmatory test evidence.',
                      'One initial seed; original-source groups are bootstrap units.','Direction proxies require broader anonymized visual review.']}
    tmp=analysis/'summary.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(analysis/'summary.json')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    lock=(OUT/'validation-analysis.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX);main()
