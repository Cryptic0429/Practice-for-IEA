"""Train-only single-coordinate label search with the original frozen editor."""
import hashlib,json,re,time
from pathlib import Path
import cv2
import numpy as np
from evaluation_core import Scorer,sample_seed,parse_parameters
from goal_v2_analyze_validation import directions,features,THRESHOLD
ROOT=Path('/mnt/sdc/zhc/IEA');DATA=ROOT/'data/goal-v2-20261103';OUT=ROOT/'outputs/goal-v2-20261103/contrast-label-search'
GRID=[-60,-40,-20,0,20,40,60]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2);start=time.monotonic()
    source=DATA/'train.json';manifest=json.loads((DATA/'manifest.json').read_text());assert sha(source)==manifest['files']['train']['sha256']
    rows=json.loads(source.read_text());protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text());s=Scorer(ROOT,protocol)
    newrows=[];records=[]
    for row in rows:
        original=s.load(row['images'][0]);ref=s.load(row['extra_info']['reference_image_path']);editor=s.editor(row['images'][0]);label=row['extra_info']['label']
        seed=sample_seed(row['id'],protocol['seed']);np.random.seed(seed);oldimage=editor.process_image(label)
        oldloss=s.diff(oldimage,ref)['pixel_L1'];oldchange=float(np.abs(oldimage.astype(np.float32)-original.astype(np.float32)).mean()/255.)
        requested=directions(row['extra_info']['instruction']);f0=features(original);fold=features(oldimage) if requested else {}
        protected={k:sign for k,sign in requested.items() if sign*(fold[k]-f0[k])>THRESHOLD}
        text=row['extra_info']['instruction'].lower();contrast_sign=0
        if re.search(r'increase (?:the )?contrast|more contrast',text):contrast_sign=1
        if re.search(r'decrease (?:the )?contrast|reduce (?:the )?contrast|less contrast',text):contrast_sign=-1
        oldstd=float(cv2.cvtColor(oldimage,cv2.COLOR_BGR2GRAY).std()/255.);originalstd=float(cv2.cvtColor(original,cv2.COLOR_BGR2GRAY).std()/255.)
        protect_contrast=contrast_sign*(oldstd-originalstd)>THRESHOLD
        best=dict(label);bestloss=oldloss;bestchange=oldchange
        for value in GRID:
            if value==label.get('contrast',0):continue
            candidate=dict(label,contrast=value);np.random.seed(seed);image=editor.process_image(candidate)
            loss=s.diff(image,ref)['pixel_L1']
            if loss>=bestloss-1e-6:continue
            change=float(np.abs(image.astype(np.float32)-original.astype(np.float32)).mean()/255.)
            if oldchange>=1/255 and change<1/255:continue
            if protected:
                f=features(image)
                if any(sign*(f[k]-f0[k])<=THRESHOLD for k,sign in protected.items()):continue
            if protect_contrast and contrast_sign*(float(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).std()/255.)-originalstd)<=THRESHOLD:continue
            best=candidate;bestloss=loss;bestchange=change
        result=json.loads(json.dumps(row));result['extra_info']['label']=best
        for message in result['conversations']:
            if message['role']=='assistant':
                assert len(re.findall(r'```json\s*.*?```',message['content'],re.S))==1
                message['content']=re.sub(r'```json\s*.*?```',lambda _: '```json\n'+json.dumps(best,indent=4)+'\n```',message['content'],count=1,flags=re.S)
                parsed=parse_parameters(message['content'],protocol);assert parsed['valid'] and parsed['parameters']==best
        newrows.append(result);record={'id':row['id'],'group_id':row['group_id'],'old_l1':oldloss,'new_l1':bestloss,
            'old_contrast':label.get('contrast',0),'new_contrast':best.get('contrast',0),'old_edit_l1':oldchange,'new_edit_l1':bestchange,
            'protected_directions':protected,'contrast_direction_protected':protect_contrast}
        records.append(record)
        with (OUT/'search.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        (OUT/'status.json').write_text(json.dumps({'status':'running','completed':len(records),'total':len(rows),'elapsed_seconds':time.monotonic()-start})+'\n')
    destination=DATA/'train-contrast-calibrated.json';assert not destination.exists()
    destination.write_text(json.dumps(newrows,ensure_ascii=False,indent=2)+'\n')
    report={'status':'completed','rows':len(rows),'source_sha256':sha(source),'training_sha256':sha(destination),
            'changed_labels':sum(r['new_contrast']!=r['old_contrast'] for r in records),
            'old_mean_l1':float(np.mean([r['old_l1'] for r in records])),'new_mean_l1':float(np.mean([r['new_l1'] for r in records])),
            'grid':GRID,'elapsed_seconds':time.monotonic()-start,
            'editor_sha256':sha(ROOT/'repo/src/image_editor.py'),
            'limits':['Training-only label fitting, not a predictive-model result.','Other tools fixed; original frozen editor used.',
                      'Previously aligned explicit global directions and substantive edit threshold protected; broader semantics not guaranteed.',
                      'Reasoning text retained as context only; this variant must use parameter-only loss because reasoning may no longer match the new parameter values.']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'status.json').write_text(json.dumps({'status':'completed','completed':len(rows)})+'\n')
    variant={'files':{'train-contrast-calibrated':{'path':str(destination),'sha256':sha(destination),'rows':len(rows),'source_sha256':sha(source)}},'search_report_sha256':sha(OUT/'summary.json')}
    (DATA/'training-variants.json').write_text(json.dumps(variant,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
