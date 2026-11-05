"""Eight CPU workers calibrate only original3k training labels; editor unchanged."""
import copy,hashlib,json,os,re,time,traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from evaluation_core import Scorer,sample_seed,parse_parameters
from goal_v2_analyze_validation import directions,features,THRESHOLD
ROOT=Path('/mnt/sdc/zhc/IEA');DATA=ROOT/'data/goal-v2-20261103';OUT=ROOT/'outputs/goal-v2-20261103/contrast-label-search3k'
GRID=[-60,-40,-20,0,20,40,60]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def init():
    global s,protocol
    cv2.setNumThreads(1);protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text());s=Scorer(ROOT,protocol)
def calculate(row):
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
    record={'id':row['id'],'group_id':row['group_id'],'old_l1':oldloss,'new_l1':bestloss,
        'old_contrast':label.get('contrast',0),'new_contrast':best.get('contrast',0),'old_edit_l1':oldchange,'new_edit_l1':bestchange,
        'protected_directions':protected,'contrast_direction_protected':protect_contrast}
    return result,record

def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic();source=DATA/'train3k.json';manifest=json.loads((DATA/'training-variants-3k.json').read_text());assert sha(source)==manifest['files']['train3k']['sha256']
    rows=json.loads(source.read_text());assert len(rows)==3000
    raw512={r['id']:r for r in json.loads((DATA/'train.json').read_text())};cal512={r['id']:r for r in json.loads((DATA/'train-contrast-calibrated.json').read_text())}
    records512={r['id']:r for r in (json.loads(l) for l in (ROOT/'outputs/goal-v2-20261103/contrast-label-search/search.jsonl').open())}
    found={};records={};pending=[]
    for row in rows:
        if row['id'] in cal512:
            assert row==raw512[row['id']], 'Cached calibration source differs'
            found[row['id']]=cal512[row['id']];records[row['id']]=records512[row['id']]
        else:pending.append(row)
    assert len(found)==512
    status={'status':'running','pid':os.getpid(),'workers':8,'completed':512,'total':3000,'cached512_reused':True};write(OUT/'status.json',status)
    import shutil;shutil.copyfile(__file__,OUT/'source.py')
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
            for result,record in pool.map(calculate,pending,chunksize=2):
                assert record['new_l1']<=record['old_l1']+1e-6
                found[result['id']]=result;records[result['id']]=record
                with (OUT/'search-new.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                status.update(completed=len(found),elapsed_seconds=time.monotonic()-start)
                if len(found)%16==0:write(OUT/'status.json',status)
                if time.monotonic()-start>3600:raise TimeoutError('Bounded3k calibration exceeded one hour; partial records preserved')
        newrows=[found[r['id']] for r in rows];ordered=[records[r['id']] for r in rows];dest=DATA/'train3k-contrast-calibrated.json';assert not dest.exists()
        dest.write_text(json.dumps(newrows,indent=2,ensure_ascii=False)+'\n')
        (OUT/'search.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in ordered))
        report={'status':'completed','rows':3000,'groups':len({r['group_id'] for r in rows}),'source_sha256':sha(source),'training_sha256':sha(dest),'changed_labels':sum(r['old_contrast']!=r['new_contrast'] for r in ordered),'old_mean_l1':float(np.mean([r['old_l1'] for r in ordered])),'new_mean_l1':float(np.mean([r['new_l1'] for r in ordered])),'grid':GRID,'cached_rows':512,'workers':8,'editor_sha256':sha(ROOT/'repo/src/image_editor.py'),'elapsed_seconds':time.monotonic()-start,'limits':['Training-reference-based label fitting, not predictive model performance.','Same contrast-only search and direction/edit protections as prior512.','Original reference/label never enter model generation; reasoning must be masked or removed.']}
        write(OUT/'summary.json',report)
        manifest['files'][dest.stem]={'path':str(dest),'sha256':sha(dest),'rows':3000,'groups':report['groups'],'source_sha256':sha(source)};manifest['calibration_summary_sha256']=sha(OUT/'summary.json');write(DATA/'training-variants-3k.json',manifest)
        status['status']='completed'
    except BaseException as e:status.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:status['elapsed_seconds']=time.monotonic()-start;write(OUT/'status.json',status)
if __name__=='__main__':main()
