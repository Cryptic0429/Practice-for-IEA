"""Training-only label replay, reference benefit, explicit-direction conflicts and bounded weights."""
import datetime,hashlib,json,os,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from goal_v2_analyze_validation import directions,features,THRESHOLD
from evaluation_core import sample_seed
ROOT=Path('/mnt/sdc/zhc/IEA');DATA=ROOT/'data/goal-v2-20261103';OUT=ROOT/'outputs/goal-v2-20261103/calibrated3k-quality-audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def init():
    global load_image,diff,ImageEditor,protocol
    cv2.setNumThreads(1);sys.path.insert(0,str(ROOT/'repo'));from src.utils import load_and_resize_image,compute_diff_score
    from src.image_editor import ImageEditor as Editor
    load_image=load_and_resize_image;diff=compute_diff_score;ImageEditor=Editor;protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
def score(row):
    original=load_image(row['images'][0]);reference=load_image(row['extra_info']['reference_image_path']);np.random.seed(sample_seed(row['id'],protocol['seed']))
    edited=ImageEditor(row['images'][0]).process_image(row['extra_info']['label']);before=diff(original,reference)['pixel_L1'];after=diff(edited,reference)['pixel_L1'];wanted=directions(row['extra_info']['instruction']);f0=features(original);f1=features(edited)
    checks={k:s*(f1[k]-f0[k]) for k,s in wanted.items()};opposed=any(v<-THRESHOLD for v in checks.values());gain=before-after
    weight=float(np.clip(gain/.05,.25,2.))
    if opposed:weight=min(weight,.25)
    return {'id':row['id'],'group_id':row['group_id'],'NoEdit_l1':before,'label_l1':after,'gain':gain,'direction_checks_signed_delta':checks,'clear_direction_opposition':opposed,'raw_quality_weight':weight}
def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic();source=DATA/'train3k-contrast-calibrated.json';variant=json.loads((DATA/'training-variants-3k.json').read_text());assert sha(source)==variant['files'][source.stem]['sha256'];rows=json.loads(source.read_text());assert len(rows)==3000
    search={x['id']:x for x in (json.loads(l) for l in (ROOT/'outputs/goal-v2-20261103/contrast-label-search3k/search.jsonl').open())};state={'status':'running','pid':os.getpid(),'completed':0,'total':3000,'workers':8};write(OUT/'status.json',state)
    records=[]
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
            for record in pool.map(score,rows,chunksize=2):
                assert abs(record['label_l1']-search[record['id']]['new_l1'])<1e-7
                records.append(record)
                with (OUT/'rows.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                if len(records)%16==0:state.update(completed=len(records),elapsed_seconds=time.monotonic()-start);write(OUT/'status.json',state)
        mean=float(np.mean([r['raw_quality_weight'] for r in records]));weights=[{'id':r['id'],'group_id':r['group_id'],'weight':r['raw_quality_weight']/mean} for r in records]
        target=DATA/'train3k-quality-weights.json';assert not target.exists();write(target,{'source_sha256':sha(source),'formula':'clip((NoEditL1-labelL1)/0.05,0.25,2); cap at0.25 for clear opposed global direction; normalize all-row mean to1; every row retained','rows':weights})
        report={'status':'completed','training_rows':3000,'source_groups':len({r['group_id'] for r in records}),'source_sha256':sha(source),'NoEdit_l1':float(np.mean([r['NoEdit_l1'] for r in records])),'label_l1':float(np.mean([r['label_l1'] for r in records])),'labels_improving_original':sum(r['gain']>1e-6 for r in records),'labels_worsening_original':sum(r['gain']<-1e-6 for r in records),'direction_assessable':sum(bool(r['direction_checks_signed_delta']) for r in records),'clear_direction_oppositions':sum(r['clear_direction_opposition'] for r in records),'quality_weights_sha256':sha(target),'mean_normalized_weight':float(np.mean([r['weight'] for r in weights])),'weight_range':[min(r['weight'] for r in weights),max(r['weight'] for r in weights)],'elapsed_seconds':time.monotonic()-start,'limits':['Training-only oracle/program replay; not model generalization.','Rows sharing a source group are dependent; counts are descriptive.','Global direction proxy excludes local/negated/ambiguous requests; not full semantics.','Weights never enter model inference, and do not use validation/test references.']}
        write(OUT/'summary.json',report);state['status']='completed';print(json.dumps(report,indent=2))
    except BaseException as e:state.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:state.update(completed=len(records),elapsed_seconds=time.monotonic()-start);write(OUT/'status.json',state)
if __name__=='__main__':main()
