"""Native-editor evaluation of a trained numeric head; train probe or validation only."""
import argparse,datetime,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from goal_v2_numerical_head import predict_checkpoint
from goal_v2_json_runtime import TOOLS
from evaluation_core import Scorer,parse_parameters
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--split',choices=['train_probe','validation'],required=True);a=ap.parse_args()
    directory=OUT/a.name;assert json.loads((directory/'status.json').read_text())['status']=='completed'
    cp=torch.load(directory/'head.pt',weights_only=False,map_location='cpu');config=cp['config'];cache=Path(config.get('feature_cache',str(OUT/'frozen-A-features-v1')))
    assert sha(cache/'train.npz')==config['feature_train_sha256'] and sha(cache/'validation.npz')==config['feature_validation_sha256']
    split='train' if a.split=='train_probe' else 'validation';blob=np.load(cache/f'{split}.npz');ids=blob['ids'].tolist()
    data=ROOT/'data/goal-v2-20261103';source=data/('train-memorize64.json' if a.split=='train_probe' else 'validation.json');rows=json.loads(source.read_text())
    assert len(rows)==(64 if a.split=='train_probe' else 200)
    index=[ids.index(r['id']) for r in rows];assert all(blob['groups'][i]==r['group_id'] for i,r in zip(index,rows))
    torch.set_num_threads(2)
    begin=time.monotonic()
    with torch.inference_mode():
        parameters=predict_checkpoint(cp,blob['features'][index]).numpy()
    head_seconds=time.monotonic()-begin;assert parameters.shape==(len(rows),16) and np.isfinite(parameters).all()
    protocol=json.loads((ROOT/'configs/evaluation-protocol-v1.json').read_text())
    for path,digest in protocol['code_hashes'].items():assert sha(path)==digest
    out=OUT/('training-probe-evaluation' if a.split=='train_probe' else 'evaluation')/f'{a.split}-{a.name}';out.mkdir(exist_ok=False,parents=True)
    identity={'head_sha256':sha(directory/'head.pt'),'frozen_feature_sha256':sha(cache/f'{split}.npz'),'source_sha256':sha(source),
              'goal_v2_json_runtime.py':sha(ROOT/'scripts/goal_v2_json_runtime.py'),'predictor':'frozen A multimodal feature extractor plus trained structured numerical head',
              'inference_inputs':'original image and original user instruction only; no reference/label; cached features are input-only',
              'head_inference_seconds':head_seconds,'split':a.split,'head_kind':cp.get('model_kind','structured-mlp'),
              'evaluator_sha256':sha(__file__),'numerical_runtime_sha256':sha(ROOT/'scripts/goal_v2_numerical_head.py')}
    shutil.copyfile(__file__,out/'evaluator-source.py');shutil.copyfile(ROOT/'scripts/goal_v2_numerical_head.py',out/'numerical-runtime-source.py')
    write(out/'run.json',{'identity':identity,'pid':os.getpid(),'started_at':stamp()});status={'status':'running','pid':os.getpid(),'completed':0};write(out/'status.json',status)
    records=[];scorer=Scorer(ROOT,protocol);start=time.monotonic()
    try:
        for row,values in zip(rows,parameters):
            params={k:float(v) for k,v in zip(TOOLS,values)};response='```json\n'+json.dumps(params)+'\n```';parsed=parse_parameters(response,protocol);assert parsed['valid']
            rec={'id':row['id'],'group_id':row['group_id'],'model':a.name,'instruction':row['extra_info']['instruction'],'instruction_type':row['extra_info']['inst_type'],
                 'input_image':row['images'][0],'reference_image':row['extra_info']['reference_image_path'],'response_text':response,'inference_error':None,
                 'finish_reason':'numeric_head','parsing':parsed,'empty_parameters':False,'all_zero_parameters':all(v==0 for v in params.values())}
            rec['scoring']=scorer.score(row,parsed,out/'edited'/f'{hashlib.sha256(row["id"].encode()).hexdigest()[:20]}.png');assert not rec['scoring'].get('metric_error')
            with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            records.append(rec);status.update(completed=len(records),updated_at=stamp(),elapsed_seconds=time.monotonic()-start);write(out/'status.json',status)
        summary={'rows':len(records),'mean_l1':float(np.mean([r['scoring']['edited_metrics']['pixel_L1'] for r in records])),
                 'baseline_l1':float(np.mean([r['scoring']['baseline_metrics']['pixel_L1'] for r in records])),
                 'executable':sum(r['scoring']['editor_success'] for r in records),'all_zero':sum(r['all_zero_parameters'] for r in records),
                 'source_sha256':sha(source),'head_sha256':sha(directory/'head.pt')}
        write(out/'summary.json',summary);status['status']='completed';print(json.dumps(summary))
    except BaseException as e:status.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:status.update(updated_at=stamp(),elapsed_seconds=time.monotonic()-start);write(out/'status.json',status)
if __name__=='__main__':main()
