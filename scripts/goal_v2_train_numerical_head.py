"""CPU fitting on frozen train inputs; validation is evaluated only after recipe lock."""
import argparse,hashlib,json,math,os,random,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from goal_v2_numerical_head import ActionHead,normalized,decode,objective
from goal_v2_json_runtime import TOOLS
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['64','512'],required=True);ap.add_argument('--seed',type=int,default=20261103);a=ap.parse_args()
    name=f'numeric-head{a.scope}-s{a.seed}';out=OUT/name;out.mkdir(exist_ok=False)
    status={'status':'loading','pid':os.getpid()};start=time.monotonic();write(out/'status.json',status)
    try:
        torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
        cache=OUT/'frozen-A-features-v1';assert json.loads((cache/'status.json').read_text())['status']=='completed'
        blob=np.load(cache/'train.npz');all_ids=blob['ids'].tolist();assert len(set(all_ids))==512
        data=ROOT/'data/goal-v2-20261103';filename='train-memorize64.json' if a.scope=='64' else 'train-contrast-calibrated.json'
        rows=json.loads((data/filename).read_text());assert len(rows)==int(a.scope)
        variants=json.loads((data/('training-variants-memorize.json' if a.scope=='64' else 'training-variants.json')).read_text())
        assert sha(data/filename)==variants['files'][Path(filename).stem]['sha256']
        indices=[all_ids.index(r['id']) for r in rows]
        assert all(blob['groups'][i]==r['group_id'] for i,r in zip(indices,rows))
        x=torch.tensor(blob['features'][indices],dtype=torch.float32);y=torch.tensor([[r['extra_info']['label'].get(k,0) for k in TOOLS] for r in rows],dtype=torch.float32)
        mean=x.mean(0);std=x.std(0).clamp_min(.02);x=normalized(x,mean,std)
        model=ActionHead(x.shape[1]);active=y!=0
        with torch.no_grad():
            frequency=active.float().mean(0).clamp(.02,.98)
            model.gates.bias.copy_(torch.logit(frequency));model.gates.weight.zero_()
            model.values.bias.copy_((y.sum(0)/active.sum(0).clamp_min(1))/25);model.values.weight.zero_()
        steps=1000 if a.scope=='64' else 640;batch=64;lr=1e-3
        config={'name':name,'seed':a.seed,'training_rows':len(rows),'unique_groups':len({r['group_id'] for r in rows}),
                'training_source_sha256':sha(data/filename),'frozen_feature_identity':json.loads((cache/'identity.json').read_text()),
                'feature_train_sha256':sha(cache/'train.npz'),'feature_validation_sha256':sha(cache/'validation.npz'),
                'steps':steps,'batch':batch,'lr':lr,'weight_decay':.01,'hidden':128,'dropout':.1,
                'gating_threshold':.5,'parameter_scale':25,'training_file':filename,'validation_used_for_training':False,
                'purpose':'Training-fit diagnostic' if a.scope=='64' else 'Bounded numerical-action prediction pilot',
                'feature_normalization':'Fit on selected training rows only; std floor .02; clamp normalized features to [-8,8]'}
        write(out/'run-config.json',config)
        for path in [__file__,str(ROOT/'scripts/goal_v2_numerical_head.py'),str(ROOT/'scripts/goal_v2_json_runtime.py')]:shutil.copyfile(path,out/Path(path).name)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
        order=torch.randperm(len(rows));cursor=0;assert len(rows)%batch==0
        for step in range(1,steps+1):
            if cursor>=len(rows):order=torch.randperm(len(rows));cursor=0
            ix=order[cursor:cursor+batch];cursor+=batch;model.train();optimizer.zero_grad(set_to_none=True)
            g,v=model(x[ix]);loss=objective(g,v,y[ix]);assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert math.isfinite(norm)
            optimizer.step();scheduler.step()
            if step%20==0:
                model.eval()
                with torch.no_grad():g,v=model(x);pred=decode(g,v)
                metrics={'step':step,'loss':float(loss),'train_mae16':float((pred-y).abs().mean()),'within_half_unit_vectors':int(((pred-y).abs().max(1).values<.5).sum()),'active_f1':float((2*((pred!=0)&active).sum())/((pred!=0).sum()+active.sum()).clamp_min(1))}
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
                status.update(status='running',step=step,elapsed_seconds=time.monotonic()-start,last_metrics=metrics);write(out/'status.json',status)
        model.eval()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        saved={'model':model.state_dict(),'mean':mean,'std':std,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'config':config}
        torch.save(saved,out/'head.pt')
        reloaded=torch.load(out/'head.pt',weights_only=False,map_location='cpu');check=ActionHead(x.shape[1]);check.load_state_dict(reloaded['model']);check.eval()
        with torch.no_grad():
            a0,b0=model(x[:2]);a1,b1=check(x[:2]);assert torch.equal(a0,a1) and torch.equal(b0,b1)
            pred=decode(*model(x))
        write(out/'train-predictions.json',{'tools':TOOLS,'ids':[r['id'] for r in rows],'parameters':pred.tolist()})
        write(out/'final-identity.json',{'head_sha256':sha(out/'head.pt'),'reload_forward_exact':True,'steps':steps,'training_mae16':float((pred-y).abs().mean())})
        status['status']='completed'
    except BaseException as e:status.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:status['elapsed_seconds']=time.monotonic()-start;write(out/'status.json',status)
if __name__=='__main__':main()
