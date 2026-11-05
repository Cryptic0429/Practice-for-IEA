"""Fivefold source-group CV on train512 chooses ridge strength; no validation selection."""
import hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from goal_v2_json_runtime import TOOLS
from goal_v2_numerical_head import predict_checkpoint
ROOT=Path('/mnt/sdc/zhc/IEA');OUT=ROOT/'outputs/goal-v2-20261103'
ALPHAS=[.01,.1,1.,10.,100.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def norm_fit(x):
    mean=x.mean(0);std=x.std(0,ddof=1).clip(.02);return mean,std
def transform(x,mean,std):return ((x-mean)/std).clip(-8,8)
def solve_eig(eig,target,alpha):
    vals,vec=eig;return vec@((vec.T@target)/(vals[:,None]+alpha)) if target.ndim==2 else vec@((vec.T@target)/(vals+alpha))
def fit_all(x,y):
    k=(x@x.T)/x.shape[1];active=(y!=0).astype(np.float64);freq=active.mean(0);eig=np.linalg.eigh(k)
    conditional=[]
    for j in range(16):
        ix=np.flatnonzero(active[:,j]);value=y[ix,j];mean=float(value.mean()) if len(ix) else 0.
        conditional.append((ix,mean,np.linalg.eigh(k[np.ix_(ix,ix)]) if len(ix) else (np.empty(0),np.empty((0,0))),value-mean))
    results=[]
    for alpha in ALPHAS:
        g=solve_eig(eig,active-freq,alpha)
        v=[{'indices':ix,'mean':mean,'coeff':solve_eig(eig0,target,alpha)} for ix,mean,eig0,target in conditional]
        results.append((g,freq,v))
    return results
def predict(x,xtrain,fitted):
    g,freq,values=fitted;k=(x@xtrain.T)/x.shape[1];gates=freq+k@g
    pred=np.stack([v['mean']+k[:,v['indices']]@v['coeff'] for v in values],axis=1)
    return np.where(gates>=.5,pred.clip(-100,100),0.)
def main():
    name='numeric-ridge512-cv-s20261103';out=OUT/name;out.mkdir(exist_ok=False);start=time.monotonic();status={'status':'running','pid':os.getpid()};write(out/'status.json',status)
    try:
        torch.set_num_threads(4);cache=OUT/'frozen-A-features-v1';assert json.loads((cache/'status.json').read_text())['status']=='completed'
        data=ROOT/'data/goal-v2-20261103';source=data/'train-contrast-calibrated.json';variants=json.loads((data/'training-variants.json').read_text());assert sha(source)==variants['files'][source.stem]['sha256']
        rows=json.loads(source.read_text());blob=np.load(cache/'train.npz');ids=blob['ids'].tolist();ix=[ids.index(r['id']) for r in rows]
        assert all(blob['groups'][i]==r['group_id'] for i,r in zip(ix,rows));x=blob['features'][ix].astype(np.float64)
        y=np.asarray([[r['extra_info']['label'].get(k,0) for k in TOOLS] for r in rows],np.float64);groups=np.asarray([r['group_id'] for r in rows]);unique=np.unique(groups);rng=np.random.default_rng(20261103);rng.shuffle(unique)
        folds={g:i%5 for i,g in enumerate(unique)};fold_index=np.asarray([folds[g] for g in groups]);oof=np.zeros((len(ALPHAS),len(rows),16));fold_reports=[]
        for fold in range(5):
            tr=fold_index!=fold;va=~tr;assert set(groups[tr]).isdisjoint(set(groups[va]));mean,std=norm_fit(x[tr]);xt=transform(x[tr],mean,std);xv=transform(x[va],mean,std)
            fits=fit_all(xt,y[tr]);scores=[]
            for j,fitted in enumerate(fits):oof[j,va]=predict(xv,xt,fitted);scores.append(float(np.abs(oof[j,va]-y[va]).mean()))
            fold_reports.append({'fold':fold,'train_groups':len(set(groups[tr])),'heldout_groups':len(set(groups[va])),'mae16_by_alpha':scores})
            write(out/'cv-progress.json',{'completed_folds':fold+1,'alphas':ALPHAS,'folds':fold_reports})
        scores=np.abs(oof-y[None]).mean((1,2));chosen=int(scores.argmin());alpha=ALPHAS[chosen];mean,std=norm_fit(x);xt=transform(x,mean,std);fitted=fit_all(xt,y)[chosen]
        report={'selection':'minimum out-of-source-group train512 MAE16; validation untouched during selection','fold_seed':20261103,'groups':len(unique),'alphas':ALPHAS,'oof_mae16':scores.tolist(),'selected_alpha':alpha,'folds':fold_reports}
        write(out/'cross-validation.json',report)
        g,freq,values=fitted
        config={'name':name,'seed':20261103,'training_rows':512,'training_source_sha256':sha(source),'feature_train_sha256':sha(cache/'train.npz'),'feature_validation_sha256':sha(cache/'validation.npz'),'feature_identity':json.loads((cache/'identity.json').read_text()),'selected_alpha':alpha,'gate_threshold':.5,'conditional_value_fit':True,'training_selection':'fivefold source-group CV on train512 only','validation_used_for_training':False}
        write(out/'run-config.json',config)
        cp={'model_kind':'conditional-kernel-ridge','mean':torch.tensor(mean,dtype=torch.float32),'std':torch.tensor(std,dtype=torch.float32),'train_features':torch.tensor(xt,dtype=torch.float64),'gate_mean':torch.tensor(freq),'gate_coeff':torch.tensor(g),'value_models':[{'indices':torch.tensor(v['indices']),'mean':torch.tensor(v['mean']),'coeff':torch.tensor(v['coeff'])} for v in values],'config':config}
        torch.save(cp,out/'head.pt')
        for p in [Path(__file__),ROOT/'scripts/goal_v2_numerical_head.py',ROOT/'scripts/goal_v2_evaluate_numerical_head.py']:shutil.copyfile(p,out/p.name)
        loaded=torch.load(out/'head.pt',weights_only=False,map_location='cpu')
        with torch.inference_mode():pred=predict_checkpoint(loaded,x).numpy()
        assert np.isfinite(pred).all();assert np.max(np.abs(pred-predict(xt,xt,fitted)))<.01
        write(out/'final-identity.json',{'head_sha256':sha(out/'head.pt'),'selected_alpha':alpha,'train_mae16':float(np.abs(pred-y).mean()),'oof_mae16':float(scores[chosen]),'serialization_forward_agreement_atol':.01,'deterministic_fit':True,'not_independent_training_seed_replication':True})
        status['status']='completed'
    except BaseException as e:status.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:status['elapsed_seconds']=time.monotonic()-start;write(out/'status.json',status)
if __name__=='__main__':main()
