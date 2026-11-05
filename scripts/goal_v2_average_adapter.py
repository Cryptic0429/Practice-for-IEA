"""Fixed50/50 average of LoRA parameter tensors from two checkpoints of the same run."""
import argparse,hashlib,json,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
ROOT=Path('/mnt/sdc/zhc/IEA')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--left',type=Path,required=True);ap.add_argument('--right',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    left=a.left.resolve();right=a.right.resolve();out=a.output.resolve();assert left.parent==right.parent and left.parent.is_relative_to(ROOT/'outputs/goal-v2-20261103')
    assert left.name=='checkpoint-512' and right.name=='checkpoint-1024';assert out.is_relative_to(ROOT)
    lc=json.loads((left/'adapter_config.json').read_text());rc=json.loads((right/'adapter_config.json').read_text());assert lc==rc and lc['r']==8 and lc['lora_alpha']==16
    x=load_file(str(left/'adapter_model.safetensors'));y=load_file(str(right/'adapter_model.safetensors'));assert x.keys()==y.keys() and len(x)==112
    assert all(x[k].shape==y[k].shape and 'lora_' in k and ('q_proj' in k or 'v_proj' in k) and 'visual' not in k for k in x)
    mean={k:((x[k].float()+y[k].float())*.5).contiguous() for k in x};assert all(torch.isfinite(t).all() for t in mean.values())
    out.mkdir(parents=True,exist_ok=False);shutil.copyfile(left/'adapter_config.json',out/'adapter_config.json');save_file(mean,str(out/'adapter_model.safetensors'),metadata={'format':'pt'})
    saved=load_file(str(out/'adapter_model.safetensors'));assert all(torch.equal(saved[k],mean[k]) for k in mean)
    report={'method':'Fixed arithmetic mean of all112 trainable LoRA A/B parameter tensors; same trajectory and basis, frozen base untouched. This is parameter averaging, not pixel/output mixing or exact average of network functions.','weights':[.5,.5],
            'sources':[{'path':str(p),'adapter_sha256':sha(p/'adapter_model.safetensors'),'config_sha256':sha(p/'adapter_config.json')} for p in [left,right]],'adapter_sha256':sha(out/'adapter_model.safetensors'),'tensors':112,'saved_tensors_exact':True,'dtype':'float32','training_state':'Raw checkpoints and optimizer/RNG remain separate; this derived inference adapter is not paired with the raw optimizer state.'}
    (out/'averaging.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(out),'adapter_sha256':report['adapter_sha256'],'tensors':112}))
if __name__=='__main__':main()
