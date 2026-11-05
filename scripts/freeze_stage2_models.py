"""Lock all three model identities after validation selection and independent reloads."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/mnt/sdc/zhc/IEA')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())


def main():
    out=ROOT/'configs/stage2-selected-models.json';assert not out.exists()
    formal=ROOT/'outputs/stage2-A-edit-grpo-v1/formal-v1'
    sft=ROOT/'outputs/stage2-A-edit-grpo-v1/sft-control'
    assert read(formal/'training-verification.json')['step']==128
    assert read(sft/'training-verification.json')['step']==128
    for run in [formal,sft]:
        reload=read(run/'reload-verification.json')
        assert reload['adapter_tensors_exact'] and reload['forward_exact']
    selection=read(formal/'validation-selection.json')
    assert {v['step'] for v in selection['validations']}=={64,128}
    best=min(selection['validations'],key=lambda x:(x['mean_l1'],x['step']))
    selected=read(formal/'selected-model.json');assert selected['step']==best['step']
    for item in selection['validations']:
        summary=read(Path(item['summary_path']))
        assert summary['rows']==200 and summary['mean_l1']==item['mean_l1']
        assert summary['adapter_sha256']==sha(Path(item['checkpoint'])/'adapter_model.safetensors')
    assert sha(formal/'selected/adapter_model.safetensors')==sha(Path(best['checkpoint'])/'adapter_model.safetensors')
    models={}
    for name,path in [('A',ROOT/'outputs/stage1-A-image-edit-3k-v1'),('SFT',sft/'final'),('GRPO',formal/'selected')]:
        models[name]=dict(adapter=str(path),adapter_sha256=sha(path/'adapter_model.safetensors'),
                          adapter_config_sha256=sha(path/'adapter_config.json'))
    assert models['A']['adapter_sha256']=='c40552634bfa9a55f2d8a1584361331b7c17b753a46f1f5efa659de459e16306'
    files=[ROOT/'scripts'/n for n in ['evaluate_stage2.py','stage2_runtime.py','evaluation_core.py']]
    files+=[ROOT/'configs/stage2-rl-installed.txt',ROOT/'data/stage2/stage2-A-edit-grpo-v1/manifest.json']
    result=dict(status='frozen',at=datetime.now(timezone.utc).isoformat(),models=models,
                selected_grpo_validation=best,evaluation_identity={str(p):sha(p) for p in files},
                generation=dict(do_sample=False,max_new_tokens=512,repetition_penalty=1.),
                test_use='New100 final test and old200 regression; no further model selection using either test')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
