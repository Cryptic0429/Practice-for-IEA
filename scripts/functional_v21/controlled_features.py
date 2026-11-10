"""Gated input-only frozen-parent features; no acceptance access or new labels."""
import argparse,gc
from .common import ROOT,EXP,PARENT,PARENT_SHA,protocol,read,write,sha,stamp,GPUJob


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);args=ap.parse_args()
    p=protocol();gate=read(EXP/'stage1-gate.json');assert gate['eligible']
    assert gate['training_protocol_sha256']==sha(EXP/'training-protocol.json')
    for row in gate['results'].values():
        for evidence in row['evidence'].values():assert sha(ROOT/evidence['path'])==evidence['sha256']
    source=p['sources']['controlled_development'];assert sha(ROOT/source['path'])==source['sha256']
    cases=read(ROOT/source['path']);assert len(cases)==100
    assert not (EXP/'controlled-feature-binding.json').exists()
    resources=read(ROOT/'experiments/functional-v1-20261104/resources.json')
    remaining=resources['limit_gpu_hours']-GPUJob.consumed(resources)
    assert resources['limit_gpu_hours']==24 and remaining>.35,'Keep at least.25GPUh trial reserve'
    import torch
    from functional_v15.runtime import load_model,processor
    from functional_v20.features import extract
    from .texture import features,NAMES
    torch.set_num_threads(4);rows=[];textures={}
    cap=min(900,int((remaining-.25)*3600))
    with GPUJob(args.gpu,'v21-gated-controlled-input-feature-cache',max_seconds=cap) as budget:
        model=load_model('language_merger',PARENT);proc=processor()
        assert not any(w.requires_grad for w in model.parameters())
        for case in cases:
            assert sha(ROOT/case['image'])==case['input_sha256']
            for language in ('zh','en'):
                budget.tick();path=EXP/'features/controlled_development'/f'{case["id"]}-{language}.pt'
                if path.exists():
                    saved=torch.load(path,map_location='cpu',weights_only=True)
                    assert saved['input_sha256']==case['input_sha256'] and saved['request']==case['instructions'][language]
                    assert saved['protocol_sha256']==sha(EXP/'training-protocol.json')
                else:
                    saved=dict(extract(model,proc,case,language),input_sha256=case['input_sha256'],request=case['instructions'][language],
                        protocol_sha256=sha(EXP/'training-protocol.json'))
                    path.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,path)
                rows.append(dict(case_id=case['id'],language=language,path=str(path.relative_to(ROOT)),sha256=sha(path)))
                write(EXP/'features/controlled-status.json',dict(status='running',completed=len(rows),total=200,at=stamp()))
        del model,proc;gc.collect();torch.cuda.empty_cache()
    manifest=EXP/'features/controlled_development-manifest.json'
    write(manifest,dict(source_sha256=source['sha256'],rows=rows,protocol_sha256=sha(EXP/'training-protocol.json'),
        parent_sha256=PARENT_SHA,labels_in_features=False,new_parent_decision_inference=False))
    for c in cases:
        if c['image'] not in textures:textures[c['image']]=dict(input_sha256=c['input_sha256'],values=features(ROOT/c['image']).tolist())
    texture=EXP/'controlled-texture-features.json'
    write(texture,dict(at=stamp(),names=list(NAMES),images=textures,implementation_sha256=sha(ROOT/'scripts/functional_v21/texture.py'),target_files_read=False))
    write(EXP/'controlled-feature-binding.json',dict(training_protocol_sha256=sha(EXP/'training-protocol.json'),source_cases_sha256=source['sha256'],
        stage1_eligible_models=gate['eligible'],stage1_gate_sha256=sha(EXP/'stage1-gate.json'),
        manifest=dict(path=str(manifest.relative_to(ROOT)),sha256=sha(manifest)),texture=dict(path=str(texture.relative_to(ROOT)),sha256=sha(texture))))
    write(EXP/'features/controlled-status.json',dict(status='completed',completed=200,total=200,at=stamp(),acceptance_opened=False))
    print('Completed200 new controlled input features, acceptance sealed.')


if __name__=='__main__':main()
