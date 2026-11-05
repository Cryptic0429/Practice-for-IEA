"""Collect actual parent errors on TRAINING photos only; no automatic preference labels."""
import argparse,json,os
from .common import ROOT,ROUND2_DATA,ROUND2_EXP,EXP,PARENT,GPUJob,read,write,append,sha,stamp,EDITOR_VERSION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);args=ap.parse_args()
    protocol=read(EXP/'collection-protocol.json');cases=read(ROUND2_DATA/'train-final-cases.json')
    assert sha(ROUND2_DATA/'train-final-cases.json')==protocol['training_case_sha256']
    assert sha(PARENT/'adapter_model.safetensors')==protocol['parent_sha256']
    assert all(c['split']=='train' and c['source_id'].startswith('train-') for c in cases)
    for file,digest in protocol['core_code_hashes'].items():assert sha(ROOT/file)==digest
    out=EXP/'collection';out.mkdir(exist_ok=True);path=out/'predictions.jsonl'
    old=list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
    done={(r['case_id'],r['language']) for r in old};status={'status':'starting','pid':os.getpid(),'gpu':args.gpu,'at':stamp()}
    write(out/'status.json',status)
    if len(done)==2*len(cases):return
    with GPUJob(args.gpu,'v3-collect-training-policy-errors',max_seconds=5400) as budget:
        import torch
        from functional_v1.runtime import load_model,processor,predict
        from functional_v1.editor import load_rgb,edit,save
        torch.set_num_threads(4);model=load_model(PARENT);p=processor()
        for case in cases:
            for lang in ('zh','en'):
                if (case['id'],lang) in done:continue
                budget.tick();assert sha(ROOT/case['image'])==case['input_sha256']
                row={'case_id':case['id'],'language':lang,'source_id':case['source_id'],'domain':case['domain'],
                    'family':case['family'],'input_image':case['image'],'input_sha256':case['input_sha256'],
                    'instruction':case['instructions'][lang],'status':'failed','parent_sha256':protocol['parent_sha256']}
                try:
                    response,parameters=predict(model,p,case,lang);file=out/'images'/f'{case["id"]}-{lang}.png'
                    save(file,edit(load_rgb(ROOT/case['image']),parameters))
                    row.update(status='executed',parameters=parameters,response=response,output_image=str(file.relative_to(ROOT)),output_sha256=sha(file))
                except Exception as e:row['error']=f'{type(e).__name__}: {e}'
                row['at']=stamp();append(path,row);old.append(row);done.add((case['id'],lang))
                status.update(status='running',completed=len(done),total=len(cases)*2,at=stamp());write(out/'status.json',status)
        status.update(status='completed',finished_at=stamp(),executed=sum(r['status']=='executed' for r in old));write(out/'status.json',status)

if __name__=='__main__':main()
