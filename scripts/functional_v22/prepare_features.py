"""Reuse exact old input features; encode only108 newly admitted input requests."""
import argparse,gc
from .common import ROOT,EXP,DATA,PARENT,PARENT_SHA,protocol,read,write,sha,stamp,GPUJob

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',choices=['0','2','3'],required=True);args=ap.parse_args();p=protocol()
    assert not (EXP/'features/train-manifest.json').exists()
    import torch
    from functional_v20.features import extract
    from functional_v15.runtime import load_model,processor
    from functional_v21.texture import features,NAMES
    torch.set_num_threads(4);cases=read(DATA/'cases.json');new=read(DATA/'new-cases.json')
    old_source=p['inherited_hidden_manifests']['train'];assert sha(ROOT/old_source['path'])==old_source['sha256']
    prior=read(ROOT/old_source['path']);assert prior['parent_sha256']==PARENT_SHA
    rows=[]
    for row in prior['rows']:
        case=cases[len(rows)//2];language=('zh','en')[len(rows)%2]
        assert (row['case_id'],row['language'])==(case['id'],language) and sha(ROOT/row['path'])==row['sha256']
        saved=torch.load(ROOT/row['path'],map_location='cpu',weights_only=True)
        assert saved['input_sha256']==case['input_sha256'] and saved['request']==case['instructions'][language]
        assert saved['protocol_sha256']==p['inherited_hidden_protocol_sha256']
        rows.append(dict(row,feature_protocol_sha256=saved['protocol_sha256'],inherited=True))
    assert len(rows)==1850 and len(new)==54
    ledger=read(ROOT/'experiments/functional-v1-20261104/resources.json');remaining=24-GPUJob.consumed(ledger);assert remaining>.30
    with GPUJob(args.gpu,'v22-reviewed-quality-training-input-cache',max_seconds=min(900,int((remaining-.25)*3600))) as budget:
        model=load_model('language_merger',PARENT);proc=processor();assert not any(w.requires_grad for w in model.parameters())
        for case in new:
            assert sha(ROOT/case['image'])==case['input_sha256']
            for language in ('zh','en'):
                budget.tick();path=EXP/'features/train-new'/f'{case["id"]}-{language}.pt'
                if path.exists():saved=torch.load(path,map_location='cpu',weights_only=True)
                else:
                    saved=dict(extract(model,proc,case,language),input_sha256=case['input_sha256'],request=case['instructions'][language],protocol_sha256=sha(EXP/'training-protocol.json'))
                    path.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,path)
                assert saved['input_sha256']==case['input_sha256'] and saved['request']==case['instructions'][language] and saved['protocol_sha256']==sha(EXP/'training-protocol.json')
                rows.append(dict(case_id=case['id'],language=language,path=str(path.relative_to(ROOT)),sha256=sha(path),feature_protocol_sha256=saved['protocol_sha256'],inherited=False))
                write(EXP/'features/status.json',dict(at=stamp(),status='running',new_completed=len(rows)-1850,new_total=108))
        del model,proc;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==1958
    write(EXP/'features/train-manifest.json',dict(at=stamp(),rows=rows,source_sha256=sha(DATA/'cases.json'),parent_sha256=PARENT_SHA,
        protocol_sha256=sha(EXP/'training-protocol.json'),labels_in_features=False,new_records=108,reused_records=1850))
    source=p['inherited_texture'];assert sha(ROOT/source['path'])==source['sha256'];texture=read(ROOT/source['path'])
    assert texture['implementation_sha256']==sha(ROOT/'scripts/functional_v21/texture.py')
    for case in new:texture['images'][case['image']]=dict(input_sha256=case['input_sha256'],values=features(ROOT/case['image']).tolist())
    texture.update(at=stamp(),new_input_paths=54,inherited_source=source,target_files_read=False);write(EXP/'texture-features.json',texture)
    write(EXP/'features/status.json',dict(at=stamp(),status='completed',new_completed=108,new_total=108,total_TRAIN_records=1958,acceptance_opened=False))
    print('108 new target-free input requests encoded;1850 exact TRAIN features reused;GPU unloaded.')

if __name__=='__main__':main()
