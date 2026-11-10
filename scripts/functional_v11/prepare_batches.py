"""CPU-only cache; input whitelist and assistant-only labels checked before GPU use."""
import os
import torch
from functional_v1.common import ROOT,read,write,sha,stamp
from .runtime import EXP,DATA,processor,build_batch

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    path=EXP/'batches.pt'; assert not path.exists()
    admission=read(DATA/'admission.json'); assert admission['semantic_cases']==672
    rows=read(DATA/'train-accepted.json');cases={c['id']:c for c in read(DATA/'cases.json')}
    p=processor();prepared=[];tokens=[]
    for row in rows:
        c=cases[row['case_id']];assert sha(ROOT/c['image'])==c['input_sha256']
        batch=build_batch(p,c,row['language'],row['decision'])
        assert all(v.device.type=='cpu' for v in batch.values() if isinstance(v,torch.Tensor))
        prepared.append(dict(batch))
        tokens.append(dict(id=row['id'],action=row['decision']['action'],input_tokens=int(batch['input_ids'].numel()),assistant_tokens=int(batch['labels'].ne(-100).sum())))
    assert len(prepared)==1344
    torch.save(prepared,path)
    write(EXP/'batch-cache.json',dict(at=stamp(),status='CPU_only_prepared',dataset_sha256=sha(DATA/'train-accepted.json'),cases_sha256=sha(DATA/'cases.json'),cache_path=str(path.relative_to(ROOT)),cache_sha256=sha(path),rows=1344,tokens=tokens,model_forward_calls=0,GPU_hours=0))
    print('CPU cache:1344 rows, original inputs and assistant-only decisions; no GPU/model forward.')

if __name__=='__main__':main()
