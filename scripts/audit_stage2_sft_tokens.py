"""CPU-only target-prefix and truncation audit of all512 shared training rows."""
import hashlib
import json
from datetime import datetime,timezone
import torch
from stage2_runtime import ROOT,processor
from train_stage2_sft_control import build_batch


def main():
    out=ROOT/'logs/stage2-A-edit-grpo-v1/sft-input-audit.json';assert not out.exists()
    source=ROOT/'data/stage2/stage2-A-edit-grpo-v1/train.json'
    manifest=json.loads((source.parent/'manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files']['train']['sha256']
    rows=json.loads(source.read_text());assert len(rows)==512
    torch.set_num_threads(4);p=processor();records=[]
    for i,row in enumerate(rows):
        batch=build_batch(p,row)
        records.append(dict(id=row['id'],input_tokens=batch['input_ids'].numel(),
            supervised_tokens=int((batch['labels']!=-100).sum()),
            token_sha256=hashlib.sha256(batch['input_ids'].numpy().tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(batch['labels'].numpy().tobytes()).hexdigest()))
        if (i+1)%64==0:
            print('AUDITED',i+1,flush=True)
    report=dict(passed=True,at=datetime.now(timezone.utc).isoformat(),rows=512,model_forwards=0,
        source_sha256=manifest['files']['train']['sha256'],max_input_tokens=max(r['input_tokens'] for r in records),
        input_tokens=sum(r['input_tokens'] for r in records),
        supervised_tokens=sum(r['supervised_tokens'] for r in records),records=records)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='records'})


if __name__=='__main__':
    main()
