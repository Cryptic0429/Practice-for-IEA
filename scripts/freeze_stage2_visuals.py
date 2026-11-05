"""Preselect visual cases without model predictions, and fix a per-case blind order."""
import hashlib
import json
import random
from pathlib import Path

ROOT=Path('/mnt/sdc/zhc/IEA')


def main():
    out=ROOT/'configs/stage2-visual-plan.json'
    assert not out.exists()
    rng=random.Random(20261101)
    cases=[]
    for split,count in [('test_new',20),('test_regression',10)]:
        source=ROOT/'data/stage2/stage2-A-edit-grpo-v1'/f'{split}.json'
        rows=json.loads(source.read_text())
        for row in rng.sample(rows,count):
            models=['A','SFT','GRPO'];rng.shuffle(models)
            cases.append(dict(case=len(cases)+1,split=split,id=row['id'],group_id=row['group_id'],
                              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              anonymous_labels=['X','Y','Z'],model_order=models))
    result=dict(status='frozen_before_stage2_predictions',seed=20261101,cases=cases,
                observations=['Instruction direction','Edit strength','Unrelated changes','No-op or failure'],
                procedure='Render original/reference and X/Y/Z. Save observations before decoding model_order.',
                limitation='AI-assisted qualitative review; not a controlled human blind study.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('FROZEN',len(cases),out)


if __name__=='__main__':
    main()
