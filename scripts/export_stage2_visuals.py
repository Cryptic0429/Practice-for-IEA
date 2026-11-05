"""Export the30 preselected test cases with model identities kept private."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/mnt/sdc/zhc/IEA')
OUT=ROOT/'outputs/stage2-A-edit-grpo-v1'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan_path=ROOT/'configs/stage2-visual-plan.json'
    plan=read(plan_path)
    lock=read(ROOT/'configs/stage2-selected-models.json')
    assert lock['status']=='frozen' and len(plan['cases'])==30
    datasets={}
    sources={}
    source_hashes={}
    for split,count in [('test_new',100),('test_regression',200)]:
        source=ROOT/'data/stage2/stage2-A-edit-grpo-v1'/f'{split}.json'
        source_hashes[split]=sha(source)
        sources[split]={r['id']:r for r in read(source)}
        for model in ['A','SFT','GRPO']:
            folder=OUT/'evaluation'/f'{split}-{model}'
            status=read(folder/'status.json')
            summary=read(folder/'summary.json')
            assert status['status']=='completed' and status['completed']==summary['rows']==count
            assert summary['source_sha256']==source_hashes[split]
            assert summary['adapter_sha256']==lock['models'][model]['adapter_sha256']
            records=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
            assert len(records)==len({r['id'] for r in records})==count
            assert {r['id'] for r in records}==set(sources[split])
            datasets[(split,model)]={r['id']:r for r in records}
    bundle=OUT/'blinded-visual-bundle'
    bundle.mkdir(exist_ok=False)
    cases=[]
    provenance=[]
    for case in plan['cases']:
        number,split,sample=case['case'],case['split'],case['id']
        assert case['source_sha256']==source_hashes[split]
        row=sources[split][sample]
        folder=bundle/f'case-{number:02d}';folder.mkdir()
        paths=[('Original',Path(row['images'][0])),('Reference',Path(row['extra_info']['reference_image_path']))]
        for label,model in zip(case['anonymous_labels'],case['model_order']):
            record=datasets[(split,model)][sample]
            paths.append((label,Path(record['scoring']['edited_image_path'])))
        images=[]
        for index,(label,source) in enumerate(paths):
            target=folder/f'panel-{index:02d}{source.suffix.lower()}'
            shutil.copy2(source,target)
            assert sha(target)==sha(source)
            images.append(dict(label=label,path=str(target.relative_to(bundle)),sha256=sha(target)))
            provenance.append(dict(case=number,panel=label,source=str(source),sha256=sha(source)))
        instruction=next(m['content'] for m in row['conversations'] if m['role']=='user').replace('<image>','').strip()
        assert instruction
        cases.append(dict(case=number,split=split,id=sample,instruction=instruction,images=images))
    report=dict(role='blinded_test_visual_review',created_at=datetime.now(timezone.utc).isoformat(),
        case_count=30,cases=cases,plan_sha256=sha(plan_path),
        instruction_source='Actual model user message, image placeholder removed for display',
        image_policy='Byte-identical copies. Render full frames with shared display scaling and no color adjustments.',
        review_scope='AI-assisted qualitative inspection; not a controlled human blind study.')
    (bundle/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'blinded-visual-private-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(cases=30,panels=150,bundle=str(bundle))))


if __name__=='__main__':
    main()
