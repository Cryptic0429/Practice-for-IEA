"""Bounded integer-coordinate native crops for exceptionally large review images."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v13-20261106'


def prepare(bundle='epoch2-narrow'):
    from PIL import Image
    from functional_v1.common import read,write,sha,stamp
    assert bundle in ('epoch2-narrow','epoch2-wide')
    source=EXP/'review/native-development'/bundle/'full-resolution/manifest.json'
    item=next(r for r in read(source)['items'] if r['case_id']=='native-dev-032')
    dest=source.parent/'native-crops';assert not (dest/'manifest.json').exists()
    images=[]
    for image in item['images']:
        assert sha(ROOT/image['path'])==image['sha256']
        for label,box in [('sky-points',[1600,900,2800,2100]),('foreground',[1000,2500,2200,3700])]:
            path=dest/f'{label}-{image["role"]}.png';path.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(ROOT/image['path']) as im:im.convert('RGB').crop(box).save(path)
            images.append(dict(role=image['role'],region=label,path=str(path.relative_to(ROOT)),sha256=sha(path),
                source=image['path'],source_sha256=image['sha256'],source_size=image['size'],box=box,size=[1200,1200]))
    write(dest/'manifest.json',dict(at=stamp(),item_id=item['item_id'],case_id=item['case_id'],
        input_sha256=item['input_sha256'],output_sha256=item['output_sha256'],images=images,
        transformations='Integer crop only, no resizing or pixel adjustment. Whole-frame context remains on the anonymous panel.'))


def record(completion,quality,evidence,bundle='epoch2-narrow'):
    import hashlib,datetime
    assert bundle in ('epoch2-narrow','epoch2-wide')
    path=EXP/'review/native-development'/bundle/'full-resolution/native-crops/manifest.json'
    item=json.loads(path.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for image in item['images']:assert digest(ROOT/image['path'])==image['sha256']
    assert completion in ('C','P','F','U') and quality in ('A','m','M','U') and len(evidence)>30
    target=EXP/'review/native-development/codex-full-resolution-observations.jsonl'
    previous=target.read_text();assert item['item_id'] not in {json.loads(x)['item_id'] for x in previous.splitlines()}
    row=dict(item,completion=dict(C='completed',P='partial',F='failed',U='uncertain')[completion],
        quality=dict(A='acceptable',m='minor_issue',M='major_issue',U='uncertain')[quality],evidence=evidence,
        manifest_sha256=digest(path),reviewer='Codex direct anonymous native-pixel bounded-crop AI review; not human',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplements_raw_contact_sheet=True)
    target.write_text(previous+json.dumps(row,ensure_ascii=False)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bundle',default='epoch2-narrow');a=p.parse_args();prepare(a.bundle)
