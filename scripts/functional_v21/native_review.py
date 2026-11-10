"""Bound original-pixel QA exports; giant photographs require explicit region coverage."""
import argparse,json,hashlib,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v21-20261107'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(split,bundle,only_cases=None,only_items=None,native_folder='native-pixels'):
    from PIL import Image
    protocol=json.loads((EXP/'training-protocol.json').read_text())
    key={'native-development':'native_development','old-native':'old_native','development':'controlled_development'}[split]
    source=ROOT/protocol['sources'][key]['path'];assert digest(source)==protocol['sources'][key]['sha256']
    cases={c['id']:c for c in json.loads(source.read_text())}
    folder=EXP/'review'/split;manifest=json.loads((folder/bundle/'manifest.json').read_text())
    notes={n['item_id']:n for n in map(json.loads,(folder/'codex-observations.jsonl').read_text().splitlines())}
    old=folder/'codex-full-resolution-observations.jsonl';done={n['item_id'] for n in map(json.loads,old.read_text().splitlines())}
    assert Path(native_folder).name==native_folder
    out=folder/bundle/native_folder;assert not (out/'manifest.json').exists();out.mkdir(parents=True,exist_ok=True);rows=[]
    for item in manifest['items']:
        if only_cases and item['case_id'] not in only_cases:continue
        if only_items and item['item_id'] not in only_items:continue
        note=notes.get(item['item_id'])
        if not note or note.get('prior_final_sources') or note['quality']!='uncertain' or item['item_id'] in done:continue
        case=cases[item['case_id']];assert digest(ROOT/case['image'])==case['input_sha256']
        images=[]
        for role,path in [('INPUT',ROOT/case['image']),(item['item_id'],ROOT/item['image'])]:
            with Image.open(path) as image:
                rgb=image.convert('RGB');size=list(rgb.size)
                if max(size)<=1600:regions=[('whole',[0,0,*size])]
                elif case['id']=='native-dev-003':regions=[('upper',[0,0,1494,1300]),('lower',[0,1150,1494,2450])]
                elif case['id']=='native-dev-023':regions=[('upper-left',[0,0,1000,1100]),('upper-right',[780,0,1780,1100]),('lower-left',[0,946,1000,2046]),('lower-right',[780,946,1780,2046])]
                elif case['id']=='native-dev-032':regions=[('sky-points',[1600,900,2800,2100]),('foreground',[1000,2500,2200,3700])]
                elif case['id']=='native-dev-148':regions=[('left-terraces',[1000,3400,2200,4600]),('right-water-terraces',[7300,3400,8500,4600]),('mountain-cloud',[3600,1600,4800,2800]),('bright-sky',[7800,0,9000,1200])]
                else:raise ValueError(f'Explicit native-region mapping needed before exporting {case["id"]},size{size}')
                for label,box in regions:
                    assert box[2]<=size[0] and box[3]<=size[1]
                    crop=rgb.crop(box);assert max(crop.size)<=1600
                    target=out/f'{case["id"]}-{role}-{label}.png'
                    if not target.exists():crop.save(target)
                    with Image.open(target) as check:assert check.size==crop.size and check.tobytes()==crop.tobytes()
                    images.append(dict(role=role,region=label,path=str(target.relative_to(ROOT)),sha256=digest(target),
                        size=list(crop.size),source=str(path.relative_to(ROOT)),source_sha256=digest(path),source_size=size,box=box))
        rows.append(dict(item_id=item['item_id'],case_id=case['id'],input_sha256=case['input_sha256'],output_sha256=item['image_sha256'],images=images))
    (out/'manifest.json').write_text(json.dumps(dict(split=split,bundle=bundle,items=rows,
        policy='Integer-coordinate native pixels only, no resizing or colour change. Whole-frame context remains on the anonymous sheet. Whole-image QA only where region=whole; bounded regions do not claim complete native coverage of giant files.'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(split=split,bundle=bundle,new_native_followups=len(rows),cases=[r['case_id'] for r in rows])))


def record(split,bundle,item_id,completion,quality,evidence,native_folder='native-pixels'):
    assert Path(native_folder).name==native_folder
    folder=EXP/'review'/split;manifest=folder/bundle/native_folder/'manifest.json';items=json.loads(manifest.read_text())['items']
    row=next(r for r in items if r['item_id']==item_id)
    assert all(digest(ROOT/i['path'])==i['sha256'] for i in row['images'])
    assert completion in 'CPFU' and quality in 'AmMU' and len(evidence)>30
    path=folder/'codex-full-resolution-observations.jsonl';old=path.read_text();assert item_id not in {n['item_id'] for n in map(json.loads,old.splitlines())}
    note=dict(row,completion=dict(C='completed',P='partial',F='failed',U='uncertain')[completion],
        quality=dict(A='acceptable',m='minor_issue',M='major_issue',U='uncertain')[quality],evidence=evidence,
        manifest_sha256=digest(manifest),reviewer='Codex direct anonymous native-pixel AI review; not human',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplements_raw_contact_sheet=True,
        scope='Native-pixel whole image or explicit integer regions, as listed; whole-frame context from the sheet. No denoising or recovered detail claim.')
    path.write_text(old+json.dumps(note,ensure_ascii=False)+'\n');print('Recorded native-pixel supplement',item_id)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',required=True,choices=['native-development','old-native','development']);parser.add_argument('--bundle',required=True)
    parser.add_argument('--only-cases',nargs='+');parser.add_argument('--only-items',nargs='+');parser.add_argument('--native-folder',default='native-pixels')
    args=parser.parse_args();prepare(args.split,args.bundle,args.only_cases,args.only_items,args.native_folder)
