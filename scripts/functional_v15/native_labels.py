"""Native-pixel supplements kept separate from all provisional label-panel notes."""
from pathlib import Path
import argparse,json,hashlib,datetime

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/functional-v15-20261106'

# Chosen from the already viewed whole-frame candidate sheets. These are critical
# native-pixel regions, not a claim of complete giant-image pixel coverage.
CRITICAL_REGIONS={
 'native-train-n127':[('face-shirt',[2200,700,3400,1900]),('cave-shadow',[0,700,1200,1900]),('sea',[1400,1872,2600,3072])],
 'native-train-n190':[('rock',[1100,200,2300,1400]),('shadow-rock',[1800,650,3000,1850]),('sea',[2400,1140,3600,2340]),('sky',[0,0,1200,1200])],
 'native-train-n225':[('rainbow-cloud',[800,600,2000,1800]),('trees-road',[850,2000,2050,3200])],
 'native-train-x151':[('subject',[1000,1200,2200,2400]),('bright-rock',[2300,300,3500,1500]),('deep-water',[0,0,1200,1200])],
 'native-train-x231':[('city-roads',[1000,500,2200,1700]),('skyline',[1800,0,3000,1000])],
 'native-train-x250':[('facade',[800,400,2000,1600]),('trees-sky',[0,0,1200,1200]),('water',[1600,1100,2800,1944])],
 'r13-train-extension-183':[('house-snow-trees',[0,300,1200,1295]),('cloud-mountain',[1024,0,2224,1000])],
 'r13-train-extension-420':[('sky-roof',[1000,300,2200,1500]),('doors-steps',[1000,1200,2200,2400]),('right-roof',[2064,700,3264,1900])],
 'r13-train-extension-526':[('left-building',[0,0,1200,1134]),('right-sky-grass',[816,0,2016,1134])],
 'r13-train-extension-551':[('left-grass',[0,411,1200,1611]),('sky-treeline',[950,0,2150,1200]),('path-grass',[1898,411,3098,1611])],
 'r13-train-native-085':[('seated-person',[0,700,1100,1900]),('standing-face-shirt',[820,200,1920,1400]),('shirt-trousers',[820,1360,1920,2560])],
 'r13-train-native-123':[('faces-flowers',[900,1400,2100,2600]),('dress-suit',[900,2500,2100,3700]),('facade',[0,300,1200,1500])],
 'r13-train-native-212':[('man-face',[800,600,2000,1800]),('woman-face',[2200,600,3400,1800]),('backlit-hair',[3000,200,4200,1400]),('shirt',[500,1392,1700,2592])],
}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(stage,bundle):
    from PIL import Image,ImageDraw
    folder=EXP/('label-candidates' if stage=='labels' else 'quality-challenges')
    items={r['id']:r for r in json.loads((folder/'manifest.json').read_text())['items']}
    raw=EXP/('label-observations.jsonl' if stage=='labels' else 'quality-challenge-observations.jsonl')
    notes=list(map(json.loads,raw.read_text().splitlines()))
    full=EXP/('label-native-observations.jsonl' if stage=='labels' else 'quality-native-observations.jsonl')
    done={r['id'] for r in map(json.loads,full.read_text().splitlines())} if full.exists() else set()
    out=EXP/'native-review'/stage/bundle;assert not (out/'manifest.json').exists();out.mkdir(parents=True,exist_ok=True);rows=[]
    for note in notes:
        if note['quality']!='U' or note['id'] in done:continue
        item=items[note['id']];images=[]
        for role,record in [('INPUT',item['input']),('CANDIDATE',item['candidate'])]:
            source=ROOT/record['path'];assert digest(source)==record['sha256']
            with Image.open(source) as im:
                rgb=im.convert('RGB');w,h=rgb.size
                if max(w,h)<=2000 and w*h<=3500000:regions=[('whole',[0,0,w,h])]
                elif item['id'] in CRITICAL_REGIONS:regions=CRITICAL_REGIONS[item['id']]
                else:raise ValueError(f'Add explicit critical-region mapping before exporting {item["id"]}:{w}x{h}')
                for label,box in regions:
                    assert box[2]<=w and box[3]<=h
                    result=rgb.crop(box);path=out/f'{item["id"]}-{role}-{label}.png'
                    if not path.exists():result.save(path,compress_level=1)
                    with Image.open(path) as check:assert check.size==result.size and check.tobytes()==result.tobytes()
                    images.append(dict(role=role,region=label,path=str(path.relative_to(ROOT)),sha256=digest(path),size=list(result.size),source=record['path'],source_sha256=record['sha256'],source_size=[w,h],box=box))
        # Exact unscaled copies on one canvas make small-source comparisons
        # readable without silently shrinking a noisy image into a contact sheet.
        pairs=[]
        for region in sorted({i['region'] for i in images}):
            parts=[i for i in images if i['region']==region]
            assert len(parts)==2 and parts[0]['size']==parts[1]['size']
            w,h=parts[0]['size']
            if w>960:continue
            canvas=Image.new('RGB',(w*2,h+48),'white');draw=ImageDraw.Draw(canvas)
            for n,part in enumerate(parts):
                draw.text((n*w+8,8),part['role']+' | '+item['id'],fill='black')
                with Image.open(ROOT/part['path']) as im:
                    rgb=im.convert('RGB');canvas.paste(rgb,(n*w,48))
                    assert canvas.crop((n*w,48,(n+1)*w,48+h)).tobytes()==rgb.tobytes()
            pair=out/f'{item["id"]}-PAIR-{region}.png';assert not pair.exists();canvas.save(pair,compress_level=1)
            pairs.append(dict(path=str(pair.relative_to(ROOT)),sha256=digest(pair),size=list(canvas.size),region=region,
                image_pixels_unscaled=True,header_height=48,components=parts))
        rows.append(dict(id=item['id'],stage=stage,input_sha256=item['input']['sha256'],candidate_sha256=item['candidate']['sha256'],images=images,paired_native=pairs))
    (out/'manifest.json').write_text(json.dumps(dict(stage=stage,bundle=bundle,items=rows,scope='Whole native pixels where labelled whole; otherwise specified integer crops only. No resizing or adjustment. Whole-frame context remains in the candidate sheet; giant-image coverage is limited to listed critical regions.'),ensure_ascii=False,indent=2)+'\n')
    print('Native followups',len(rows),';cases:',[r['id'] for r in rows])


def record(stage,bundle,cid,completion,quality,evidence):
    assert completion in 'CPFU' and quality in 'AmMU' and len(evidence)>30
    manifest=EXP/'native-review'/stage/bundle/'manifest.json';item=next(r for r in json.loads(manifest.read_text())['items'] if r['id']==cid)
    assert all(digest(ROOT/i['path'])==i['sha256'] for i in item['images'])
    assert all(digest(ROOT/i['path'])==i['sha256'] for i in item.get('paired_native',[]))
    dest=EXP/('label-native-observations.jsonl' if stage=='labels' else 'quality-native-observations.jsonl')
    old=dest.read_text() if dest.exists() else '';assert cid not in {r['id'] for r in map(json.loads,old.splitlines())}
    row=dict(item,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),completion=completion,quality=quality,evidence=evidence,
        manifest_sha256=digest(manifest),reviewer='Codex direct native-pixel AI review, not human',supplements_raw_panel=True,training_label_admitted=False)
    temp=dest.with_suffix('.tmp');temp.write_text(old+json.dumps(row,ensure_ascii=False)+'\n');temp.replace(dest)
    print('Native supplement preserved',cid,completion,quality)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['labels','quality']);p.add_argument('--bundle',required=True);a=p.parse_args();prepare(a.stage,a.bundle)
