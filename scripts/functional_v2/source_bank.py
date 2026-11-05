"""Prepare source-only contact sheets; photo/domain labels must be directly observed."""
import argparse
import collections
import hashlib
from pathlib import Path
import textwrap
import re
import cv2
from PIL import Image,ImageDraw,ImageOps
from functional_v1.prepare import fingerprint,near
from functional_v1.curate import font
from functional_v1.editor import load_rgb,statistics
from .common import ROOT,DATA,EXP,NAME,read,write,sha,stamp


def prepare():
    output=EXP/'source-review'
    assert not (output/'manifest.json').exists()
    output.mkdir(parents=True,exist_ok=True)
    first=ROOT/'data/training/functional-v1-20261104'
    prior={r['group_id'] for split in ('development','acceptance') for r in read(first/f'{split}-cases.json')}
    candidates=[];seen=set()
    # Existing reviewed training photographs and original-pool requests are candidates only.
    natural=read(EXP/'planning/natural-candidates.json')['candidates']
    pool=read(ROOT/'data/training/quality-v2-20261103/train3k.json')
    groups={r['group_id']:r for r in pool}
    ordered=[groups[r['group_id']] for r in natural]
    ordered += [groups[c['group_id']] for c in read(first/'train-cases.json')]
    sources={'train':ordered,'development':read(ROOT/'data/training/quality-v2-20261103/validation.json'),
             'acceptance':read(ROOT/'data/stage1/splits/image-edit-3k-v1/test.json')}
    for split,rows in sources.items():
        if split!='train':rows=sorted(rows,key=lambda r:hashlib.sha256((NAME+split+r['group_id']).encode()).hexdigest())
        count=0
        for r in rows:
            gid=r['group_id']
            if gid in seen or gid in prior:continue
            path=Path(r['images'][0]);rgb=load_rgb(path);stat=statistics(cv2.resize(rgb,(256,256)))
            if not .15<stat['mean_luminance']<.85 or stat['luminance_std']<.06:continue
            seen.add(gid);count+=1
            candidates.append({'id':f'{split}-{count:03}','split':split,'group_id':gid,'record_id':r['id'],
                'source_image':str(path),'source_sha256':sha(path),'original_request':r['extra_info']['instruction'],
                'source_metadata':r['extra_info'],'statistics':stat,
                'status':'source_not_yet_visually_classified'})
            if split!='train' and count>=80:break
    sheets=[]
    for split in sources:
        rows=[r for r in candidates if r['split']==split]
        for offset in range(0,len(rows),12):
            subset=rows[offset:offset+12];canvas=Image.new('RGB',(1800,1530),'white');d=ImageDraw.Draw(canvas)
            d.text((12,8),'Source classification only | '+split+' | portraits / landscapes / reject',font=font(25),fill='black')
            for i,r in enumerate(subset):
                x,y=(i%4)*450,55+(i//4)*490
                d.text((x+8,y),r['id'],font=font(23),fill='black')
                with Image.open(r['source_image']) as im:im=ImageOps.exif_transpose(im).convert('RGB');im=ImageOps.contain(im,(430,440),Image.Resampling.LANCZOS)
                canvas.paste(im,(x+10+(430-im.width)//2,y+36+(440-im.height)//2))
            path=output/f'{split}-{offset//12+1:02}.jpg';canvas.save(path,quality=95,subsampling=0)
            sheets.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'source_ids':[r['id'] for r in subset]})
    write(output/'manifest.json',{'created_at':stamp(),'candidates':candidates,'sheets':sheets,
        'rule':'Directly classify real portrait photographs, landscape photographs, or reject. This is not label admission or an editing-quality rating.',
        'excludes_all_round1_development_and_acceptance_groups':True})
    print(collections.Counter(r['split'] for r in candidates),len(sheets))


def record(notes):
    root=EXP/'source-review';manifest=read(root/'manifest.json');byid={r['id']:r for r in manifest['candidates']}
    sheets={s:r for r in manifest['sheets'] for s in r['source_ids']}
    for item in read(notes):
        assert item['domain'] in ('portrait','landscape','reject')
        assert len(item['evidence'])>=12
        source=byid[item['id']];sheet=sheets[item['id']]
        assert sha(ROOT/sheet['path'])==sheet['sha256']
        append=__import__('functional_v1.common',fromlist=['append']).append
        append(root/'observations.jsonl',{**item,'group_id':source['group_id'],'source_sha256':source['source_sha256'],
            'sheet':sheet['path'],'sheet_sha256':sheet['sha256'],'observed_at':stamp(),
            'reviewer':'Codex direct visual; AI-assisted, not human'})


def extend():
    output=EXP/'source-review';manifest=read(output/'manifest.json')
    assert not (DATA/'source-splits.json').exists(), 'Source splits already frozen'
    write(output/f'manifest-before-extension-{len(manifest["candidates"])}.json',manifest)
    prior={r['group_id'] for split in ('development','acceptance') for r in read(ROOT/'data/training/functional-v1-20261104'/f'{split}-cases.json')}
    seen={r['group_id'] for r in manifest['candidates']}|prior
    sources={'train':ROOT/'data/training/quality-v2-20261103/train3k.json',
        'development':ROOT/'data/training/quality-v2-20261103/validation.json',
        'acceptance':ROOT/'data/stage1/splits/image-edit-3k-v1/test.json'}
    for split,path in sources.items():
        rows=read(path)
        score=lambda r:len(re.findall(r'\b(?:sky|landscape|scenery|mountain|tree|grass|sunset|ocean|cloud|beach|river|lake|forest|garden)\b',r['extra_info']['instruction'].lower()))
        rows=sorted(rows,key=lambda r:(-score(r),hashlib.sha256((NAME+'extend'+r['group_id']).encode()).hexdigest()))
        added=[];count=sum(r['split']==split for r in manifest['candidates'])
        for r in rows:
            if r['group_id'] in seen:continue
            path=Path(r['images'][0]);rgb=load_rgb(path);stat=statistics(cv2.resize(rgb,(256,256)))
            if not .15<stat['mean_luminance']<.85 or stat['luminance_std']<.06:continue
            seen.add(r['group_id']);count+=1
            added.append({'id':f'{split}-{count:03}','split':split,'group_id':r['group_id'],'record_id':r['id'],
                'source_image':str(path),'source_sha256':sha(path),'original_request':r['extra_info']['instruction'],
                'source_metadata':r['extra_info'],'statistics':stat,'status':'source_not_yet_visually_classified',
                'candidate_retrieval':'landscape-related request keywords then deterministic hash; not visual classification'})
            if len(added)>={'train':120,'development':36,'acceptance':60}[split]:break
        for offset in range(0,len(added),12):
            subset=added[offset:offset+12];canvas=Image.new('RGB',(1800,1530),'white');d=ImageDraw.Draw(canvas)
            d.text((12,8),'Additional source candidates | '+split+' | classification only',font=font(25),fill='black')
            for i,r in enumerate(subset):
                x,y=(i%4)*450,55+(i//4)*490;d.text((x+8,y),r['id'],font=font(23),fill='black')
                with Image.open(r['source_image']) as im:im=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(430,440),Image.Resampling.LANCZOS)
                canvas.paste(im,(x+10+(430-im.width)//2,y+36+(440-im.height)//2))
            target=output/f'extra-{split}-{offset//12+1:02}.jpg';canvas.save(target,quality=95,subsampling=0)
            manifest['sheets'].append({'path':str(target.relative_to(ROOT)),'sha256':sha(target),'source_ids':[r['id'] for r in subset]})
        manifest['candidates']+=added
        print(split,'added',len(added))
    manifest['extended_at']=stamp();write(output/'manifest.json',manifest)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','record','extend']);ap.add_argument('--notes',type=Path);a=ap.parse_args()
    {'prepare':prepare,'extend':extend,'record':lambda:record(a.notes)}[a.command]()
