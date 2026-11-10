"""Actual-pixel QA crops; never training input or adjusted/sharpened evidence."""
import os
from PIL import Image
from functional_v1.common import ROOT,read,write,sha,stamp
from .prepare_labels import EXP


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    destination=EXP/'quality-revalidation/details/manifest.json';assert not destination.exists()
    manifest=read(EXP/'quality-revalidation/manifest.json');rows=[]
    for cid in ('native-train-n070','native-train-n127','native-train-n190','visual-risk-07-normal-light'):
        attempts=[r for r in manifest['attempts'] if r['case_id']==cid]
        sources=[dict(role='INPUT',**attempts[0]['input'])]+[dict(role=r['id'],**r['output']) for r in attempts]
        for item in sources:
            path=ROOT/item['path'];assert sha(path)==item['sha256']
            with Image.open(path) as im:
                rgb=im.convert('RGB');w,h=rgb.size
                if cid=='native-train-n070':box=(310,60,min(w,680),min(h,440))
                elif cid=='native-train-n127':
                    cx,cy=int(w*.58),int(h*.43);ww,hh=min(w,1024),min(h,1280);x=max(0,min(w-ww,cx-ww//2));y=max(0,min(h-hh,cy-hh//2));box=(x,y,x+ww,y+hh)
                elif cid=='native-train-n190':
                    ww,hh=min(w,1200),min(h,800);cx,cy=int(w*.66),int(h*.63);x=max(0,min(w-ww,cx-ww//2));y=max(0,min(h-hh,cy-hh//2));box=(x,y,x+ww,y+hh)
                else:box=(0,0,w,h)
                result=rgb.crop(box);target=destination.parent/f'{cid}-{item["role"]}.png';target.parent.mkdir(parents=True,exist_ok=True);result.save(target)
                with Image.open(target) as check:assert check.tobytes()==result.tobytes() and check.size==result.size
            rows.append(dict(case_id=cid,role=item['role'],source=item,source_size=[w,h],box=list(box),path=str(target.relative_to(ROOT)),sha256=sha(target),crop_size=list(result.size)))
    write(destination,dict(at=stamp(),status='CPU_exact_pixel_details_for_review_not_training',items=rows,
        transformations='Integer crop only; no resize, colour adjustment, sharpening or generation. Source/output coordinates identical per case.',GPU_hours=0))
    print([(r['case_id'],r['role'],r['source_size'],r['crop_size']) for r in rows])

if __name__=='__main__':main()
