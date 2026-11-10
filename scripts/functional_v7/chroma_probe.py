"""CPU training-only follow-up: keep luminance detail, restore colour separately."""
import hashlib,json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from .diagnose_editor import EXP

def restore_components(target,clean,luminance_fraction):
    a=cv2.cvtColor(target.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    b=cv2.cvtColor(clean.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    b[...,0]=a[...,0]*(1-luminance_fraction)+b[...,0]*luminance_fraction
    out=cv2.cvtColor(b,cv2.COLOR_LAB2RGB)
    return np.clip(np.rint(out*255),0,255).astype(np.uint8)

def main():
    cv2.setNumThreads(4);source=EXP/'learned-denoise-feasibility';out=EXP/'chroma-denoise-feasibility';assert not (out/'protocol.json').exists()
    assert (source/'codex-observations.json').exists(),'Complete direct review before follow-up design.'
    manifest=read(source/'manifest.json');sources=read(source/'protocol.json')['sources']
    write(out/'protocol.json',dict(at=stamp(),status='frozen_before_chroma_probe_outputs',source_manifest_sha256=sha(source/'manifest.json'),
        source_observations_sha256=sha(source/'codex-observations.json'),code_sha256=sha(__file__),sources=sources,
        fixed_variants=['original edited target','full external SCUNet RGB','cleaned Lab a/b with original L','cleaned Lab a/b with 50% restored L'],
        implementation='Use already frozen uint8 denoiser output. OpenCV float32 RGB0-1 toLab, a/b from clean, L blend0 or0.5, Lab toRGB, round/clamp uint8.',
        objective='Reduce visible colour noise while preserving stars, cloth and face luminance detail. Training-only tool capacity, not instruction-model gain.',
        GPU_hours=0,training_started=False,acceptance_access=False,development_access=False,new_labels_automatic=False))
    records=[];sheets=[]
    for c in sources:
        cid=c['case_id'];old=next(r for r in manifest['records'] if r['case_id']==cid and r['strength']==0)
        nn=next(r for r in manifest['records'] if r['case_id']==cid and r['strength']==100)
        assert sha(ROOT/old['image'])==old['image_sha256'] and sha(ROOT/nn['image'])==nn['image_sha256']
        target,clean=load_rgb(ROOT/old['image']),load_rgb(ROOT/nn['image'])
        entries=[]
        variants=[('original',target),('full-RGB',clean),('chroma-only',restore_components(target,clean,0)),('half-luminance',restore_components(target,clean,.5))]
        for variant,image in variants:
            key=cid+'-'+hashlib.sha256(image.tobytes()).hexdigest()[:12];path=out/'images'/f'{key}.png';save(path,image)
            row=dict(case_id=cid,item_id=key,variant=variant,image=str(path.relative_to(ROOT)),image_sha256=sha(path),new_label_admitted=False);entries.append(row);records.append(row)
        entries.sort(key=lambda r:r['item_id']);canvas=Image.new('RGB',(1800,1320),'white');d=ImageDraw.Draw(canvas)
        d.text((12,10),cid,fill='black',font=font(24));d.text((12,50),c['instructions']['en'][:125],fill='black',font=font(22))
        for i,(label,path) in enumerate([('INPUT',ROOT/c['input'])]+[(r['item_id'][-12:],ROOT/r['image']) for r in entries]):
            x,y=(i%3)*600,120+(i//3)*600;d.text((x+12,y),label,fill='black',font=font(22))
            with Image.open(path) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
            canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+40+(550-fitted.height)//2))
        path=out/(cid+'.jpg');canvas.save(path,quality=95,subsampling=0);sheets.append(dict(case_id=cid,sheet=str(path.relative_to(ROOT)),sha256=sha(path),items=[r['item_id'] for r in entries]))
    write(out/'manifest.json',dict(at=stamp(),records=records,sheets=sheets,protocol_sha256=sha(out/'protocol.json')))
    print('Eight new colour/luminance variants, four training sources; no training/admission or heldout inference.')

if __name__=='__main__':main()
