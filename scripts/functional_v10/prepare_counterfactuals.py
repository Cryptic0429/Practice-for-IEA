"""CPU-only training-pool candidates; no label admission or model training."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from functional_v7.outcomes import execute,EDITOR_VERSION

EXP=ROOT/'experiments/functional-v10-20261106'
DATA=ROOT/'data/training/functional-v10-20261106'
INSTRUCTIONS=dict(zh='请显著提亮这张照片，保留自然色彩和细节，不要明显放大噪声。',
    en='Brighten this photograph substantially while preserving natural colour and detail, without markedly amplifying noise.')


def resize(rgb):
    im=Image.fromarray(rgb)
    if im.width*im.height>262144:
        scale=math.sqrt(262144/(im.width*im.height));im=im.resize((int(im.width*scale),int(im.height*scale)),Image.Resampling.LANCZOS)
    return np.asarray(im).copy()


def noise(rgb,index):
    rng=np.random.default_rng(20261117+index)
    image=rgb.astype(np.float32)/255
    h,w=image.shape[:2]
    # Synthetic chroma blocks/banding are deliberately varied. They are NOT real camera noise.
    if index%3==0:
        coarse=rng.normal(0,.025,(max(2,h//10),max(2,w//10),3)).astype(np.float32)
        field=cv2.resize(coarse,(w,h),interpolation=cv2.INTER_NEAREST)
        method='independent chroma blocks plus fine grain'
    elif index%3==1:
        bands=rng.normal(0,.018,(max(2,h//5),1,3)).astype(np.float32)
        field=cv2.resize(bands,(w,h),interpolation=cv2.INTER_LINEAR)
        coarse=rng.normal(0,.015,(max(2,h//18),max(2,w//18),3)).astype(np.float32)
        field+=cv2.resize(coarse,(w,h),interpolation=cv2.INTER_LINEAR)
        method='horizontal chroma banding plus coarse smooth chroma noise'
    else:
        coarse=rng.normal(0,.027,(max(2,h//7),max(2,w//7),3)).astype(np.float32)
        field=cv2.resize(coarse,(w,h),interpolation=cv2.INTER_NEAREST)
        method='small chroma blocks plus fine grain'
    field+=rng.normal(0,.008,image.shape).astype(np.float32)
    out=np.clip(np.rint((image+field)*255),0,255).astype(np.uint8)
    return out,dict(seed=20261117+index,kind=method,synthetic=True,not_real_sensor_capture=True)


def render(rgb,parameters):
    return execute(rgb,dict(action='edit',parameters=parameters))['result']


def main():
    manifest=EXP/'counterfactual-candidates/manifest.json';assert not manifest.exists()
    assert not (EXP/'training-protocol.json').exists()
    prior=ROOT/'data/training/functional-v7-20261105/cases.json';cases=read(prior)
    sources={c['source_id']:c for c in cases if c.get('canonical_photo') and c['family']=='lighting' and c.get('previous_action')=='restore-exposure-mild'}
    selected=[]
    for domain in ('portrait','landscape'):
        selected.extend(sorted([c for c in sources.values() if c['domain']==domain],key=lambda c:c['source_id'])[:10])
    assert len(selected)==20 and Counter(c['domain'] for c in selected)==dict(portrait=10,landscape=10)
    allowed={c['group_id'] for c in cases}
    records=[]
    for index,c in enumerate(selected):
        assert c['group_id'] in allowed
        cid=f'visual-risk-{index+1:02d}'
        canonical=ROOT/c['canonical_photo'];base=resize(load_rgb(canonical))
        originals={}
        originals['canonical']=base
        originals['deep-clean-input']=render(base,{'exposure':-65})
        originals['deep-clean-edit']=render(originals['deep-clean-input'],{'exposure':65})
        originals['shallow-clean-input']=render(base,{'exposure':-30})
        originals['shallow-clean-edit']=render(originals['shallow-clean-input'],{'exposure':40})
        originals['noisy-input'],corruption=noise(originals['deep-clean-input'],index)
        parameters={'noisy-exposure65':{'exposure':65},'noisy-brightness35':{'brightness':35},
                    'noisy-exposure40-shadows50':{'exposure':40,'shadows':50}}
        for name,p in parameters.items():originals[name]=render(originals['noisy-input'],p)
        images={}
        for name,rgb in originals.items():
            file=EXP/'counterfactual-candidates/images'/f'{cid}-{name}.png';save(file,rgb)
            images[name]=dict(path=str(file.relative_to(ROOT)),sha256=sha(file),size=list(Image.fromarray(rgb).size))
        canvas=Image.new('RGB',(1800,1940),'white');d=ImageDraw.Draw(canvas)
        d.text((15,12),cid+' | '+c['domain']+' | SAME request for all three inputs',fill='black',font=font(24))
        d.text((15,48),'Substantially brighten, retain natural colour/detail, do not markedly amplify noise.',fill='black',font=font(22))
        for i,(name,rgb) in enumerate(originals.items()):
            x,y=(i%3)*600,90+(i//3)*610
            d.text((x+10,y),name,fill='black',font=font(22))
            im=ImageOps.contain(Image.fromarray(rgb),(580,560),Image.Resampling.LANCZOS)
            canvas.paste(im,(x+10+(580-im.width)//2,y+36+(560-im.height)//2))
        sheet=EXP/'counterfactual-candidates/panels'/f'{cid}.jpg';sheet.parent.mkdir(parents=True,exist_ok=True);canvas.save(sheet,quality=95,subsampling=0)
        records.append(dict(id=cid,source_id=c['source_id'],group_id=c['group_id'],domain=c['domain'],
            canonical_photo=str(canonical.relative_to(ROOT)),canonical_sha256=sha(canonical),
            original_source_sha256=c['original_source_sha256'],instructions=INSTRUCTIONS,
            input_policy='Only one derived input image, the identical original-language request and frozen round7 tool system prompt. Canonical/target/noise metadata/expected decision never enter inference.',
            images=images,corruption=corruption,candidate_parameters=parameters,
            proposed_decisions={'deep-clean':{'action':'edit','parameters':{'exposure':65}},
                'shallow-clean':{'action':'edit','parameters':{'exposure':40}},
                'noisy':{'action':'decline','reason':'unreliable_quality'}},
            target_status='CPU_candidate_only_NOT_admitted',sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet)))
    write(manifest,dict(at=stamp(),status='CPU_candidates_only_no_admission_no_training',source_cases_sha256=sha(prior),
        original_photo_groups=20,domains=dict(portrait=10,landscape=10),semantic_candidates=60,bilingual_candidates=120,
        requests='100% AI-authored identical semantic request across changed photographic conditions; synthetic darkening/noise, NOT untouched real user requests.',
        editor=EDITOR_VERSION,cases=records,all_groups_within_previous_training160=True,development_or_acceptance_photos_added=False,
        label_review_required='Every proposed edit must be visually completed and acceptable. Proposed quality refusals require direct review of input and all three actual attempted outputs; disagreement stays uncertain and is not admitted. No L1-based selection.',
        limitation='Synthetic corruption may create an easy shortcut; heldout real-photo functional judgment is still required. These candidates do not prove refusal reliability.',
        previous_round9_status='Remaining prespecified checkpoints pending at CPU preparation; no round10 GPU work is authorized by this file alone. Requires separate final admission and frozen protocol within original16GPUh cap.'))
    print('Prepared20 existing training groups10/10,60 semantic CPU candidates; ZERO admitted, zero new GPU work.')

if __name__=='__main__':main()
