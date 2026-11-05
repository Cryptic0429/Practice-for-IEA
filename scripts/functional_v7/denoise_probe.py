"""Training-pool-only tool-capacity probe, before deciding on new labels/training."""
import hashlib,json,math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from .diagnose_editor import EXP

def denoise(rgb,value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=100:
        raise ValueError('denoise_strength_must_be_between_0_and_100')
    if value==0:return rgb.copy()
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    filtered=cv2.fastNlMeansDenoisingColored(bgr,None,float(value)/10,float(value)/10,7,21)
    return cv2.cvtColor(filtered,cv2.COLOR_BGR2RGB)

def main():
    cv2.setNumThreads(4);dest=EXP/'denoise-feasibility';assert not (dest/'protocol.json').exists()
    data=ROOT/'data/training/functional-v6-20261105';oldexp=ROOT/'experiments/functional-v6-20261105'
    original={c['id']:c for c in read(data/'native-training-candidates.json')}
    accepted={c['id']:c for c in read(data/'cases.json')}
    cid='native-train-x351';c=original[cid]
    rejected=next(r for r in read(oldexp/'native-label-revision/manifest.json')['proposals'] if r['case_id']==cid)['candidates']['C']
    sources=[dict(case_id=cid,input=c['image'],target=rejected['image'],instructions=c['instructions'],
        meaning='Previously rejected training-only brightening target; no new admission or model gain implied.')]
    for cid in ('native-train-x348','native-train-n048','native-train-n204'):
        c=accepted[cid];assert c['split']=='train'
        sources.append(dict(case_id=cid,input=c['image'],target=c['target'],instructions=c['instructions'],meaning='Existing admitted training target, used only as texture preservation control.'))
    for r in sources:r.update(input_sha256=sha(ROOT/r['input']),target_sha256=sha(ROOT/r['target']))
    write(dest/'protocol.json',dict(at=stamp(),stage='training_pool_tool_capacity_probe_not_evaluation',sources=sources,
        strengths=[0,30,60,90],implementation='OpenCV CPU fastNlMeansDenoisingColored, h=hColor=strength/10, template7, search21, applied AFTER existing edit.',
        code_sha256=sha(__file__),no_parameter_search_on_development=True,no_acceptance_access=True,no_training_started=True,
        question='Can a fixed global denoising tool make a previously rejected noisy training target acceptable without destroying stars, fine clothes or hair?',
        decision='Direct visual feasibility only. Do not admit any target, change live tools, or start training without an explicit follow-up dataset review.'))
    rows=[];sheets=[]
    for r in sources:
        before=load_rgb(ROOT/r['input']);target=load_rgb(ROOT/r['target']);assert np.array_equal(denoise(target,0),target)
        entries=[]
        for strength in (0,30,60,90):
            out=denoise(target,strength);key=r['case_id']+'-'+hashlib.sha256(out.tobytes()).hexdigest()[:12]
            path=dest/'images'/f'{key}.png';save(path,out)
            row=dict(case_id=r['case_id'],item_id=key,strength=strength,image=str(path.relative_to(ROOT)),image_sha256=sha(path))
            rows.append(row);entries.append(row)
        entries.sort(key=lambda x:x['item_id'])
        canvas=Image.new('RGB',(1800,1320),'white');d=ImageDraw.Draw(canvas);d.text((12,10),r['case_id'],fill='black',font=font(24));d.text((12,50),r['instructions']['en'][:125],fill='black',font=font(22))
        for i,(label,path) in enumerate([('INPUT',ROOT/r['input'])]+[(x['item_id'][-12:],ROOT/x['image']) for x in entries]):
            x,y=(i%3)*600,120+(i//3)*600;d.text((x+12,y),label,fill='black',font=font(22))
            with Image.open(path) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
            canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+40+(550-fitted.height)//2))
        path=dest/(r['case_id']+'.jpg');canvas.save(path,quality=95,subsampling=0);sheets.append(dict(case_id=r['case_id'],sheet=str(path.relative_to(ROOT)),sha256=sha(path),items=[x['item_id'] for x in entries]))
    write(dest/'manifest.json',dict(at=stamp(),records=rows,sheets=sheets,protocol_sha256=sha(dest/'protocol.json')))
    print('Four training sources, 16 output variants. Review identities/strengths hidden in panels; no labels admitted.')

if __name__=='__main__':main()
