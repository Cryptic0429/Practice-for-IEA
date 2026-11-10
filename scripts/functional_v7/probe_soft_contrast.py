"""Training-only contrast-capacity check before bulk label admission."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save,edit as legacy
from functional_v1.curate import font
from .editor_soft import edit as soft_edit
from .editor_rgb import edit as rgb_edit
from .diagnose_editor import EXP

def main():
    out=EXP/'soft-contrast-capacity';assert not (out/'protocol.json').exists()
    ids=['train-007-restore-contrast-mild','train-007-restore-contrast-strong',
         'train-010-style-contrast-weak','train-010-style-contrast-strong',
         'train-016-restore-contrast-mild','train-016-restore-contrast-strong',
         'train-025-restore-contrast-strong','train-025-style-contrast-weak','train-025-style-contrast-strong',
         'train-034-restore-contrast-strong','train-034-compound','native-train-n048']
    path=ROOT/'data/training/functional-v6-20261105/cases.json';bank={c['id']:c for c in read(path)}
    cases=[bank[cid] for cid in ids];assert all(c['split']=='train' for c in cases)
    # These tests precede all capacity images. No quality acceptance is computed.
    x=np.linspace(0,1,65537,dtype=np.float32)
    from .editor_rgb import tonal_value
    for v in (-100,-50,0,50,100):
        y=tonal_value(x,{'contrast':v},'bounded-fixed');assert np.isfinite(y).all() and y.min()>=0 and y.max()<=1 and np.diff(y).min()>=-1e-7
    from .editor_soft import soft_contrast
    for v in (-100,-50,-1,0,.0001,1,50,100):
        y=soft_contrast(x,v)
        assert np.isfinite(y).all() and y.min()>=0 and y.max()<=1 and np.diff(y).min()>=-1e-7
        assert ((np.abs(y-.5)-np.abs(x-.5))*v>=-1e-6).all()
    assert np.array_equal(soft_contrast(x,0),x)
    assert np.max(np.abs(soft_contrast(x,.0001)-x))<1e-5
    write(out/'protocol.json',dict(at=stamp(),status='frozen_before_RGB_capacity_outputs',cases=ids,source_cases_sha256=sha(path),
        source_screening_sha256=sha(EXP/'curve-label-candidates/initial-screening.json'),
        code_hashes={str(f.relative_to(ROOT)):sha(f) for f in (Path(__file__),Path(__file__).with_name('editor_soft.py'),Path(__file__).with_name('editor_rgb.py'))},
        hypothesis='The bounded logit response still changes the contrast/colour relationship. Retain linear RGB contrast over interior values with a continuous soft shoulder/toe for positive contrast. Negative contrast retains exact legacy response. No parameters are fitted.',
        variants=['frozen old RGB linear/clipped editor','bounded RGB logit contrast','soft bounded linear RGB contrast'],
        controlled_variables='Original training input/request and old parameter JSON fixed across all three; exposure/brightness and later tools unchanged between bounded candidates. No parameter fitting.',
        scope='Twelve deliberately selected training tasks, five previously inspected source groups plus native bridal detail control. Not balanced validation or model improvement.',
        editor_exponent='Linear coefficient2^(contrast/100), positive softplus shoulder/toe thickness0.08*(gain-1), normalize analytic source endpoints; negative exact linear. Zero neutral and continuous.',
        new_labels_admitted=False,training_started=False,live_tools_changed=False,heldout_outputs_generated=False,GPU_hours=0))
    records=[];sheets=[]
    for c in cases:
        source=load_rgb(ROOT/c['image']);assert sha(ROOT/c['image'])==c['input_sha256'];entries=[]
        for variant,fn in [('legacy',legacy),('bounded-soft',lambda x,p:soft_edit(x,p,'bounded-fixed')),('bounded-RGB',lambda x,p:rgb_edit(x,p,'bounded-fixed'))]:
            result=fn(source,c['parameters']);key=c['id']+'-'+hashlib.sha256(result.tobytes()).hexdigest()[:12]
            path=out/'images'/f'{key}.png';save(path,result)
            item=dict(case_id=c['id'],item_id=key,variant=variant,image=str(path.relative_to(ROOT)),image_sha256=sha(path));records.append(item);entries.append(item)
        entries.sort(key=lambda r:r['item_id']);canvas=Image.new('RGB',(1800,1320),'white');d=ImageDraw.Draw(canvas)
        d.text((12,10),c['id'],fill='black',font=font(23));d.text((12,48),c['instructions']['en'][:128],fill='black',font=font(21))
        for i,(label,p) in enumerate([('INPUT',c['image'])]+[(r['item_id'][-12:],r['image']) for r in entries]):
            x,y=(i%3)*600,120+(i//3)*600;d.text((x+12,y),label,fill='black',font=font(21))
            with Image.open(ROOT/p) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
            canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+40+(550-fitted.height)//2))
        path=out/(c['id']+'.jpg');canvas.save(path,quality=95,subsampling=0);sheets.append(dict(case_id=c['id'],sheet=str(path.relative_to(ROOT)),sha256=sha(path),items=[r['item_id'] for r in entries]))
    write(out/'manifest.json',dict(at=stamp(),records=records,sheets=sheets,protocol_sha256=sha(out/'protocol.json')))
    print('Twelve training tasks replayed for bounded soft linear contrast; no labels admitted or new training.')

if __name__=='__main__':main()
