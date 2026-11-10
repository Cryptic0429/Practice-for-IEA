"""Training-pool tool feasibility, not labels, policy predictions or heldout results."""
import os,textwrap
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from functional_v11.editor import edit as old_edit
from .editor import edit as new_edit

EXP=ROOT/'experiments/functional-v15-20261106'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    target=EXP/'shadow-tool-probes/manifest.json';assert not target.exists()
    bank={r['id']:r for r in read(ROOT/'experiments/functional-v13-20261106/primary-candidates/manifest.json')['items']}
    probes=[('train-native-088',bank['train-native-088']['input_image']),('train-native-231',bank['train-native-231']['input_image'])]
    cases=read(ROOT/'data/training/functional-v14-20261106/cases.json')
    noisy=next(c for c in cases if c['id']=='visual-risk-01-severe-light')
    probes.append(('seen-synthetic-noise-training-probe',noisy['image']))
    items=[]
    for cid,path in probes:
        rgb=load_rgb(ROOT/path);folder=target.parent/cid;folder.mkdir(parents=True)
        for strength in (30,60,90):
            parameters={'shadows':strength};sources=[('INPUT',ROOT/path)]
            for name,function in [('OLD TOOL',old_edit),('COMMON RGB GAIN PROTOTYPE',new_edit)]:
                image=function(rgb,parameters,'bounded-fixed');out=folder/f'{strength}-{name.split()[0].lower()}.png';save(out,image);sources.append((name,out))
            canvas=Image.new('RGB',(1800,750),'white');draw=ImageDraw.Draw(canvas)
            draw.text((16,12),cid+' | TOOL PROBE ONLY; NOT A MODEL OUTPUT OR ADMITTED LABEL',font=font(22),fill='black')
            draw.text((16,48),f'Fixed parameters shadows={strength}; training sources only. No quality grade has been assigned.',font=font(20),fill='black')
            for n,(label,file) in enumerate(sources):
                draw.text((n*600+12,120),label,font=font(20),fill='black')
                with Image.open(file) as im:im=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
                canvas.paste(im,(n*600+10+(580-im.width)//2,170+(550-im.height)//2))
            sheet=folder/f'shadows-{strength}.jpg';canvas.save(sheet,quality=95,subsampling=0)
            items.append(dict(source_id=cid,input=path,input_sha256=sha(ROOT/path),parameters=parameters,original_size=list(rgb.shape[1::-1]),
                sheet=str(sheet.relative_to(ROOT)),sheet_sha256=sha(sheet),old_output=str(sources[1][1].relative_to(ROOT)),new_output=str(sources[2][1].relative_to(ROOT)),
                old_sha256=sha(sources[1][1]),new_sha256=sha(sources[2][1]),training_only=True,synthetic_noise=cid.startswith('seen-'),label_admitted=False))
    write(target,dict(at=stamp(),items=items,scope='3 training-pool inputs x3 fixed shadow strengths; algorithmic probe only. No model inference, new label or development/acceptance tuning.',
        old_editor_sha256=sha(ROOT/'scripts/functional_v11/editor.py'),new_editor_sha256=sha(ROOT/'scripts/functional_v15/editor.py'),GPU_hours=0))
    print('9 training-only tool panels generated; no grade, data admission or model inference.')


if __name__=='__main__':main()
