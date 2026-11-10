"""Additional CPU alternatives and mild-noise counterexamples; never admit labels."""
from pathlib import Path
import math
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from .prepare_counterfactuals import EXP,render,INSTRUCTIONS

COMPOUND=dict(zh='请增加照片对比度，稍微提亮，并让颜色更鲜明；保留主体、天空和暗部细节，不要明显放大噪声。',
    en='Increase contrast, brighten the photograph a little, and make the colours more vivid while retaining subject, sky and shadow detail without markedly amplifying noise.')


def panel(cid,name,images):
    rows=math.ceil(len(images)/3);canvas=Image.new('RGB',(1800,100+rows*610),'white');d=ImageDraw.Draw(canvas)
    d.text((15,12),cid+' | '+name+' | CPU candidates, not admitted',fill='black',font=font(24))
    for i,(label,path) in enumerate(images.items()):
        x,y=(i%3)*600,90+(i//3)*610
        d.text((x+10,y),label,fill='black',font=font(21))
        with Image.open(path) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,560),Image.Resampling.LANCZOS)
        canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+36+(560-fitted.height)//2))
    target=EXP/'extended-candidates/panels'/f'{cid}-{name}.jpg';target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target,quality=95,subsampling=0)
    return dict(path=str(target.relative_to(ROOT)),sha256=sha(target))


def main():
    destination=EXP/'extended-candidates/manifest.json';assert not destination.exists()
    source=EXP/'counterfactual-candidates/manifest.json';original=read(source)
    records=[]
    for index,c in enumerate(original['cases']):
        cid=c['id'];originals={name:ROOT/r['path'] for name,r in c['images'].items()}
        assert all(sha(originals[k])==r['sha256'] for k,r in c['images'].items())
        deep=load_rgb(originals['deep-clean-input']);shallow=load_rgb(originals['shallow-clean-input']);severe=load_rgb(originals['noisy-input'])
        rng=np.random.default_rng(20261120+index)
        mild=np.clip(np.rint(deep.astype(np.float32)+rng.normal(0,.55,deep.shape)),0,255).astype(np.uint8)
        settings={
            'mild-noise-light':dict(exposure=65),
            'severe-light-damped':dict(exposure=65,contrast=-35,saturation=-15),
            'severe-light-shadow-damped':dict(exposure=20,shadows=80,contrast=-15),
            'deep-compound':dict(exposure=20,contrast=10,shadows=35,saturation=15),
            'shallow-compound':dict(exposure=10,contrast=20,shadows=15,saturation=15),
            'mild-noise-compound':dict(exposure=20,contrast=10,shadows=35,saturation=15),
            'severe-compound':dict(exposure=20,contrast=10,shadows=35,saturation=15),
            'severe-compound-gentle':dict(exposure=10,contrast=5,shadows=25,saturation=10),
            'severe-compound-shadows':dict(exposure=10,contrast=15,shadows=55,saturation=15)}
        inputs={'mild-noise-input':mild};input_roles={}
        for name,p in settings.items():
            rgb=deep if name=='deep-compound' else shallow if name=='shallow-compound' else mild if name.startswith('mild-noise') else severe
            inputs[name]=render(rgb,p)
            input_roles[name]='deep-clean-input' if name=='deep-compound' else 'shallow-clean-input' if name=='shallow-compound' else 'mild-noise-input' if name.startswith('mild-noise') else 'noisy-input'
        images={}
        for name,rgb in inputs.items():
            file=EXP/'extended-candidates/images'/f'{cid}-{name}.png';save(file,rgb)
            images[name]=dict(path=str(file.relative_to(ROOT)),sha256=sha(file),size=list(Image.fromarray(rgb).size))
        paths={name:ROOT/r['path'] for name,r in images.items()}
        light={key:path for key,path in [('mild-noise-input',paths['mild-noise-input']),('mild-noise-light',paths['mild-noise-light']),
            ('severe-input',originals['noisy-input']),('severe-light-damped',paths['severe-light-damped']),('severe-light-shadow-damped',paths['severe-light-shadow-damped'])]}
        compound={key:path for key,path in [('deep-input',originals['deep-clean-input']),('deep-compound',paths['deep-compound']),
            ('shallow-input',originals['shallow-clean-input']),('shallow-compound',paths['shallow-compound']),
            ('mild-noise-input',paths['mild-noise-input']),('mild-noise-compound',paths['mild-noise-compound']),
            ('severe-input',originals['noisy-input']),('severe-compound',paths['severe-compound']),
            ('severe-compound-gentle',paths['severe-compound-gentle']),('severe-compound-shadows',paths['severe-compound-shadows'])]}
        sheets={'light-alternatives':panel(cid,'light-alternatives',light),'compound':panel(cid,'compound',compound)}
        records.append(dict(id=cid,domain=c['domain'],source_id=c['source_id'],group_id=c['group_id'],images=images,
            settings=settings,input_roles=input_roles,panels=sheets,
            requests=dict(light=INSTRUCTIONS,compound=COMPOUND),
            decision_status='All new decisions pending direct visual review; no automatic decline or parameter admission',
            note='Mild-noise input is a counterexample to refusing every imperfect image. Compensated severe-noise alternatives must be checked; a visibly good feasible edit takes precedence over the proposed refusal.'))
    write(destination,dict(at=stamp(),status='CPU_extended_candidates_only_not_training_labels',
        original_manifest=str(source.relative_to(ROOT)),original_manifest_sha256=sha(source),
        original_groups=20,domains=dict(portrait=10,landscape=10),maximum_semantic_candidates=160,
        new_input_conditions=['deep clean','shallow clean','deep with very mild synthetic grain','deep with stronger synthetic chroma corruption'],
        requests_per_condition=2,all_requests_AI_authored=True,labels_admitted=0,new_GPU_hours=0,
        intended_question='Can matching identical words to different photographic conditions teach safe image-dependent parameters and quality decisions, rather than a fixed text-to-enhancement lookup?',
        cautions=['Synthetic corruption creates a domain gap and possible shortcut; it cannot certify real-photo safety.',
            'Every final edit/decline must have direct visual evidence. Uncertain or failed proposals do not enter training.',
            'No protocol or model training is started here. Preserve round9 failure and the original shared16GPUh cap.'],cases=records))
    print('Extended CPU candidates:20 groups10/10,4 input conditions x2 requests; all160 decisions unadmitted.')

if __name__=='__main__':main()
