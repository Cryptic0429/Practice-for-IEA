"""Create matched TRAIN conditions and actual tool attempts, never auto-label."""
import hashlib,textwrap
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from functional_v1.editor import save
from functional_v15.outcomes import execute,EDITOR_VERSION,SCHEMA_VERSION

EXP=ROOT/'experiments/functional-v22-20261107'
DATA=ROOT/'data/training/functional-v22-20261107'


def main():
    plan=read(EXP/'quality-data-plan.json');assert not (EXP/'quality-candidates/manifest.json').exists()
    assert sha(ROOT/plan['source_cases'])==plan['source_sha256']
    old=read(ROOT/plan['source_cases']);all_groups={c['group_id'] for c in old}
    bad={c['group_id'] for c in old if c['decision'].get('reason')=='unreliable_quality'}
    previous=read(ROOT/'experiments/functional-v21-20261107/training-protocol.json')
    held=set()
    for spec in list(previous['sources'].values())+list(previous['sealed_acceptance_sources'].values()):
        assert sha(ROOT/spec['path'])==spec['sha256']
        if spec['path']!=plan['source_cases']:
            held.update(c['group_id'] for c in read(ROOT/spec['path']))
    # The R21 source is the only TRAIN entry; equality is checked above.
    assert not all_groups & held
    items=[]
    for ordinal,s in enumerate(plan['groups'],1):
        assert s['group_id'] in all_groups and s['group_id'] not in bad and s['group_id'] not in held
        path=ROOT/s['canonical_photo']
        with Image.open(path) as opened:
            original_size=list(opened.size);image=ImageOps.exif_transpose(opened).convert('RGB')
            image.thumbnail((1024,1024),Image.Resampling.LANCZOS);rgb=np.asarray(image,dtype=np.float32)
        digest=hashlib.sha256(('functional-v22-synthetic-noise:'+s['group_id']).encode()).digest()
        seed=int.from_bytes(digest[:8],'little');sigma=(8,12,16)[digest[8]%3]
        rng=np.random.default_rng(seed);noisy=.60*rgb+rng.normal(0,sigma,rgb.shape)
        for condition,array in [('underexposed',.60*rgb),('underexposed-noisy',noisy)]:
            cid=f'quality-boundary-{ordinal:02d}-'+('a' if condition=='underexposed' else 'b')
            inp=DATA/'candidate-inputs'/f'{cid}.png';save(inp,np.clip(np.rint(array),0,255).astype(np.uint8))
            source_rgb=np.asarray(Image.open(inp).convert('RGB')).copy();attempts=[]
            for index,params in enumerate(plan['candidate_parameters'],1):
                result=execute(source_rgb,dict(action='edit',parameters=params));assert result['result'] is not None
                output=EXP/'quality-candidates/images'/f'{cid}-{index}.png';save(output,result['result'])
                attempts.append(dict(id=f'{cid}-candidate-{index}',parameters=params,path=str(output.relative_to(ROOT)),sha256=sha(output),
                    decision=dict(action='edit',parameters=params),label_admitted=False))
            anonymous=sorted(attempts,key=lambda a:hashlib.sha256((cid+a['id']).encode()).hexdigest())
            canvas=Image.new('RGB',(2200,780),'white');draw=ImageDraw.Draw(canvas)
            draw.text((16,12),cid+' | TRAIN CANDIDATES; NO MODEL RESULT OR ADMITTED LABEL',font=font(24),fill='black')
            for i,line in enumerate(textwrap.wrap(plan['instructions']['en'],150)):
                draw.text((16,49+25*i),line,font=font(20),fill='black')
            sources=[('INPUT',inp)]+[(f'CANDIDATE {chr(65+i)}',ROOT/a['path']) for i,a in enumerate(anonymous)]
            for index,(label,image_path) in enumerate(sources):
                x=index*550;draw.text((x+12,120),label,font=font(21),fill='black')
                with Image.open(image_path) as opened:fitted=ImageOps.contain(opened.convert('RGB'),(530,550),Image.Resampling.LANCZOS)
                canvas.paste(fitted,(x+10+(530-fitted.width)//2,165+(550-fitted.height)//2))
            draw.text((16,742),'Whole frames for initial triage only; native-size evidence required before label admission.',font=font(19),fill='black')
            panel=EXP/'quality-candidates/panels'/f'{cid}.jpg';panel.parent.mkdir(parents=True,exist_ok=True);canvas.save(panel,quality=95,subsampling=0)
            items.append(dict(id=cid,group_id=s['group_id'],domain=s['domain'],source_partition=s['source_partition'],
                original=dict(path=s['canonical_photo'],sha256=sha(path),native_size=original_size),
                image=str(inp.relative_to(ROOT)),input_sha256=sha(inp),input_size=list(source_rgb.shape[1::-1]),instructions=plan['instructions'],
                condition=dict(kind=condition,synthetic=True,underexposure_factor=.60,max_edge=1024,noise_sigma=sigma if condition=='underexposed-noisy' else 0,
                    RNG_seed=seed if condition=='underexposed-noisy' else None,not_real_camera_capture=True),
                candidates=attempts,panel=dict(path=str(panel.relative_to(ROOT)),sha256=sha(panel)),
                anonymous_candidate_order=[a['id'] for a in anonymous],status='not_viewed_not_admitted',label_admitted=False))
        write(EXP/'quality-preparation-progress.json',dict(at=stamp(),original_groups=ordinal,semantic_candidates=len(items),labels_admitted=0,GPU_hours=0))
    write(EXP/'quality-candidates/manifest.json',dict(at=stamp(),plan_sha256=sha(EXP/'quality-data-plan.json'),items=items,
        total_original_groups=32,total_candidates=64,total_actual_edit_attempts=192,new_original_photos=0,new_independent_groups=0,
        editor=EDITOR_VERSION,schema=SCHEMA_VERSION,labels_admitted=0,GPU_hours=0,training_started=False,
        source_isolation='All32 are known original TRAIN groups outside23 prior quality-refusal groups, disjoint from evaluation original group IDs; existing source/near-duplicate isolation inherited, not new independent photographs.',
        limitation='Synthetic matched noise conditions may not represent real camera noise. All labels pending direct AI and native-size review; failed finite attempts never prove mathematical impossibility.'))
    print('64 candidates/192 actual tool attempts;32 existing TRAIN groups;0 labels admitted or GPU use.')


if __name__=='__main__':main()
