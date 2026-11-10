"""Execute all changed labels; unchanged pixels inherit explicit prior evidence only."""
import copy
import os
import textwrap
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.curate import font
from functional_v1.editor import load_rgb,save
from .outcomes import execute,EDITOR_VERSION,SCHEMA_VERSION
EXP=ROOT/'experiments/functional-v11-20261106'


def panel(cid,instruction,images,path):
    canvas=Image.new('RGB',(600*len(images),750),'white');draw=ImageDraw.Draw(canvas)
    draw.text((16,12),cid+' | TRAINING CANDIDATE, not model output',font=font(23),fill='black')
    for i,line in enumerate(textwrap.wrap(instruction,140)):
        draw.text((16,48+i*25),line,font=font(20),fill='black')
    for i,(label,p) in enumerate(images):
        draw.text((i*600+12,135),label,font=font(20),fill='black')
        with Image.open(p) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
        canvas.paste(fitted,(i*600+10+(580-fitted.width)//2,170+(550-fitted.height)//2))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=95,subsampling=0)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    dest=EXP/'label-candidates/manifest.json';assert not dest.exists()
    origins=[ROOT/'data/training/functional-v7-20261105',ROOT/'data/training/functional-v10-20261106']
    original=[(c,origins[0]) for c in read(origins[0]/'cases.json')]
    original += [(c,origins[1]) for c in read(origins[1]/'cases.json') if c['id'].startswith('visual-risk-')]
    assert len(original)==662 and len({c['id'] for c,_ in original})==662
    groups={c['group_id']:c['domain'] for c,_ in original};assert len(groups)==160 and Counter(groups.values())==dict(portrait=80,landscape=80)
    changed=[];unchanged=[];allcases=[]
    for old,source in original:
        c=copy.deepcopy(old);cid=c['id'];assert sha(ROOT/c['image'])==c['input_sha256']
        rgb=load_rgb(ROOT/c['image']);decision=c['decision'];result=execute(rgb,decision)
        provenance=dict(cases=str((source/'cases.json').relative_to(ROOT)),cases_sha256=sha(source/'cases.json'),admission=str((source/'admission.json').relative_to(ROOT)),admission_sha256=sha(source/'admission.json'),case_id=cid,previous_label_evidence=c['label_evidence'])
        is_changed=c['action']=='edit' and any(c['parameters'].get(k,0)<0 for k in ('exposure','brightness'))
        c.update(editor=EDITOR_VERSION,schema=SCHEMA_VERSION,previous_record=provenance,status='CPU_candidate_not_admitted')
        if is_changed:
            assert not np.array_equal(result['result'],load_rgb(ROOT/old['target'])),cid
            output=EXP/'label-candidates/images'/f'{cid}.png';save(output,result['result'])
            c.update(target=str(output.relative_to(ROOT)),target_sha256=sha(output))
            sheet=EXP/'label-candidates/panels'/f'{cid}.jpg'
            p=panel(cid,c['instructions']['en'],[('INPUT',ROOT/c['image']),('OLD FROZEN LABEL',ROOT/old['target']),('NEW NEGATIVE CURVE',output)],sheet)
            changed.append(dict(id=cid,kind='negative_exposure_or_brightness_reexecution',domain=c['domain'],group_id=c['group_id'],
                instruction=c['instructions'],parameters=c['parameters'],input=dict(path=c['image'],sha256=c['input_sha256']),
                previous_target=dict(path=old['target'],sha256=old['target_sha256']),candidate=dict(path=c['target'],sha256=c['target_sha256']),panel=p,status='not_viewed_not_admitted'))
        else:
            if c['action']=='decline':assert result['result'] is None and c['target'] is None
            else:
                assert sha(ROOT/c['target'])==c['target_sha256'] and np.array_equal(result['result'],load_rgb(ROOT/c['target'])),cid
                if c['action']=='keep':assert np.array_equal(rgb,result['result'])
            unchanged.append(dict(id=cid,action=c['action'],target_sha256=c['target_sha256'],exact_pixel_or_no_output_replay=True,source=provenance))
        allcases.append(c)
    assert len(changed)==51 and len(unchanged)==611
    selected=[1,2,5,7,9,11,12,13,17,20]
    requests=[
     ('请减弱整张照片的棕黄色偏色，保留自然肤色和细节。','Reduce the overall brown-yellow colour cast while retaining natural skin and detail.'),
     ('请去掉画面过重的棕黄暖色，让颜色自然一些。','Tone down the excessive brownish warmth and keep the colours natural.'),
     ('把整体偏棕黄的颜色调淡一些，保持人物和场景原样。','Make the overall brown-yellow cast less pronounced while preserving the person and scene.'),
     ('请减少整体棕色偏色，保留人物与白衣的自然颜色。','Reduce the global brownish cast while retaining natural colours in the people and white clothing.'),
     ('请让照片少一点棕黄感，保留自然肤色。','Give the photo less of a brown-yellow cast while retaining natural skin colour.'),
     ('请减弱整张照片的棕黄暖色偏色，保持景物和细节。','Reduce the overall brown-yellow warm cast while retaining the scene and detail.'),
     ('让画面的棕黄色偏色更弱一些，保持自然景物颜色。','Make the brown-yellow colour cast weaker while keeping natural scene colours.'),
     ('请去掉多余的棕黄暖色，让照片颜色自然一些。','Remove the excess brownish warmth so that the photograph has more natural colours.'),
     ('请把整体棕黄偏色减轻，保留水面、树木和细节。','Lessen the overall brown-yellow cast while retaining the water, trees and detail.'),
     ('请减弱照片里的棕黄暖色，不要改变场景内容。','Tone down the brown-yellow warmth without changing the scene content.')]
    sources={c['id']:c for c in read(ROOT/'experiments/functional-v10-20261106/counterfactual-candidates/manifest.json')['cases']}
    cool=[]
    for number,(index,request) in enumerate(zip(selected,requests),1):
        src=sources[f'visual-risk-{index:02d}'];canonical=src['images']['canonical'];assert sha(ROOT/canonical['path'])==canonical['sha256']
        prepared=np.rint(load_rgb(ROOT/canonical['path']).astype(np.float32)*.84+20).astype(np.uint8)
        assert prepared.min()>=20 and prepared.max()<=234
        # Explicit headroom/footroom avoids clipping an artificial additive cast;
        # this condition is documented and is not a new real-camera photograph.
        warm=execute(prepared,dict(action='edit',parameters=dict(temperature=24)))['result']
        assert warm.min()>0 and warm.max()<255
        decision=dict(action='edit',parameters=dict(temperature=-24));out=execute(warm,decision)['result']
        assert np.abs(out.astype(int)-prepared.astype(int)).max()<=1
        cid=f'cool-cast-{number:02d}';ip=EXP/'label-candidates/inputs'/f'{cid}.png';op=EXP/'label-candidates/images'/f'{cid}.png';save(ip,warm);save(op,out)
        sheet=panel(cid,request[1],[('AI WARM-CAST INPUT',ip),('EXECUTED CANDIDATE',op)],EXP/'label-candidates/panels'/f'{cid}.jpg')
        case=dict(id=cid,source_id=src['source_id'],group_id=src['group_id'],domain=src['domain'],split='train',family='white_balance',
            image=str(ip.relative_to(ROOT)),input_sha256=sha(ip),instructions=dict(zh=request[0],en=request[1]),
            expected_change='减弱整幅图的棕黄暖色偏色，保留自然颜色和内容。',must_preserve='主体、构图、皮肤／白衣／天空／景物颜色与细节；不新增明显蓝青偏色、过曝或压黑。',
            failure_conditions=['棕黄偏色未减弱','出现明显蓝青偏色或其他副作用','未请求的额外调整','非法负sepia或其他非法输出'],
            instruction_origin='AI-authored brown-yellow white-balance correction on an explicitly synthetic warm-cast training condition; not original GIER or user feedback.',
            original_source_sha256=src['original_source_sha256'],canonical_photo=src['canonical_photo'],
            source_partition='original training_pool; existing Round7 group',
            condition_origin=dict(canonical=canonical,preparation='round(0.84*RGB+20) to reserve channel headroom/footroom',temperature=24,synthetic=True,not_real_camera_capture=True),
            action='edit',decision=decision,parameters=decision['parameters'],target=str(op.relative_to(ROOT)),target_sha256=sha(op),
            editor=EDITOR_VERSION,schema=SCHEMA_VERSION,status='CPU_candidate_not_admitted')
        allcases.append(case);cool.append(dict(id=cid,kind='new_AI_brown_yellow_cast_correction',domain=src['domain'],group_id=src['group_id'],instruction=case['instructions'],parameters=case['parameters'],
            input=dict(path=case['image'],sha256=case['input_sha256']),candidate=dict(path=case['target'],sha256=case['target_sha256']),panel=sheet,status='not_viewed_not_admitted'))
    assert len(allcases)==672 and Counter(c['domain'] for c in cool)==dict(portrait=5,landscape=5)
    quality=[dict(id=c['id'],image=c['image'],instructions=c['instructions'],previous_record=c['previous_record'])
             for c in allcases if c['action']=='decline' and c['decision']['reason']=='unreliable_quality']
    assert len(quality)==44
    write(EXP/'candidate-cases.json',allcases)
    write(dest,dict(at=stamp(),status='CPU_candidates_not_reviewed_not_admitted',total_semantic_candidates=672,negative_curve_candidates=changed,cool_cast_candidates=cool,
        unchanged_exact_replay=unchanged,editor=EDITOR_VERSION,editor_sha256=sha(Path(__file__).with_name('editor.py')),
        inherited_quality_decisions_requiring_new_feasibility_review=quality,
        schema_sha256=sha(Path(__file__).with_name('outcomes.py')),all_cases_sha256=sha(EXP/'candidate-cases.json'),
        source_photos=160,photo_domains=dict(Counter(groups.values())),GPU_hours=0,training_started=False,
        limitations=['51 changed executed labels and10 new cast-correction targets require direct viewing before any admission.',
            '611 unchanged records inherit exact previous AI evidence and unresolved uncertainty, not fresh independent review.',
            'For44 quality declines, no-output replay checks the interface contract only. Their feasibility must be rechecked after the editor change before admission.',
            'Synthetic warm casts and low/high tone conditions are not independent original photographs or real camera corruption.']))
    print('CPU candidates only:51 changed labels +10 new cast corrections require direct viewing;611 exact replays;672 total,160groups80/80; no GPU.')

if __name__=='__main__':main()
