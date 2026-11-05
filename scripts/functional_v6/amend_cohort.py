"""Replace unapprovable TRAINING targets, preserving every rejected attempt and heldout."""
import json
import textwrap
from PIL import Image,ImageOps,ImageDraw
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,edit,save
from functional_v1.curate import font
from .source_bank import EXP,DATA
from .isolation import observed_bank

SPECS=[
 ('n034','train-native-034','compound','提高亮度，并增强色彩。',{'shadows':50,'saturation':12},{'shadows':70,'saturation':18}),
 ('x251','train-extension-251','saturation','增强色彩，让颜色更鲜明。',{'saturation':25},{'saturation':40}),
 ('x348','train-extension-348','lighting','把照片稍微提亮一点。',{'exposure':15},{'exposure':25}),
]

def main():
    out=EXP/'native-cohort-amendment'; assert not (out/'amendment.json').exists()
    repairs=read(EXP/'native-label-revision/observations.json')
    excluded={r['case_id']:r for r in repairs if r['selected'] is None}
    assert set(excluded)=={'native-train-n041','native-train-x121','native-train-x351'}
    bank,notes,_=observed_bank();old=read(DATA/'native-training-candidates.json')
    isolation=read(EXP/'planning/observed-source-isolation.json')
    for path,digest in isolation['source_files'].items():assert sha(ROOT/path)==digest
    used={c['group_id'] for c in old}
    held={c['group_id'] for file in ('development-native-cases.json','acceptance-native-cases.json') for c in read(DATA/file)}
    older={c['group_id'] for c in read(ROOT/'data/training/functional-v4-20261105/cases.json')}
    added=[];proposals=[]
    canvas=Image.new('RGB',(1800,2220),'white');d=ImageDraw.Draw(canvas)
    d.text((12,10),'TRAINING source amendment | authored targets, pending direct admission',font=font(25),fill='black')
    for i,(short,sid,family,zh,e,f) in enumerate(SPECS):
        source,note=bank[sid],notes[sid]
        assert note['source_usable'] and note['request_scope']=='global' and source['split']=='train'
        assert source['group_id'] not in used|held|older; used.add(source['group_id'])
        assert source['source_sha256']==sha(source['source_image'])
        assert not any(x['a']==sid or (x['reason']=='near_current_sources' and x['b']==sid) for x in isolation['collisions'])
        c=dict(old[0]);image=DATA/'inputs'/f'native-train-{short}.png';save(image,load_rgb(source['source_image']))
        c.update(id=f'native-train-{short}',source_id=sid,group_id=source['group_id'],domain=note['domain'],family=family,
            image=str(image.relative_to(ROOT)),input_sha256=sha(image),instructions={'en':source['instruction'],'zh':zh},
            expected_change=zh+' 所有请求项均须可见完成，并保持基本图像质量。',source_record_id=source['record_id'],
            original_source_sha256=source['source_sha256'],native_annotation=source['native_annotation'],
            source_classification_evidence=note['evidence'])
        added.append(c); choices={}
        y=55+i*715;d.text((12,y),c['id'],font=font(23),fill='black')
        for j,line in enumerate(textwrap.wrap(c['instructions']['en'],123)):d.text((12,y+32+j*24),line,font=font(21),fill='black')
        paths=[('BEFORE',image)]
        for code,p in [('E',e),('F',f)]:
            path=out/'images'/f'{c["id"]}-{code}.png';save(path,edit(load_rgb(image),p))
            choices[code]=dict(parameters=p,image=str(path.relative_to(ROOT)),sha256=sha(path));paths.append((code,path))
        proposals.append(dict(case_id=c['id'],input_sha256=c['input_sha256'],candidates=choices))
        for j,(code,path) in enumerate(paths):
            x=j*600;d.text((x+12,y+90),code,font=font(23),fill='black')
            with Image.open(path) as im:fit=ImageOps.contain(im.convert('RGB'),(575,570),Image.Resampling.LANCZOS)
            canvas.paste(fit,(x+12+(575-fit.width)//2,y+125+(570-fit.height)//2))
    path=out/'proposals.jpg';canvas.save(path,quality=97,subsampling=0)
    write(DATA/'native-training-additions.json',added)
    write(out/'amendment.json',dict(at=stamp(),status='training_sources_amended_before_any_round6_training',
        original_cases_sha256=sha(DATA/'native-training-candidates.json'),excluded=list(excluded),
        reasons={'native-train-n041':'Both model proposals and two authored global corrections lacked clean brown-cast removal; do not supervise a purple or partially completed target.',
          'native-train-x121':'Both models clipped clouds. Two authored preservation attempts darkened/crushed trees, missing significant brightening. No fully acceptable target established with frozen tools.',
          'native-train-x351':'Strong global/shadow lifts expose extensive coloured block noise. No acceptable significant-brightening target established without unavailable denoising.'},
        excluded_targets_retained=True, excluded_may_not_enter_training=True,
        additions=[c['id'] for c in added],additions_sha256=sha(DATA/'native-training-additions.json'),
        evidence='Three original training-reserved, previously source-reviewed and isolation-audited groups. Source-only reinspection saved in sources.png. Original English retained with AI Chinese translation. These are purposively curated training data, not unbiased success estimates.',
        target_proposals=proposals,panel=str(path.relative_to(ROOT)),panel_sha256=sha(path),
        review_pending=True, heldout_cases_or_gates_changed=False,round6_training_started=False,
        source_isolation_sha256=sha(EXP/'planning/observed-source-isolation.json'),implementation_sha256=sha(__file__)))
    print('Three replacements proposed; still no training admission')

if __name__=='__main__':main()
