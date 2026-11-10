"""Revisit inherited quality declines after an operator changes what is feasible."""
import os
from collections import Counter
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from .prepare_labels import EXP,panel
from .outcomes import execute


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    path=EXP/'quality-revalidation/manifest.json';assert not path.exists()
    source=EXP/'candidate-cases.json';cases=[c for c in read(source) if c['action']=='decline' and c['decision']['reason']=='unreliable_quality']
    assert len(cases)==44
    rows=[];sheets=[];grouped={}
    for c in cases:
        cid=c['id'];rgb=load_rgb(ROOT/c['image']);assert sha(ROOT/c['image'])==c['input_sha256']
        if cid.startswith('visual-risk-') and '-severe-' in cid:
            settings=[dict(exposure=-40,brightness=80,contrast=-15,saturation=-10)] if cid.endswith('-light') else [dict(exposure=-20,brightness=25,contrast=10,shadows=25,saturation=15)]
            family='strong_synthetic_corruption'
        elif cid=='visual-risk-07-normal-light':
            settings=[dict(exposure=-20,brightness=45,contrast=20,highlights=-15,shadows=35),
                      dict(exposure=-30,brightness=65,contrast=8,highlights=-10,shadows=25)]
            family='normal_white_clothing_substantial_lightening'
        elif cid=='native-train-n190':
            settings=[dict(exposure=-40,contrast=60,shadows=35,saturation=35),
                      dict(exposure=-60,contrast=70,shadows=55,saturation=40),
                      dict(exposure=-25,contrast=45,highlights=-25,shadows=35,saturation=35)]
            family='large_contrast_and_colour_with_wave_detail'
        else:
            assert cid in ('native-train-n070','native-train-n127')
            settings=[dict(exposure=-40,brightness=80),
                      dict(exposure=-20,brightness=65,contrast=-10,shadows=25,saturation=-10)]
            family='real_low_light_portrait_substantial_lightening'
        group=cid.rsplit('-severe-',1)[0] if family=='strong_synthetic_corruption' else cid
        grouped.setdefault(group,dict(input=c['image'],instruction=c['instructions']['en'],candidates=[]))
        for index,p in enumerate(settings,1):
            rid=cid+f'-alt{index}';result=execute(rgb,dict(action='edit',parameters=p));assert result['status']=='edited'
            image=EXP/'quality-revalidation/images'/f'{rid}.png';save(image,result['result'])
            row=dict(id=rid,case_id=cid,group=group,family=family,domain=c['domain'],instructions=c['instructions'],parameters=p,
                input=dict(path=c['image'],sha256=c['input_sha256']),output=dict(path=str(image.relative_to(ROOT)),sha256=sha(image)),status='not_viewed_no_training_admission')
            rows.append(row);grouped[group]['candidates'].append(row)
    assert len(rows)==49 and len(grouped)==24
    for group,g in grouped.items():
        labels=[]
        for r in g['candidates']:
            label=('STRONG LIGHT' if r['case_id'].endswith('-severe-light') else 'MILD COMPOUND' if r['case_id'].endswith('-severe-compound') else 'ALT '+r['id'].rsplit('alt',1)[1])
            labels.append((label,ROOT/r['output']['path']))
        title=('Substantial lightening / mild contrast+brightness+colour; preserve natural detail, no conspicuous added noise.' if len(g['candidates'])==2 and group.startswith('visual-risk-') and group!='visual-risk-07-normal-light' else g['instruction'])
        sheet=panel(group,title,[('INPUT',ROOT/g['input']),*labels],EXP/'quality-revalidation/panels'/f'{group}.jpg')
        sheets.append(dict(group=group,candidates=[r['id'] for r in g['candidates']],panel=sheet))
    write(path,dict(at=stamp(),status='CPU_attempts_not_viewed_not_admitted',cases=44,source_cases_sha256=sha(source),attempts=rows,panels=sheets,
        family_counts=dict(Counter(r['family'] for r in rows)),GPU_hours=0,training_started=False,
        rule='A feasible, visually acceptable candidate takes precedence over an inherited quality refusal. No-output replay alone cannot establish feasibility after a tool change. Retain genuine uncertainty and failed alternatives; do not claim exhaustive search.'))
    print('CPU quality revalidation:49 executed alternatives across44 inherited declines,24 sheets; no review/admission/training yet.')

if __name__=='__main__':main()
