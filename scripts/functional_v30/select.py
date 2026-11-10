"""Fit a CPU quality selector on photographed comparisons and evaluate held photos."""
from collections import Counter
import re

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb
from .constants import EXP,PLAN,RECIPES
from .editor import LUMA,apply

NOTICEABLE = re.compile(r'明显|显著|大幅|\b(?:noticeably|substantially|markedly)\b',re.I)
NAMES = ('plate_area','source_dark_share','source_median','plate_weighted_drop',
         'new_dark_share','midtone_drop','shoulder','strength_mismatch')


def features(source,outputs,request):
    h,w = source.shape[:2]
    ratio = min(1,640/max(h,w))
    shape = (max(1,round(w*ratio)),max(1,round(h*ratio)))
    img=cv2.resize(source,shape,interpolation=cv2.INTER_AREA).astype(np.float32)/255
    lum=img@LUMA
    neutral_white = ((img.min(axis=2)>=250/255)&(np.ptp(img,axis=2)<=5/255)).astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(neutral_white,connectivity=8)
    plate = labels==int(1+stats[1:,cv2.CC_STAT_AREA].argmax()) if len(stats)>1 else np.zeros(lum.shape,bool)
    area,dark=float(plate.mean()),float((lum<.12).mean())
    mid=(lum>.18)&(lum<.72)
    strength=30 if NOTICEABLE.search(request) else 12
    rows=[]
    for name,out in outputs.items():
        result=cv2.resize(out,shape,interpolation=cv2.INTER_AREA).astype(np.float32)/255
        y=result@LUMA
        mode,value=RECIPES[name]
        rows.append([area,dark,float(np.median(lum)),float((lum-y)[plate].mean()*area) if plate.any() else 0,
                     float((y<.07).mean()-(lum<.07).mean()),float((lum-y)[mid].mean()) if mid.any() else 0,
                     float(mode=='shoulder'),abs(-value-strength)/30])
    return np.array(rows),dict(plate_area=area,dark_share=dark,noticeable=strength==30)


def train(rows):
    x=np.concatenate([r['features'] for r in rows])
    y=np.array([v=='complete' for r in rows for v in r['grades'].values()])
    return make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',
                          solver='liblinear',random_state=20261107)).fit(x,y)


def predict(model,row):
    p=model.predict_proba(row['features'])[:,1]
    i=int(np.argmax(p))
    return list(RECIPES)[i],float(p[i]),p.tolist()


def calibrate(rows):
    oof=[]
    for r in rows:
        model=train([c for c in rows if c['id']!=r['id']])
        choice,prob,all_scores=predict(model,r)
        oof.append(dict(id=r['id'],choice=choice,probability=prob,
                        verdict=r['grades'][choice],all_scores=all_scores))
    possibilities={r['probability'] for r in oof}|{float(np.nextafter(max(r['probability'] for r in oof),np.inf))}
    legal=[]
    for t in possibilities:
        selected=[r for r in oof if r['probability']>=t]
        if all(r['verdict']=='complete' for r in selected):
            legal.append((len(selected),t))
    return max(legal)[-1],oof


def heuristic(info):
    if info['noticeable'] and info['dark_share']>.38:
        return None
    prefix='shoulder' if info['plate_area']>=.02 else 'linear'
    return prefix+('30' if info['noticeable'] else '12')


def load():
    plan=read(PLAN)
    ref=read(EXP/'visual-review.json')
    iso=read(EXP/'isolation.json')
    replay=read(EXP/'replay.json')
    if iso['flags'] or ref['protocol_sha256']!=sha(PLAN) or (
        ref['replay_sha256'],ref['isolation_sha256'])!=(sha(EXP/'replay.json'),sha(EXP/'isolation.json')):
        raise ValueError('Photo group/reference mismatch')
    notes={r['id']:r for r in ref['cases']}
    cases=[]
    for c in replay['cases']:
        note=notes[c['id']]
        if sha(ROOT/c['source'])!=c['source_sha256'] or sha(ROOT/c['panel'])!=note['panel_sha256']:
            raise ValueError('Original or review panel changed')
        if c['cohort']=='held' and c['source_sha256']!=iso['source_hashes'][c['id']]:
            raise ValueError('Held original was not isolated')
        outputs={}
        for name,v in c['outputs'].items():
            if sha(ROOT/v['path'])!=v['sha256']:
                raise ValueError('Rendered result changed')
            outputs[name]=load_rgb(ROOT/v['path'])
        source=load_rgb(ROOT/c['source'])
        # Same request semantics, one outcome per original group; literal words determine the strength hint.
        zh,info=features(source,outputs,c['requests']['zh'])
        en,info_en=features(source,outputs,c['requests']['en'])
        if not np.array_equal(zh,en) or info!=info_en:
            raise ValueError(f"Literal strength discrepancy: {c['id']}")
        cases.append(dict(c,grades=note['grades'],features=zh,diagnostic=info))
    return cases


def main():
    cases=load()
    fit=[c for c in cases if c['cohort']=='fit']
    held=[c for c in cases if c['cohort']=='held']
    if {c['group_id'] for c in fit}&{c['group_id'] for c in held}:
        raise ValueError('Original group overlap')
    threshold,oof=calibrate(fit)
    model=train(fit)
    scal,log=model.steps[0][1],model.steps[1][1]
    fitting=dict(at=stamp(),protocol_sha256=sha(PLAN),visual_review_sha256=sha(EXP/'visual-review.json'),
        threshold=threshold,oof=oof,feature_names=NAMES,mean=scal.mean_.tolist(),
        scale=scal.scale_.tolist(),coefficient=log.coef_.tolist(),intercept=log.intercept_.tolist(),
        fit_groups=len(fit),GPU_hours=0)
    write(EXP/'fit.json',fitting)
    route_result=read(EXP/'route-result.json')
    if route_result['protocol_sha256']!=sha(PLAN):
        raise ValueError('Route result from another experiment')
    routed={(r['id'],r['language']):r for r in route_result['rows'] if r['cohort']=='held_photo'}
    cases_out=[]
    for r in held:
        choice,p,scores=predict(model,r)
        learned=choice if p>=threshold else None
        choices=dict((name,name) for name in RECIPES)
        choices.update(heuristic=heuristic(r['diagnostic']),learned=learned)
        choices['head_fixed_linear30']='linear30' if all(routed[r['id'],lang]['head']=='global_down' for lang in ('zh','en')) else None
        choices['guarded_learned']=learned if all(routed[r['id'],lang]['guarded']=='global_down' for lang in ('zh','en')) else None
        source=load_rgb(ROOT/r['source'])
        for name,v in r['outputs'].items():
            if not np.array_equal(apply(source,name),load_rgb(ROOT/v['path'])):
                raise ValueError('Chosen path did not reproduce candidate pixels')
        entries={}
        for method,name in choices.items():
            if name is None:
                entries[method]=dict(recipe=None,verdict='deferred',output=None)
                continue
            v=r['outputs'][name]
            entries[method]=dict(recipe=name,verdict=r['grades'][name],output=v['path'])
        cases_out.append(dict(id=r['id'],domain=r['domain'],grades=r['grades'],diagnostic=r['diagnostic'],
            scores=dict(zip(RECIPES,scores)),chosen=entries))
    summary={name:dict(Counter(r['chosen'][name]['verdict'] for r in cases_out))
             for name in cases_out[0]['chosen']}
    route_summary={}
    for cohort in ('held_photo','held_control','r29_regression'):
        rows=[r for r in route_result['rows'] if r['cohort']==cohort]
        ids={r['id'] for r in rows}
        route_summary[cohort]={m:dict(rows=len(rows),correct=sum(r[m]==r['expected'] for r in rows),
            bilingual=sum(all(r[m]==r['expected'] for r in rows if r['id']==cid) for cid in ids),
            active_errors=sum(r[m] not in (r['expected'],'defer') for r in rows),
            deferred_expected_active=sum(r[m]=='defer' and r['expected']!='defer' for r in rows))
            for m in ('head','explicit','guarded')}
    write(EXP/'result.json',dict(at=stamp(),fit_sha256=sha(EXP/'fit.json'),
        route_sha256=sha(EXP/'route-result.json'),fit_groups=len(fit),held_groups=len(held),
        oracle_completed=sum('complete' in c['grades'].values() for c in held),
        image_summary=summary,route_summary=route_summary,cases=cases_out,
        GPU_hours=0,deployed=False,final_acceptance_opened=False))
    print(summary,route_summary,flush=True)


if __name__=='__main__':
    main()
