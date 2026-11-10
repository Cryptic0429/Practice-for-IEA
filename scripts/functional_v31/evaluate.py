"""Compare frozen R30 with new global and region-aware selectors."""
from collections import Counter

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.special import expit

from functional_v1.common import ROOT,read,sha,stamp,write
from functional_v15.editor import load_rgb
from functional_v30.constants import RECIPES
from functional_v30.editor import apply
from functional_v30.select import NAMES,calibrate,features,predict,train
from .face import FACE_NAMES,FaceRegions,measures
from .render import DATA,EXP,PLAN,R30


def checked(path,digest):
    if sha(path)!=digest:
        raise ValueError(f'Frozen artifact changed: {path}')
    return read(path)


def rows_and_detector():
    plan=read(PLAN)
    old={name:checked(R30/f'{name}.json',plan[key]) for name,key in (
        ('replay','r30_replay_sha256'),('visual-review','r30_visual_sha256'),
        ('fit','r30_fit_sha256'),('result','r30_result_sha256'))}
    notes=read(EXP/'visual-review.json')
    if notes['protocol_sha256']!=sha(PLAN):
        raise ValueError('Visual judgment from another protocol')
    replay=checked(EXP/'replay.json',notes['replay_sha256'])
    isolation=checked(EXP/'isolation.json',notes['isolation_sha256'])
    checked(EXP/'details.json',notes['details_sha256'])
    if isolation['flags']:
        raise ValueError('Held source isolation failed')
    new_labels={c['id']:c for c in notes['cases']}
    prior_labels={c['id']:c for c in old['visual-review']['cases']}
    active=[(c,new_labels[c['id']]) for c in replay['cases']]
    active.extend((dict(c,cohort='regression' if c['id']=='p162' else 'fit'),prior_labels[c['id']])
                  for c in old['replay']['cases'] if c['cohort']=='fit' or c['id'] in ('p479','p162'))
    detector=FaceRegions()
    rows=[]
    overlay_folder=EXP/'face-locations'
    overlay_folder.mkdir(exist_ok=True)
    for c,note in active:
        cid=c['id']
        src=ROOT/c['source']
        if sha(src)!=c['source_sha256'] or sha(ROOT/c['panel'])!=note['panel_sha256']:
            raise ValueError(f'Visual source changed: {cid}')
        if note.get('detail') and sha(ROOT/note['detail']['path'])!=note['detail']['sha256']:
            raise ValueError(f'Detail crop changed: {cid}')
        if c['cohort']=='held' and c['source_sha256']!=isolation['source_hashes'][cid]:
            raise ValueError(f'Held image did not match isolated original: {cid}')
        rgb=load_rgb(src)
        pixels={}
        for name in RECIPES:
            a=c['outputs'][name]
            p=ROOT/a['path']
            if sha(p)!=a['sha256']:
                raise ValueError('Candidate file changed')
            pixels[name]=load_rgb(p)
        x,diag=features(rgb,pixels,c['requests']['zh'])
        xx,diag2=features(rgb,pixels,c['requests']['en'])
        if diag!=diag2 or not np.array_equal(x,xx):
            raise ValueError(f'Literal request strengths disagree: {cid}')
        face,loc=measures(detector,rgb,pixels,c['requests'])
        overlay=None
        if loc['face_required']:
            im=Image.fromarray(rgb)
            w,h=im.size
            small=ImageOps.contain(im,(640,480))
            draw=ImageDraw.Draw(small)
            for x1,y1,x2,y2 in loc['boxes']:
                draw.rectangle((round(x1*small.width/w),round(y1*small.height/h),
                                round(x2*small.width/w),round(y2*small.height/h)),
                               outline=(255,230,0),width=2)
            path=overlay_folder/f'{cid}.jpg'
            small.save(path,quality=94)
            overlay=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
        rows.append(dict(c,grades=note['grades'],features=x,region_features=np.c_[x,face],
                         face_features=face.tolist(),face=loc,diagnostic=diag,overlay=overlay))
    return rows,old,detector


def baseline(x,params):
    std=(x-params['mean'])/params['scale']
    p=expit(std @ np.asarray(params['coefficient']).T+params['intercept']).ravel()
    i=int(p.argmax())
    return (list(RECIPES)[i] if p[i]>=params['threshold'] else None),p.tolist()


def train_models(fit):
    out,params={},{}
    for name,key in [('global','features'),('region','region_features')]:
        pool=[dict(c,features=c[key]) for c in fit]
        threshold,oof=calibrate(pool)
        model=train(pool)
        scaler,lr=model.steps[0][1],model.steps[1][1]
        out[name]=(model,threshold,key)
        params[name]=dict(threshold=threshold,oof=oof,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                          coefficient=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),
                          feature_names=NAMES if name=='global' else (*NAMES,*FACE_NAMES))
    return out,params


def main():
    rows,old,detector=rows_and_detector()
    fit=[r for r in rows if r['cohort']=='fit']
    held=[r for r in rows if r['cohort']=='held']
    regression=next(r for r in rows if r['cohort']=='regression')
    if {r['group_id'] for r in fit}&{r['group_id'] for r in held+[regression]}:
        raise ValueError('Source leakage')
    previous=next(r for r in old['result']['cases'] if r['id']==regression['id'])
    saved,scores=baseline(regression['features'],old['fit'])
    if saved!=previous['chosen']['learned']['recipe'] or not np.allclose(scores,list(previous['scores'].values())):
        raise ValueError('Frozen R30 scores did not reproduce')
    models,params=train_models(fit)
    fit_path=EXP/'fit.json'
    write(fit_path,dict(at=stamp(),protocol_sha256=sha(PLAN),visual_sha256=sha(EXP/'visual-review.json'),
        candidate_fit_groups=len(fit),no_usable_fit_groups=sum('complete' not in r['grades'].values() for r in fit),
        models=params,opencv_version=cv2.__version__,
        local_detector_hashes={p.name:sha(p) for p in detector.paths},GPU_hours=0))
    route=read(EXP/'route-result.json')
    if route['protocol_sha256']!=sha(PLAN):
        raise ValueError('Router result changed')
    by_route={(r['id'],r['language']):r['guarded'] for r in route['rows']}
    cases=[]
    for r in held+[regression]:
        best,bp=baseline(r['features'],old['fit'])
        chosen={'r30':best}
        all_scores={'r30':bp}
        for name,(model,threshold,key) in models.items():
            recipe,prob,values=predict(model,dict(r,features=r[key]))
            chosen[name]=recipe if prob>=threshold else None
            all_scores[name]=values
        found=not r['face']['face_required'] or bool(r['face']['boxes'])
        chosen['region_require_detection']=chosen['region'] if found else None
        if r['cohort']=='held':
            accepted=all(by_route[r['id'],l]=='global_down' for l in ('zh','en'))
            chosen['guarded_region']=chosen['region_require_detection'] if accepted else None
        decisions={}
        source=load_rgb(ROOT/r['source'])
        for recipe in set(chosen.values())-{None}:
            output=r['outputs'][recipe]['path']
            if not np.array_equal(apply(source,recipe),load_rgb(ROOT/output)):
                raise ValueError('Selected recipe did not reproduce pixels')
        for name,recipe in chosen.items():
            if recipe is None:
                decisions[name]=dict(recipe=None,verdict='deferred',output=None)
            else:
                output=r['outputs'][recipe]['path']
                decisions[name]=dict(recipe=recipe,verdict=r['grades'][recipe],output=output)
        cases.append(dict(id=r['id'],cohort=r['cohort'],domain=r['domain'],grades=r['grades'],
            face=r['face'],face_features=r['face_features'],overlay=r['overlay'],
            scores=all_scores,chosen=decisions))
    real=[r for r in cases if r['cohort']=='held']
    summary={}
    for method in real[0]['chosen']:
        verdicts=Counter(r['chosen'][method]['verdict'] for r in real)
        true_defer=sum(r['chosen'][method]['verdict']=='deferred' and 'complete' not in r['grades'].values() for r in real)
        summary[method]=dict(verdicts=dict(verdicts),correct_no_output=true_defer,
            missed_feasible=sum(r['chosen'][method]['verdict']=='deferred' and 'complete' in r['grades'].values() for r in real))
    row_path=DATA/'candidate-records.json'
    write(row_path,[{k:r[k] for k in ('id','cohort','domain','group_id','requests','source','source_sha256','outputs','grades','face')}
                    for r in rows])
    write(EXP/'result.json',dict(at=stamp(),fit_sha256=sha(fit_path),route_sha256=sha(EXP/'route-result.json'),
        candidate_records_sha256=sha(row_path),held_groups=len(real),
        oracle_completed=sum('complete' in r['grades'].values() for r in real),
        summary=summary,cases=cases,GPU_hours=0,deployed=False,final_acceptance_opened=False))
    print(summary,flush=True)


if __name__=='__main__':
    main()
