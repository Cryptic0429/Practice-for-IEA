"""Training-only manual target proposals and explicit synthetic-input revisions.

No changed request, accepted dataset, development inference or model training.
"""
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, sha, stamp, validate
from functional_v1.editor import load_rgb, save
from functional_v1.curate import font
from .editor_soft import edit
from .diagnose_editor import EXP


def panel(path, title, request, items):
    canvas=Image.new('RGB',(1800,1320),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,10),title,fill='black',font=font(23))
    draw.text((12,46),request[:135],fill='black',font=font(21))
    for i,(label,source) in enumerate(items):
        x,y=(i%2)*900,105+(i//2)*600
        draw.text((x+12,y),label,fill='black',font=font(21))
        with Image.open(ROOT/source) as im:
            fitted=ImageOps.contain(im.convert('RGB'),(870,550),Image.Resampling.LANCZOS)
        canvas.paste(fitted,(x+12+(870-fitted.width)//2,y+38+(550-fitted.height)//2))
    canvas.save(path,quality=95,subsampling=0)


def main():
    out=EXP/'label-repairs';assert not (out/'protocol.json').exists()
    source=ROOT/'data/training/functional-v6-20261105/cases.json'
    cases={c['id']:c for c in read(source)}
    review=EXP/'soft-label-candidates/panel-review-summary.json'
    assert read(review)['changed_semantic_cases']==164
    failed=read(review)['incomplete_or_uncertain_candidates']
    compounds=[cases[r['case_id']] for r in failed if r['case_id'].endswith('-compound')]
    exposure_ids=[r['case_id'] for r in failed if 'restore-exposure-' in r['case_id']]
    assert len(compounds)==20 and len(exposure_ids)==5
    assert all(c['split']=='train' and c['instruction_origin'].startswith('AI-') for c in compounds)
    proposals={}
    for cid in exposure_ids:
        assert set(cases[cid]['parameters'])=={'exposure'}
        proposals[cid]=[dict(cases[cid]['parameters'],contrast=12),dict(cases[cid]['parameters'],contrast=22)]
    proposals.update({
        'native-train-n068':[
            dict(exposure=-5,contrast=45,sepia=45,temperature=25,tint=-15),
            dict(exposure=-10,contrast=60,sepia=65,temperature=15,tint=-15)],
        'native-train-n029':[
            dict(exposure=-20,contrast=25,saturation=15,highlights=-30),
            dict(exposure=-15,contrast=40,saturation=15,highlights=-15)],
        'native-train-n093':[
            dict(exposure=10,contrast=60,saturation=15,temperature=10,tint=-5),
            dict(exposure=15,contrast=80,saturation=10,temperature=15,tint=-10)],
        'native-train-n180':[
            dict(exposure=15,contrast=25,saturation=45,shadows=30,highlights=-10),
            dict(exposure=20,contrast=35,saturation=60,shadows=35,highlights=-15)],
        'native-train-x035':[
            dict(exposure=-25,saturation=-25,highlights=-25),
            dict(exposure=-15,brightness=-15,saturation=-30,highlights=-15)],
    })
    assert len(proposals)==10
    assert all(cases[cid]['split']=='train' for cid in proposals)
    for values in proposals.values():
        for parameters in values:validate(parameters)
    synthetic=[]
    for c in compounds:
        index=int(hashlib.sha256(('v7-compound-input'+c['id']).encode()).hexdigest()[:8],16)%3
        magnitude=(20,30,40)[index];contrast=(5,8,10)[index]
        synthetic.append(dict(replaces_case_id=c['id'],new_case_id=c['id']+'-v7-underexposed',
                              canonical_photo=c['canonical_photo'],instructions=c['instructions'],
                              group_id=c['group_id'],domain=c['domain'],
                              input_degradation=dict(exposure=-magnitude),parameters=dict(exposure=magnitude,contrast=contrast)))
    write(out/'protocol.json',dict(
        at=stamp(),status='frozen_before_training_only_label_repair_outputs',source_cases_sha256=sha(source),
        source_panel_review_sha256=sha(review),code_sha256=sha(__file__),editor_sha256=sha(Path(__file__).with_name('editor_soft.py')),
        same_input_target_proposals=proposals,explicit_synthetic_input_revisions=synthetic,
        rationale='Five exposure labels look flat under the new curve; add two modest contrast proposals without changing input/request. Five natural requests need stronger visible colour/contrast/darkening targets, still with original inputs/requests. Twenty AI-authored compound tasks have weak changes on already lit inputs: create explicit bounded underexposure from the same canonical training photos and propose restoration plus slight contrast. Old failed cases remain in the audit and are not reclassified as successes.',
        selection='Every proposed output must be directly visually inspected. No numerical loss, automatic admission, or quota filling. Full-size followup for material uncertainty.',
        source_scope='Original round6 training groups only. English/Chinese requests unchanged and grouped. Synthetic inputs clearly identified; natural inputs are never replaced.',
        new_training_admission=False,new_training_started=False,live_tools_changed=False,heldout_access=False,GPU_hours=0))
    current={r['case_id']:r for r in read(EXP/'soft-label-candidates/manifest.json')['cases']}
    records=[];sheets=[]
    for cid,values in proposals.items():
        c=cases[cid];assert sha(ROOT/c['image'])==c['input_sha256'];rgb=load_rgb(ROOT/c['image']);items=[]
        for p in values:
            result=edit(rgb,p,'bounded-fixed');label=hashlib.sha256(result.tobytes()).hexdigest()[:12]
            path=out/'images'/f'{cid}-{label}.png';save(path,result)
            records.append(dict(case_id=cid,label=label,parameters=p,input_image=c['image'],input_sha256=c['input_sha256'],
                                image=str(path.relative_to(ROOT)),image_sha256=sha(path),instructions=c['instructions'],kind='same_input_manual_parameter_proposal',admitted=False))
            items.append((label,str(path.relative_to(ROOT))))
        items.sort()
        path=out/(cid+'.jpg')
        panel(path,cid,c['instructions']['en'],[('INPUT',c['image']),('CURRENT FAILED CANDIDATE',current[cid]['new_candidate'])]+items)
        sheets.append(dict(case_id=cid,sheet=str(path.relative_to(ROOT)),sha256=sha(path),kind='parameter_repair'))
    for r in synthetic:
        c=cases[r['replaces_case_id']];rgb=load_rgb(ROOT/r['canonical_photo'])
        original=ROOT/c['image'];assert sha(original)==c['input_sha256']
        before=edit(rgb,r['input_degradation'],'bounded-fixed')
        input_path=out/'images'/(r['new_case_id']+'-input.png');save(input_path,before)
        result=edit(before,r['parameters'],'bounded-fixed')
        result_path=out/'images'/(r['new_case_id']+'-target.png');save(result_path,result)
        records.append(dict(**r,case_id=r['new_case_id'],kind='explicit_revised_AI_synthetic_input',
                            input_image=str(input_path.relative_to(ROOT)),input_sha256=sha(input_path),
                            image=str(result_path.relative_to(ROOT)),image_sha256=sha(result_path),admitted=False))
        path=out/(r['new_case_id']+'.jpg')
        panel(path,r['new_case_id'],c['instructions']['en'],
              [('OLD INPUT',c['image']),('OLD FAILED LABEL',current[c['id']]['new_candidate']),
               ('REVISED SYNTHETIC INPUT',str(input_path.relative_to(ROOT))),('NEW CANDIDATE',str(result_path.relative_to(ROOT)))])
        sheets.append(dict(case_id=r['new_case_id'],sheet=str(path.relative_to(ROOT)),sha256=sha(path),kind='explicit_input_revision'))
    write(out/'manifest.json',dict(at=stamp(),protocol_sha256=sha(out/'protocol.json'),records=records,sheets=sheets,
                                  same_input_parameter_candidates=20,revised_AI_inputs=20,new_training_admission=False))
    print('Generated20same-input parameter proposals and20explicit synthetic input revisions; no labels admitted.')


if __name__=='__main__':main()
