"""Explicit candidate revisions and SHA-bound admission, no automatic visual grading."""
import argparse, copy, collections, json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from functional_v1.curate import font
from functional_v1.editor import edit, load_rgb, save
from functional_v1.common import validate, system_prompt
from .common import ROOT, DATA, EXP, read, write, sha, stamp

REVISIONS = {
 'train-007-style-saturation-strong': {'saturation':25},
 'train-034-style-temperature-strong': {'temperature':22},
 'train-035-style-saturation-strong': {'saturation':25},
 'train-041-style-contrast-strong': {'contrast':22,'shadows':20},
 'train-045-style-temperature-strong': {'temperature':-22},
 'train-064-style-temperature-strong': {'temperature':22},
 'train-066-style-saturation-strong': {'saturation':25},
 'train-072-style-contrast-strong': {'contrast':22,'shadows':20},
 'train-102-style-temperature-strong': {'temperature':-20},
 'train-149-style-contrast-strong': {'contrast':22,'shadows':25},
 'train-214-style-contrast-strong': {'contrast':22,'shadows':25},
}

def revise():
    cases={c['id']:c for c in read(DATA/'train-cases.json')}
    notes=read(EXP/'label-review/notes-pass1.json')
    assert len(notes)==60 and {c for n in notes for c in n['rejected_cases']}==set(REVISIONS)
    out=EXP/'label-review/revisions';out.mkdir(parents=True,exist_ok=False)
    records=[]
    for cid,params in REVISIONS.items():
        c=cases[cid];target=DATA/'targets-revised'/f'{cid}.png'
        save(target,edit(load_rgb(ROOT/c['image']),params))
        panels=[('Input',ROOT/c['image']),('Weak target',ROOT/cases[cid.replace('-strong','-weak')]['target']),
                ('Rejected strong target',ROOT/c['target']),('Revised strong target',target)]
        sheet=Image.new('RGB',(2400,740),'white');d=ImageDraw.Draw(sheet)
        d.text((10,8),cid+' | REQUEST: '+c['instructions']['en'],font=font(22),fill='black')
        for j,(title,path) in enumerate(panels):
            d.text((j*600+10,50),title,font=font(22),fill='black')
            im=ImageOps.contain(Image.open(path).convert('RGB'),(580,620),Image.Resampling.LANCZOS)
            sheet.paste(im,(j*600+10+(580-im.width)//2,85+(620-im.height)//2))
        sp=out/f'{cid}.jpg';sheet.save(sp,quality=96,subsampling=0)
        records.append({'case_id':cid,'old_parameters':c['parameters'],'old_target_sha256':c['target_sha256'],
           'parameters':params,'target':str(target.relative_to(ROOT)),'target_sha256':sha(target),
           'sheet':str(sp.relative_to(ROOT)),'sheet_sha256':sha(sp),'status':'pending_direct_visual_review'})
    write(out/'manifest.json',records)

def criteria(c):
    c=copy.deepcopy(c)
    # Clarify before model inference: requested stylisation permits intended hue changes.
    c['must_preserve']='保留主体结构和明暗细节；仅按明确的色彩或风格要求改变色调，避免额外偏色。人像避免非预期肤色失真，风景保留天空与景物层次。'
    zh=c['instructions']['zh']
    for prefix in ('我想让这张照片','把这张照片'):
        if zh.startswith(prefix):zh=('对这张照片，请' if c['split']=='development' else '这次修图请')+zh[len(prefix):]
    c['instructions']['zh']=zh
    return c

def finalize():
    assert not (DATA/'train-accepted.json').exists()
    notes=read(EXP/'label-review/notes-pass1.json');by_source={n['source_id']:n for n in notes}
    sheets={n['source_id']:n for n in read(EXP/'label-review/manifest.json')}
    revisions={n['case_id']:n for n in read(EXP/'label-review/revisions/manifest.json')}
    second={n['case_id']:n for n in read(EXP/'label-review/revisions/notes.json')}
    assert set(second)==set(revisions) and all(n['decision']=='approved' for n in second.values())
    accepted=[];observations=[]
    for candidate in read(DATA/'train-cases.json'):
        c=criteria(candidate);note=by_source[c['source_id']];s=sheets[c['source_id']]
        assert sha(ROOT/s['sheet'])==s['sha256']
        assert sha(ROOT/c['image'])==c['input_sha256'] and sha(ROOT/c['target'])==c['target_sha256']
        evidence=note['evidence'];revision=None
        if c['id'] in note['rejected_cases']:
            revision=revisions[c['id']];n=second[c['id']]
            assert sha(ROOT/revision['sheet'])==revision['sheet_sha256']
            assert sha(ROOT/revision['target'])==revision['target_sha256']
            c.update(parameters=revision['parameters'],target=revision['target'],target_sha256=revision['target_sha256'])
            evidence=n['evidence']
        result=edit(load_rgb(ROOT/c['image']),validate(c['parameters']))
        assert np.array_equal(result,load_rgb(ROOT/c['target']))
        if c['action']=='identity':assert np.array_equal(result,load_rgb(ROOT/c['image']))
        c['status']='direct_AI_visual_approved_not_human_reviewed'
        accepted.append(c)
        observations.append({'case_id':c['id'],'input_sha256':c['input_sha256'],'target_sha256':c['target_sha256'],
          'sheet_sha256':revision['sheet_sha256'] if revision else s['sha256'],'evidence':evidence,
          'reviewer':'Codex direct visual AI-assisted','human_review':False,'decision':'approved',
          'review_scope':note['scope'],'candidate_manifest_sha256':sha(DATA/'train-cases.json')})
    assert len(accepted)==360 and collections.Counter(c['domain'] for c in accepted)=={'portrait':180,'landscape':180}
    write(DATA/'train-final-cases.json',accepted);write(EXP/'label-review/final-observations.json',observations)
    rows=[{'id':c['id']+'-'+lang,'case_id':c['id'],'language':lang,'parameters':c['parameters']} for c in accepted for lang in ('zh','en')]
    write(DATA/'train-accepted.json',rows)
    # Old-data control has exactly 720 rows, 440 unique language rows, sampled deterministically.
    old=read(ROOT/'data/training/functional-v1-20261104/train-accepted.json')
    rng=random.Random(20261106);extra=rng.sample(old,280);control=[copy.deepcopy(r) for r in old+extra]
    for idx,r in enumerate(control):r['source_row_id']=r['id'];r['id']=f'control-{idx:04d}'
    write(DATA/'control-accepted.json',control)
    amendments=[]
    for split in ('development','acceptance'):
        original=DATA/f'{split}-cases.json';final=DATA/f'{split}-final-cases.json'
        write(final,[criteria(c) for c in read(original)])
        amendments.append({'split':split,'draft_sha256':sha(original),'frozen_final_sha256':sha(final)})
    protocol=read(EXP/'protocol.json');protocol.update(status='labels_reviewed_and_frozen_before_training',freeze_at=stamp(),
        final_case_hashes={s:sha(DATA/f'{s}-final-cases.json') for s in ('train','development','acceptance')},
        instruction_criteria_amendments=amendments,
        amendment_reason='Before training or model inference: grammatical ZH prefixes; requested stylisation allowed, avoid contradictory preserve-natural-colour criteria. Original drafts retained.',
        train_dataset_sha256=sha(DATA/'train-accepted.json'),control_dataset_sha256=sha(DATA/'control-accepted.json'),
        system_prompt_sha256=__import__('hashlib').sha256(system_prompt().encode()).hexdigest())
    write(EXP/'protocol.json',protocol)
    write(DATA/'label-acceptance.json',{'status':'accepted_for_round2_pilot_training','at':stamp(),'semantic_cases':360,'photo_groups':60,
       'bilingual_rows':720,'domains':{'portrait':30,'landscape':30},'initial_rejected_and_revised':len(revisions),
       'dataset_sha256':sha(DATA/'train-accepted.json'),'case_sha256':sha(DATA/'train-final-cases.json'),
       'control_dataset_sha256':sha(DATA/'control-accepted.json'),'control_unique_language_rows':440,
       'instruction_origin':'100% AI-authored controlled requests; 0 untouched natural user requests; original GIER photo provenance retained.',
       'visual_review':'Codex actually viewed all training inputs/targets; AI-assisted, not human validation; reduced sheets validate global tone only.',
       'replay':'All 360 targets re-executed and pixel verified; all 60 identity targets pixel-identical.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['revise','finalize']);args=ap.parse_args()
    (revise if args.command=='revise' else finalize)()
