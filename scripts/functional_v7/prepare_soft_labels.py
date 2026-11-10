"""Full training-only replay after the twelve-case soft contrast capacity probe.

Creates candidates and review panels, never an admitted training dataset.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb, edit as legacy_edit, save
from functional_v1.curate import font
from .editor_soft import edit
from .outcomes import validate_decision
from .diagnose_editor import EXP


def main():
    data = ROOT/'data/training/functional-v6-20261105'
    out = EXP/'soft-label-candidates'
    assert not (out/'protocol.json').exists(), 'Do not overwrite frozen candidate replays'
    cases = read(data/'cases.json')
    assert len(cases) == 460 and all(c['split'] == 'train' for c in cases)
    decision = EXP/'soft-contrast-capacity/decision.json'
    assert read(decision)['decision'] == 'advance_to_full_training_label_audit_only'
    write(out/'protocol.json', dict(
        at=stamp(), status='frozen_before_soft_training_candidate_replay',
        source_cases_sha256=sha(data/'cases.json'), source_training_sha256=sha(data/'train-accepted.json'),
        source_admission_sha256=sha(data/'admission.json'), capacity_decision_sha256=sha(decision),
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                     (Path(__file__),Path(__file__).with_name('editor_soft.py'),Path(__file__).with_name('outcomes.py'))},
        renderer='independent soft RGB contrast + bounded value exposure/brightness, fixed mode',
        scope='460 previously admitted round6 training semantic tasks,160original groups80portrait80landscape; all original inputs/requests/parameters unchanged. No development or acceptance images.',
        purpose='Full target audit under changed tool semantics, not model training or automatic label approval.',
        admission='Exact prior pixels may reuse prior direct observation for the same request. Changed outputs need new direct visual inspection; failures and uncertainties must be repaired or excluded. No train-accepted.json is written.',
        visual_protocol='Three cases per panel, each original input plus old/new target; per-case role labels hidden. Preserve both observations if exact prior baseline pixels receive conflicting re-reviews. Full-size follow-up for material detail uncertainty.',
        new_labels_admitted=0, new_training_started=False, live_tools_changed=False, heldout_access=False, GPU_hours=0))
    rows=[]
    pending=[]
    for c in cases:
        source=load_rgb(ROOT/c['image'])
        old=load_rgb(ROOT/c['target'])
        assert sha(ROOT/c['image']) == c['input_sha256']
        assert sha(ROOT/c['target']) == c['target_sha256']
        assert np.array_equal(old,legacy_edit(source,c['parameters']))
        result=edit(source,c['parameters'],'bounded-fixed')
        identical=bool(np.array_equal(old,result))
        path=out/'images'/f'{c["id"]}.png'
        save(path,result)
        target = validate_decision({'action':'edit','parameters':c['parameters']} if any(c['parameters'].values()) else {'action':'keep'})
        row=dict(case_id=c['id'],group_id=c['group_id'],domain=c['domain'],family=c['family'],instructions=c['instructions'],
                 input_image=c['image'],input_sha256=c['input_sha256'],old_target=c['target'],old_target_sha256=sha(ROOT/c['target']),
                 parameters=c['parameters'],candidate_decision=target,new_candidate=str(path.relative_to(ROOT)),new_candidate_sha256=sha(path),
                 pixel_identical_to_previously_reviewed_label=identical,
                 status='exact_prior_render_review_available' if identical else 'pending_new_direct_visual_review',admitted_for_new_training=False)
        rows.append(row)
        if not identical:
            pending.append(row)
    sheets=[]
    for offset in range(0,len(pending),3):
        subset=pending[offset:offset+3]
        canvas=Image.new('RGB',(1800,660*len(subset)),'white')
        draw=ImageDraw.Draw(canvas)
        entries=[]
        for i,r in enumerate(subset):
            y=i*660
            draw.text((12,y+6),r['case_id']+' | '+r['instructions']['en'][:125],fill='black',font=font(22))
            variants=sorted([('old',r['old_target']),('new',r['new_candidate'])],
                            key=lambda v:hashlib.sha256(('soft-v1'+r['case_id']+v[0]).encode()).hexdigest())
            label=lambda role:hashlib.sha256(('soft-v1'+r['case_id']+role).encode()).hexdigest()[:12]
            entries.append(dict(case_id=r['case_id'],variants=[dict(role=role,label=label(role)) for role,_ in variants]))
            for j,(visible,path) in enumerate([('INPUT',r['input_image'])]+[(label(role),path) for role,path in variants]):
                x=j*600
                draw.text((x+12,y+42),visible,fill='black',font=font(21))
                with Image.open(ROOT/path) as im:
                    fitted=ImageOps.contain(im.convert('RGB'),(580,570),Image.Resampling.LANCZOS)
                canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+80+(570-fitted.height)//2))
        path=out/f'sheet-{offset//3+1:03d}.jpg'
        canvas.save(path,quality=95,subsampling=0)
        sheets.append(dict(sheet=str(path.relative_to(ROOT)),sha256=sha(path),entries=entries))
    write(out/'manifest.json',dict(at=stamp(),protocol_sha256=sha(out/'protocol.json'),cases=rows,sheets=sheets,
                                 semantic_cases=460,original_groups=160,domains={'portrait':80,'landscape':80},
                                 exact_prior_render_cases=len(rows)-len(pending),changed_cases_requiring_new_direct_review=len(pending),
                                 sheet_count=len(sheets),new_dataset_admitted=False,new_training_started=False))
    print(dict(cases=460,exact_prior_renders=len(rows)-len(pending),changed_cases=len(pending),sheets=len(sheets),training_started=False))


if __name__ == '__main__':
    main()
