"""Training-only candidate replay under fixed bounded curves; never automatic relabeling."""
import hashlib,json,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,edit as old_edit,save
from functional_v1.curate import font
from .editor import edit
from .diagnose_editor import EXP

def main():
    data=ROOT/'data/training/functional-v6-20261105';out=EXP/'curve-label-candidates';assert not (out/'protocol.json').exists()
    cases=read(data/'cases.json');assert len(cases)==460 and all(c['split']=='train' for c in cases)
    assert read(EXP/'editor-diagnostic/decision.json')['live_editor_unchanged']
    write(out/'protocol.json',dict(at=stamp(),status='frozen_before_training_candidate_replay',source_cases_sha256=sha(data/'cases.json'),
        source_training_sha256=sha(data/'train-accepted.json'),source_admission_sha256=sha(data/'admission.json'),
        renderer='bounded-fixed',renderer_sha256=sha(Path(__file__).with_name('editor.py')),code_sha256=sha(__file__),
        scope='Only460 previously admitted round6 training tasks /160original groups80portrait80landscape. No new requests, translations, group changes or heldout samples.',
        purpose='Inspect whether exactly the old parameters remain acceptable under changed tone semantics; these are CANDIDATES, not new admitted targets.',
        admission='Byte/pixel-identical renders may reuse prior direct observation for the same request. Every changed render requires new direct visual review; reject or revise unclear/failed cases. No train-accepted data is created by this script.',
        visual_protocol='Original input and two candidate labels at identical full-frame scale; old/new editor identity and parameters hidden in panels. Full-size follow-up required for cloth, face and high-dynamic-range detail.',
        GPU_hours=0,training_started=False,live_editor_changed=False,acceptance_access=False))
    rows=[];pending=[]
    for c in cases:
        source=load_rgb(ROOT/c['image']);old=load_rgb(ROOT/c['target'])
        assert sha(ROOT/c['image'])==c['input_sha256'] and np.array_equal(old,old_edit(source,c['parameters']))
        new=edit(source,c['parameters'],'bounded-fixed');identical=bool(np.array_equal(old,new))
        path=out/'images'/f'{c["id"]}.png';save(path,new)
        row=dict(case_id=c['id'],group_id=c['group_id'],domain=c['domain'],family=c['family'],instructions=c['instructions'],
            input_image=c['image'],input_sha256=c['input_sha256'],old_target=c['target'],old_target_sha256=sha(ROOT/c['target']),
            parameters=c['parameters'],new_candidate=str(path.relative_to(ROOT)),new_candidate_sha256=sha(path),
            pixel_identical_to_previously_reviewed_label=identical,status='exact_prior_render_review_available' if identical else 'pending_new_direct_visual_review',
            admitted_for_new_training=False)
        rows.append(row)
        if not identical:pending.append(row)
    sheets=[]
    for offset in range(0,len(pending),3):
        subset=pending[offset:offset+3];canvas=Image.new('RGB',(1800,660*len(subset)),'white');d=ImageDraw.Draw(canvas);entries=[]
        for i,r in enumerate(subset):
            y=i*660;d.text((12,y+6),r['case_id']+' | '+r['instructions']['en'][:125],fill='black',font=font(22))
            variants=sorted([('old',r['old_target']),('new',r['new_candidate'])],key=lambda t:hashlib.sha256((r['case_id']+t[0]).encode()).hexdigest())
            sources=[('INPUT',r['input_image'])]+[(hashlib.sha256((r['case_id']+role).encode()).hexdigest()[:12],p) for role,p in variants]
            entries.append(dict(case_id=r['case_id'],variants=[dict(role=role,label=hashlib.sha256((r['case_id']+role).encode()).hexdigest()[:12]) for role,_ in variants]))
            for j,(label,path) in enumerate(sources):
                x=j*600;d.text((x+12,y+42),label,fill='black',font=font(21))
                with Image.open(ROOT/path) as im:fitted=ImageOps.contain(im.convert('RGB'),(580,570),Image.Resampling.LANCZOS)
                canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+80+(570-fitted.height)//2))
        path=out/f'sheet-{offset//3+1:03d}.jpg';canvas.save(path,quality=95,subsampling=0);sheets.append(dict(sheet=str(path.relative_to(ROOT)),sha256=sha(path),entries=entries))
    write(out/'manifest.json',dict(at=stamp(),protocol_sha256=sha(out/'protocol.json'),cases=rows,sheets=sheets,
        semantic_cases=460,original_groups=160,domains={'portrait':80,'landscape':80},
        exact_prior_render_cases=sum(r['pixel_identical_to_previously_reviewed_label'] for r in rows),
        changed_cases_requiring_new_direct_review=len(pending),sheet_count=len(sheets),new_dataset_admitted=False,new_training_started=False))
    print(json.dumps(dict(cases=460,exact_prior_renders=len(rows)-len(pending),pending_changed_cases=len(pending),review_sheets=len(sheets),training_started=False)))

if __name__=='__main__':main()
