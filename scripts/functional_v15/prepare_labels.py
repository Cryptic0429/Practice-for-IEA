"""Replay all R13 labels; new shadow results are candidates, never auto-admitted."""
import os,copy,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from functional_v1.curate import font
from .outcomes import execute,EDITOR_VERSION,SCHEMA_VERSION

EXP=ROOT/'experiments/functional-v15-20261106'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    target=EXP/'label-candidates/manifest.json';assert not target.exists()
    source=ROOT/'data/training/functional-v13-20261106';cases=read(source/'cases.json')
    assert len(cases)==925;changed=[];unchanged=[];declines=[]
    for case in cases:
        rgb=load_rgb(ROOT/case['image']);assert sha(ROOT/case['image'])==case['input_sha256']
        result=execute(rgb,case['decision']);p=case.get('parameters') or {}
        if case['action']=='decline':
            assert result['result'] is None and case['target'] is None
            declines.append(dict(id=case['id'],reason=case['decision']['reason'],previous_label_evidence=case['label_evidence'],
                pending_new_tool_feasibility_review=case['decision']['reason']=='unreliable_quality'))
        if case['action']=='edit' and p.get('shadows',0):
            assert sha(ROOT/case['target'])==case['target_sha256']
            previous=load_rgb(ROOT/case['target']);exact=np.array_equal(result['result'],previous)
            out=target.parent/'images'/f'{case["id"]}.png'
            if out.exists():assert np.array_equal(load_rgb(out),result['result']), 'Preserve different previous outputs'
            else:save(out,result['result'])
            canvas=Image.new('RGB',(1800,820),'white');draw=ImageDraw.Draw(canvas)
            draw.text((16,12),case['id']+' | TRAINING LABEL CANDIDATE, NOT MODEL OUTPUT',font=font(23),fill='black')
            for n,line in enumerate(textwrap.wrap(case['instructions']['en'],140)):
                draw.text((16,48+25*n),line,font=font(20),fill='black')
            for n,(label,path) in enumerate([('INPUT',ROOT/case['image']),('PREVIOUS USED LABEL',ROOT/case['target']),('SHARED SHADOW GAIN CANDIDATE',out)]):
                draw.text((n*600+12,150),label,font=font(20),fill='black')
                with Image.open(path) as image:fitted=ImageOps.contain(image.convert('RGB'),(580,600),Image.Resampling.LANCZOS)
                canvas.paste(fitted,(n*600+10+(580-fitted.width)//2,190+(600-fitted.height)//2))
            panel=target.parent/'panels'/f'{case["id"]}.jpg';panel.parent.mkdir(parents=True,exist_ok=True)
            if not panel.exists():canvas.save(panel,quality=95,subsampling=0)
            changed.append(dict(id=case['id'],group_id=case['group_id'],domain=case['domain'],family=case['family'],
                instructions=case['instructions'],instruction_origin=case['instruction_origin'],expected_change=case.get('expected_change'),must_preserve=case.get('must_preserve'),
                review_requirements=dict(required_change=case.get('expected_change') or case['instructions']['zh'],
                    must_preserve=case.get('must_preserve') or 'Preserve subject, composition, unrelated content and all details/colours expressly protected by the unchanged original request; avoid added clipping, shadow loss, noise amplification or an unrequested cast.',
                    criteria_origin='Explicit prior fields where present; otherwise literal unchanged request plus the frozen basic-quality preservation policy. Derived review metadata, not an original dataset annotation or instruction rewrite.'),
                input=dict(path=case['image'],sha256=case['input_sha256'],size=list(rgb.shape[1::-1])),
                previous_target=dict(path=case['target'],sha256=case['target_sha256']),candidate=dict(path=str(out.relative_to(ROOT)),sha256=sha(out)),
                parameters=p,decision=case['decision'],editor=EDITOR_VERSION,schema=SCHEMA_VERSION,pixels_identical_to_previous=exact,
                panel=dict(path=str(panel.relative_to(ROOT)),sha256=sha(panel)),status='CPU_executed_pending_direct_AI_review_not_admitted',label_admitted=False))
        else:
            if case['action']!='decline':
                assert sha(ROOT/case['target'])==case['target_sha256']
                assert np.array_equal(result['result'],load_rgb(ROOT/case['target']))
            unchanged.append(dict(id=case['id'],action=case['action'],exact_pixels_or_no_output=True,
                quality_decline_requires_new_feasibility_review=case['action']=='decline' and case['decision']['reason']=='unreliable_quality'))
        write(EXP/'label-replay-progress.json',dict(at=stamp(),completed=len(changed)+len(unchanged),changed_candidates=len(changed),GPU_hours=0,label_admissions=0))
    assert len(changed)==132 and len(unchanged)==793
    assert sum(r['pending_new_tool_feasibility_review'] for r in declines)==44
    write(target,dict(at=stamp(),source_cases=str((source/'cases.json').relative_to(ROOT)),source_cases_sha256=sha(source/'cases.json'),
        source_admission_sha256=sha(source/'admission.json'),editor=EDITOR_VERSION,editor_sha256=sha(ROOT/'scripts/functional_v15/editor.py'),
        schema_sha256=sha(ROOT/'scripts/functional_v15/outcomes.py'),items=changed,other_replays=unchanged,decline_cases=declines,
        actual_changed_parameter_cases=132,other_replays_count=793,quality_decline_feasibility_reviews_required=44,
        label_admissions=0,training_started=False,GPU_hours=0,
        policy='Every changed result needs visible functional/quality review and native-pixel followup when uncertain. No lowest-L1 ranking or automatic admission. Exact prior pixels can inherit existing AI evidence; quality refusals still need revalidation under the new tool.'))
    print('925 actual replays:132 shadow candidates pending review,793 other replays,44 quality-decline feasibility checks still required; zero admissions.')


if __name__=='__main__':main()
