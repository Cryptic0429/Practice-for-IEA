"""Summarize actual direct observations and preserve disagreements with exact old renders."""
import json
from collections import defaultdict
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v6.evaluate import EVAL
from .diagnose_editor import EXP,MODELS,VARIANTS,IDS

def main():
    dest=EXP/'editor-diagnostic/decision.json';assert not dest.exists()
    manifest=read(EXP/'editor-diagnostic/manifest.json')
    notes={r['item_id']:r for r in read(EXP/'editor-diagnostic/codex-observations.json')}
    assert len(notes)==len(manifest['items'])==37
    for r in notes.values():assert sha(ROOT/r['sheet'])==r['sheet_sha256']
    replay=read(EXP/'editor-diagnostic/replays.json')
    # Historical review is evidence of uncertainty, never automatic approval.
    prior={}
    for split in IDS:
        for mid in MODELS:
            scores={(r['case_id'],r['language']):r['assessment'] for r in read(EVAL/split/mid/'direct-review.json')}
            for p in map(json.loads,(EVAL/split/mid/'predictions.jsonl').read_text().splitlines()):
                if p['status']=='executed':prior[p['case_id'],p['output_sha256']]=scores[p['case_id'],p['language']]
    conflicts=[];assessments={}
    for item in manifest['items']:
        n=notes[item['item_id']];assessment=dict(n);old=prior.get((item['case_id'],item['image_sha256']))
        if old:
            fields=[k for k in ('completion','quality') if old[k]!=n[k]]
            if fields:
                conflicts.append(dict(item_id=item['item_id'],disputed_fields=fields,old=old,new=n))
                for k in fields:assessment[k]='uncertain'
        assessments[item['item_id']]=assessment
    write(EXP/'editor-diagnostic/consistency-adjudication.json',dict(at=stamp(),conflicts=conflicts,
        policy='Disputed fields on exactly matching old pixels remain uncertain; original old and new notes are immutable.'))
    summaries={}
    for mid in MODELS:
        for variant in ('legacy',)+VARIANTS:
            grouped=defaultdict(list)
            for r in replay:
                if r['model_id']==mid and r.get('variant')==variant:grouped[r['case_id']].append(assessments[r['item_id']])
            assert len(grouped)==8 and all(len(v)==2 for v in grouped.values())
            summaries[mid+'/'+variant]=dict(qualified_cases=sum(all(r['completion']=='completed' and r['quality']=='acceptable' for r in v) for v in grouped.values()),
                major_cases=sum(any(r['quality']=='major_issue' for r in v) for v in grouped.values()),
                uncertain_cases=sum(any('uncertain' in (r['completion'],r['quality']) for r in v) for v in grouped.values()),
                cases={cid:dict(assessments=v,qualified=all(r['completion']=='completed' and r['quality']=='acceptable' for r in v)) for cid,v in grouped.items()})
    write(dest,dict(at=stamp(),status='targeted_CPU_diagnostic_completed_no_promotion',models=MODELS,variants=VARIANTS,
        summaries=summaries,unique_images_directly_viewed=37,language_renderer_records=96,independent_user_review=False,
        training_started=False,GPU_hours=0,acceptance_closed=True,live_editor_unchanged=True,
        observations_sha256=sha(EXP/'editor-diagnostic/codex-observations.json'),protocol_sha256=sha(EXP/'editor-diagnostic-protocol.json'),
        conclusion='Smooth bounded curves can reduce clipping at the same model parameters, but do not fix noise, missing brightness or wrong strength. No practical readiness claim; broader frozen-development replay is required before adopting any editor.',
        limitation='A targeted already-seen diagnostic. Equal slider values across editors need not have equal perceptual strength; this is an execution effect, not an additional learned-model gain.'))
    print(json.dumps({k:{f:v[f] for f in ('qualified_cases','major_cases','uncertain_cases')} for k,v in summaries.items()}))

if __name__=='__main__':main()
