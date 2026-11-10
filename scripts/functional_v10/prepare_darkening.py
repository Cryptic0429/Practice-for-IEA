"""CPU strength coverage candidates from the same20 admitted training sources."""
from pathlib import Path
import re
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from .prepare_counterfactuals import EXP,render
from .extend_candidates import panel

REQUEST=dict(zh='请把整张照片大幅调暗，保留原有颜色和主体细节。',
             en='Darken the whole photograph substantially while retaining its original colours and subject detail.')


def main():
    destination=EXP/'darkening-candidates/manifest.json';assert not destination.exists()
    data=ROOT/'data/training/functional-v7-20261105/cases.json';rows=read(data)
    negative=[];explicit=[]
    for c in rows:
        p=c['decision'].get('parameters',{});text=c['instructions']['en'].lower()
        if p.get('exposure',0)<0 or p.get('brightness',0)<0:
            negative.append(dict(id=c['id'],instruction_en=c['instructions']['en'],parameters=p))
        if re.search(r'darken|darker',text) and re.search(r'a lot|much|large|significant|substantial|considerabl|strong',text):
            explicit.append(dict(id=c['id'],instruction_en=c['instructions']['en']))
    write(EXP/'training-gap-audit.json',dict(at=stamp(),cases_sha256=sha(data),semantic_cases=len(rows),
        negative_exposure_or_brightness_cases=negative,negative_parameter_count=len(negative),
        explicit_strong_darken_lexical_matches=explicit,explicit_match_count=len(explicit),
        evidence_policy='All11 negative global-parameter requests were directly read. Some request brightening but correctly combine negative exposure with lifted shadows; parameter signs alone do NOT judge functional direction. Exact English lexical coverage diagnosis only, not a formal paraphrase-complete semantic proof.',
        source_for_conditioned_refusal_gap='experiments/functional-v9-20261106/request-conditioning-audit.json'))
    source=EXP/'counterfactual-candidates/manifest.json';candidates=[]
    for c in read(source)['cases']:
        base=load_rgb(ROOT/c['images']['canonical']['path']);cid=c['id'];images={}
        for name,pixels in [('bright-deep-input',render(base,{'exposure':65})),('bright-shallow-input',render(base,{'exposure':30}))]:
            file=EXP/'darkening-candidates/images'/f'{cid}-{name}.png';save(file,pixels)
            images[name]=dict(path=str(file.relative_to(ROOT)),sha256=sha(file))
            amount=-65 if name=='bright-deep-input' else -40
            edited=render(pixels,{'exposure':amount});target=EXP/'darkening-candidates/images'/f'{cid}-{name}-edit.png';save(target,edited)
            images[name+'-edit']=dict(path=str(target.relative_to(ROOT)),sha256=sha(target))
        paths={name:ROOT/r['path'] for name,r in images.items()}
        sheet=panel(cid,'darkening',{**{'canonical':ROOT/c['images']['canonical']['path']},**paths})
        candidates.append(dict(id=cid,domain=c['domain'],group_id=c['group_id'],source_id=c['source_id'],
            instructions=REQUEST,images=images,panel=sheet,
            proposed_parameters={'bright-deep-input':{'exposure':-65},'bright-shallow-input':{'exposure':-40}},
            status='CPU_candidates_only_not_admitted',
            note='Both are synthetic brightened inputs on an existing training photograph. The canonical source and expected parameters are NOT model inputs; actual visible strength and quality must be reviewed.'))
    write(destination,dict(at=stamp(),status='CPU_darkening_candidates_only_no_admission_or_training',
        source_manifest=str(source.relative_to(ROOT)),source_manifest_sha256=sha(source),original_groups=20,
        domains=dict(portrait=10,landscape=10),semantic_candidates=40,requests_AI_authored=True,cases=candidates,
        purpose='Add explicit strong negative lighting requests to the same-image-condition supervision; no L1 selects labels.'))
    print('CPU darkening40 semantic candidates, same20 training groups; all unadmitted.')

if __name__=='__main__':main()
