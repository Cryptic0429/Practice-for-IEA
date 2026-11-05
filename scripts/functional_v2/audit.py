"""Data/lineage/input-boundary checks; image-quality decisions remain direct visual."""
from collections import Counter,defaultdict
import copy,hashlib,json
from functional_v1.common import input_row,validate
from .common import ROOT,DATA,EXP,read,write,sha,stamp

def main():
    acceptance=read(DATA/'label-acceptance.json');protocol=read(EXP/'protocol.json')
    cases=read(DATA/'train-final-cases.json');rows=read(DATA/'train-accepted.json')
    assert len(cases)==360 and len(rows)==720 and len({r['id'] for r in rows})==720
    assert sha(DATA/'train-final-cases.json')==acceptance['case_sha256']
    assert sha(DATA/'train-accepted.json')==acceptance['dataset_sha256']
    byid={c['id']:c for c in cases};groups=defaultdict(list)
    for c in cases:
        groups[c['source_id']].append(c);validate(c['parameters'])
        assert sha(ROOT/c['image'])==c['input_sha256'] and sha(ROOT/c['target'])==c['target_sha256']
        baseline=input_row(c,'zh');poison=copy.deepcopy(c)
        poison.update(parameters={'sepia':100},reference='secret_reference.png',review='secret_label',canonical_photo='secret_source.png')
        assert input_row(poison,'zh')==baseline
        assert 'secret_' not in json.dumps(baseline)
    assert len(groups)==60 and all(len(v)==6 for v in groups.values())
    assert Counter(c['domain'] for c in cases)=={'portrait':180,'landscape':180}
    restored=styles=0
    for v in groups.values():
        restore=[c for c in v if c['action'].startswith('restore-')]
        assert len(restore)==2 and restore[0]['instructions']==restore[1]['instructions']
        assert restore[0]['input_sha256']!=restore[1]['input_sha256'] and restore[0]['parameters']!=restore[1]['parameters']
        restored+=1
        style=[c for c in v if c['action'].startswith('style-')]
        assert len(style)==2 and style[0]['input_sha256']==style[1]['input_sha256']
        assert style[0]['instructions']!=style[1]['instructions'] and style[0]['parameters']!=style[1]['parameters']
        styles+=1
        identity=next(c for c in v if c['action']=='identity')
        assert identity['parameters']=={} and identity['target_sha256']==identity['input_sha256']
    assert Counter(r['case_id'] for r in rows)=={c['id']:2 for c in cases}
    assert all(r['language'] in ('zh','en') and r['parameters']==byid[r['case_id']]['parameters'] for r in rows)
    observations=read(EXP/'label-review/final-observations.json')
    assert {r['case_id'] for r in observations}==set(byid)
    assert all(r['decision']=='approved' and r['target_sha256']==byid[r['case_id']]['target_sha256'] for r in observations)
    control=read(DATA/'control-accepted.json');old={r['id']:r for r in read(ROOT/'data/training/functional-v1-20261104/train-accepted.json')}
    assert len(control)==720 and len({r['id'] for r in control})==720 and len({r['source_row_id'] for r in control})==440
    assert all(r['parameters']==old[r['source_row_id']]['parameters'] and r['case_id']==old[r['source_row_id']]['case_id'] for r in control)
    splitgroups={s:{c['group_id'] for c in read(DATA/f'{s}-final-cases.json')} for s in ('train','development','acceptance')}
    assert not(splitgroups['train']&splitgroups['development'] or splitgroups['train']&splitgroups['acceptance'] or splitgroups['development']&splitgroups['acceptance'])
    for file,digest in protocol['core_code_hashes'].items():assert sha(ROOT/file)==digest
    write(EXP/'planning/data-boundary-audit.json',{'status':'passed','at':stamp(),'semantic_cases':360,'photo_groups':60,'bilingual_rows':720,
       'same_request_different_problem_pairs':restored,'same_photo_different_strength_pairs':styles,'exact_identity_cases':60,
       'reviewed_targets':len(observations),'control_rows':720,'control_distinct_language_rows':440,
       'unique_request_wordings':{lang:len({c['instructions'][lang] for c in cases}) for lang in ('zh','en')},
       'data_sha256':sha(DATA/'train-accepted.json'),'control_sha256':sha(DATA/'control-accepted.json'),
       'limits':'Boundary checks, not proof of photo quality or generalisation; all instruction templates are AI-authored.'})
    print('Round2 data boundary audit passed')
if __name__=='__main__':main()
