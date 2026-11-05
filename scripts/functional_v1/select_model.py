"""Development-only selection with complete direct review and explicit tie handling."""
from pathlib import Path
import argparse
import json
from .common import ROOT, DATA, EXP, EVAL, PARENT, read, write, sha, stamp, EDITOR_VERSION


def predictions(mid):
    return {(r['case_id'],r['language']):r for r in map(json.loads,(EVAL/'development'/mid/'predictions.jsonl').read_text().splitlines())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',action='store_true')
    args=ap.parse_args()
    if (EXP/'final-model-lock.json').exists():
        raise RuntimeError('Final acceptance already locked; do not reselect from final outcomes')
    ids=['base','current']+[f'{s}-e{e}' for s in ('base','current') for e in range(1,5)]
    summaries={mid:read(EVAL/'development'/mid/'direct-functional-summary.json') for mid in ids}
    assert all(s['status']=='completed' for s in summaries.values()), 'Every candidate needs complete direct functional review'
    assert all(s['cases_sha256']==sha(DATA/'development-cases.json') and not s['selection_uses_L1'] for s in summaries.values())
    pair_file=EXP/'development-pair-preferences.json'
    pair=read(pair_file) if pair_file.exists() else []
    decisions=[]
    def rank(mid):
        s=summaries[mid]
        return s['fully_completed_both_languages'], -s['major_quality_issues']
    def prefer(a,b):
        pa,pb=predictions(a),predictions(b)
        if all(pa[k].get('output_sha256')==pb[k].get('output_sha256') and pa[k]['status']==pb[k]['status'] for k in pa):
            decisions.append({'a':a,'b':b,'preference':'tie','reason':'Every case-language output is exactly identical.'})
            return 0
        r=next((r for r in pair if {r['a'],r['b']}=={a,b}),None)
        if r is None:
            raise RuntimeError(f'Functional count tied; direct anonymous paired preference required: {a} / {b}')
        assert r['reviewer']=='Codex direct visual review; AI-assisted, not human'
        assert r['development_only'] and r['uncertain']==0
        score=r['wins_a']-r['wins_b']
        decisions.append(r)
        return (1 if score>0 else -1 if score<0 else 0)*(1 if r['a']==a else -1)
    chosen={}
    for start in ('base','current'):
        options=[f'{start}-e{e}' for e in range(1,5)]
        tied=[m for m in options if rank(m)==max(map(rank,options))]
        winner=tied[0]
        for other in tied[1:]:
            if prefer(other,winner)>0: winner=other
        chosen[start]=winner
    a,b=chosen['base'],chosen['current']
    if rank(a)>rank(b):winner=a
    elif rank(b)>rank(a):winner=b
    else:winner=b if prefer(b,a)>0 else a
    baseline=summaries['current']; candidate=summaries[winner]
    no_new_major=set(candidate['major_issue_cases']).issubset(baseline['major_issue_cases'])
    improved=candidate['fully_completed_both_languages']>baseline['fully_completed_both_languages'] and no_new_major
    registry={}
    for start,mid in chosen.items():
        identity=read(EVAL/'development'/mid/'identity.json')
        model_id=f'functional-{start}-20261104'
        registry[model_id]={'evaluation_id':mid,'adapter':str(Path(identity['adapter']).relative_to(ROOT)),
            'adapter_sha256':identity['adapter_sha256'],'epoch':int(mid[-1]),'functional_summary':summaries[mid]}
    selected_id=f'functional-{winner.split("-")[0]}-20261104'
    selection={'status':'development_selected','at':stamp(),'selected_model_id':selected_id,
        'models':registry,'baseline':baseline,'selected_summary':candidate,'tie_decisions':decisions,
        'functional_improvement_over_current':improved,'no_new_major_issue_cases':no_new_major,
        'replication_authorized_by_protocol':improved,'selection_uses_L1':False,
        'dataset_sha256':sha(DATA/'train-accepted.json'),'development_cases_sha256':sha(DATA/'development-cases.json'),
        'human_review':False}
    write(EXP/'development-selection.json',selection)
    if args.lock:
        replication=None
        if improved:
            replication=read(EVAL/'development/replication-selected/direct-functional-summary.json')
            identity=read(EVAL/'development/replication-selected/identity.json')
            start=winner.split('-')[0]
            epoch=int(winner[-1])
            expected=EXP/'runs'/f'{start}-seed-20261105'/f'checkpoint-{epoch*110}'
            state=read(expected.parent/'status.json')
            settings=read(expected.parent/'settings.json')
            assert state['status']=='completed' and state['weights_changed']
            assert settings['seed']==20261105 and settings['dataset_sha256']==selection['dataset_sha256']
            assert identity['adapter']==str(expected) and identity['adapter_sha256']==sha(expected/'adapter_model.safetensors')
            assert replication['status']=='completed' and replication['cases_sha256']==selection['development_cases_sha256']
            rid=f'functional-{start}-20261105'
            registry[rid]={'evaluation_id':'replication-selected','adapter':str(expected.relative_to(ROOT)),
                'adapter_sha256':identity['adapter_sha256'],'epoch':epoch,'functional_summary':replication,
                'selection':'Same epoch as the first seed; no checkpoint reselection.'}
            selection['models']=registry
            selection['replication_summary']=replication
            selection['replication_improves_current']=replication['fully_completed_both_languages']>baseline['fully_completed_both_languages'] and set(replication['major_issue_cases']).issubset(baseline['major_issue_cases'])
            write(EXP/'development-selection.json',selection)
        lock={'locked_at':stamp(),'editor':EDITOR_VERSION,'acceptance_cases_sha256':sha(DATA/'acceptance-cases.json'),
            'development_selection_sha256':sha(EXP/'development-selection.json'),
            'code_hashes':{f'scripts/functional_v1/{f}':sha(Path(__file__).with_name(f)) for f in ('common.py','runtime.py','editor.py')},
            'models':{mid:{'adapter_sha256':entry['adapter_sha256'],'adapter':entry['adapter']} for mid,entry in registry.items()}}
        lock['models']['current']={'adapter':str(PARENT.relative_to(ROOT)), 'adapter_sha256':sha(PARENT/'adapter_model.safetensors')}
        lock['models']['base']={'adapter':None,'adapter_sha256':None}
        write(EXP/'final-model-lock.json',lock)
    print(json.dumps(selection,ensure_ascii=False))


if __name__=='__main__':main()
