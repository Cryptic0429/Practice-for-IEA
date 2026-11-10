"""Resolve A/B to exact images and summarize swap and bilingual stability."""
from collections import Counter

from functional_v1.common import ROOT,read,sha,stamp,write
from .boards import EXP,PROTOCOL


def main():
    protocol,boards,reference = read(PROTOCOL),read(EXP/'boards.json'),read(EXP/'reference.json')
    score_path = EXP/'scores.json'
    scores = read(score_path)
    bindings = {'protocol_sha256':PROTOCOL,'boards_sha256':EXP/'boards.json',
                'reference_sha256':EXP/'reference.json'}
    if any(scores[k]!=sha(path) for k,path in bindings.items()):
        raise ValueError('Reference or visual protocol changed')
    manifest = {c['id']:c for c in boards['cases']}
    notes = {r['id']:r for r in reference['cases']}
    decisions = {r['id']:[] for r in protocol['cases']}
    for r in scores['rows']:
        case = manifest[r['id']]
        board = case['boards'][r['order']]
        note = notes[r['id']]
        if r['board']!=board['path'] or r['board_sha256']!=sha(ROOT/board['path']):
            raise ValueError('Qwen evidence image changed')
        if note['viewed_board_sha256']!=case['boards']['12']['sha256']:
            raise ValueError('Direct review evidence changed')
        selected = max(r['logits'],key=r['logits'].get)
        if selected!=r['selected']:
            raise ValueError('Logit decision mismatch')
        canonical = 'C' if selected=='C' else str(board['order']['AB'.index(selected)])
        decisions[r['id']].append(dict(order=r['order'],language=r['language'],label=selected,
            canonical=canonical,correct=canonical==note['expected'],damage=canonical in note['damaged']))
    items = []
    for case in protocol['cases']:
        cid = case['id']
        rows = decisions[cid]
        note = notes[cid]
        if {(r['order'],r['language']) for r in rows}!={(o,l) for o in ('12','21') for l in ('zh','en')} or len(rows)!=4:
            raise ValueError(f'Missing or repeated evaluation view: {cid}')
        votes = {r['canonical'] for r in rows}
        per_language = {}
        for lang in ('zh','en'):
            choices = {r['canonical'] for r in rows if r['language']==lang}
            per_language[lang] = next(iter(choices)) if len(choices)==1 else None
        items.append(dict(id=cid,group_id=case['group_id'],kind=case['kind'],expected=note['expected'],
            correct_views=sum(r['correct'] for r in rows),damaged_views=sum(r['damage'] for r in rows),
            by_language_order_consensus=per_language,all_views_consensus=next(iter(votes)) if len(votes)==1 else None,
            views=rows))
    types = {}
    for kind in sorted({c['kind'] for c in items}):
        subset = [c for c in items if c['kind']==kind]
        types[kind] = dict(semantic_cases=len(subset),correct_views=sum(c['correct_views'] for c in subset),
                           all_views_correct=sum(c['correct_views']==4 for c in subset))
    rows = [r for c in items for r in c['views']]
    counts = Counter(c['all_views_consensus'] for c in items)
    result = dict(at=stamp(),scores_sha256=sha(score_path),by_kind=types,
        summary=dict(original_groups=len({c['group_id'] for c in items}),semantic_cases=len(items),
            literal_order_views=len(rows),correct_views=sum(r['correct'] for r in rows),
            all_views_correct=sum(c['correct_views']==4 for c in items),
            all_views_consistent=len(items)-counts[None],unstable_semantic_cases=counts[None],
            order_disagreements=sum(v is None for c in items for v in c['by_language_order_consensus'].values()),
            language_disagreements_fixed_order=sum(
                len({r['canonical'] for r in c['views'] if r['order']==o})>1 for c in items for o in ('12','21')),
            correct_neither_views=sum(c['correct_views'] for c in items if c['expected']=='C'),
            incorrect_safe_refusal_views=sum(r['canonical']=='C' for c in items if c['expected']!='C' for r in c['views']),
            confirmed_damaged_choices=sum(r['damage'] for r in rows),
            confirmed_damaged_semantic_cases=sum(c['damaged_views']>0 for c in items),
            unsafe_consensus_cases=sum(c['all_views_consensus'] in notes[c['id']]['damaged'] for c in items),
            consensus_successful_edit_cases=sum(c['all_views_consensus']==c['expected']!='C' for c in items),
            consensus_correct_neither_cases=sum(c['all_views_consensus']==c['expected']=='C' for c in items)),
        cases=items,GPU_hours=0,deployed=False,final_acceptance_opened=False)
    write(EXP/'result.json',result)
    print(result['summary'],flush=True)


if __name__ == '__main__':
    main()
