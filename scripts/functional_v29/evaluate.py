"""Compare frozen, supervised heads and trained final-block LoRA on locked held inputs."""
from collections import Counter

import torch
from torch.nn.functional import linear

from functional_v1.common import ROOT,read,sha,stamp,write
from .cache import EXP,PLAN,TRAIN,rows_and_plan
from .train import CACHE,RUN,Tail,load_cache

METHODS = ('frozen','legacy_head','augmented_head','tail_lora')


def check(path,digest):
    if sha(path)!=digest:
        raise ValueError(f'Bound input changed: {path}')


def images():
    path = EXP/'visual-review.json'
    review = read(path)
    check(PLAN,review['protocol_sha256'])
    for key,name in [('replay_sha256','replay'),('isolation_sha256','isolation'),('details_sha256','details')]:
        check(EXP/f'{name}.json',review[key])
    isolation = read(EXP/'isolation.json')
    if isolation['flags']:
        raise ValueError('Held photo source isolation failed')
    r = {c['id']:c for c in read(EXP/'replay.json')['cases'] if c['cohort']=='held_photos'}
    notes = {c['id']:c for c in review['cases']}
    for cid,c in r.items():
        if c['source_sha256']!=isolation['photos'][cid]:
            raise ValueError('Held source did not match isolated source')
        check(ROOT/c['source'],c['source_sha256'])
        for p,digest in notes[cid]['panels'].items():
            check(ROOT/p,digest)
        for route,v in c['results'].items():
            check(ROOT/v['path'],v['sha256'])
            if v['sha256']!=notes[cid]['observations'][route]['result_sha256']:
                raise ValueError('Reviewed photo is not the output')
    return r,notes,path


def metrics(rows,name):
    both = {}
    for r in rows:
        both.setdefault(r['id'],[]).append(r['predictions'][name]['route']==r['expected'])
    return dict(rows=len(rows),correct_rows=sum(sum(x) for x in both.values()),
        semantic_requests=len(both),bilingual_correct=sum(all(v) for v in both.values()),
        wrong_active_route=sum(r['predictions'][name]['route'] not in (r['expected'],'defer') for r in rows),
        opposite_direction=sum(
            r['expected']!='defer' and r['predictions'][name]['route']!='defer'
            and r['expected'].split('_')[1]!=r['predictions'][name]['route'].split('_')[1] for r in rows),
        expected_defer_but_active=sum(r['expected']=='defer' and r['predictions'][name]['route']!='defer' for r in rows),
        wanted_edit_but_deferred=sum(r['expected']!='defer' and r['predictions'][name]['route']=='defer' for r in rows))


def main():
    rows,plan = rows_and_plan()
    torch.set_num_threads(plan['cpu_threads'])
    torch.set_num_interop_threads(1)
    receipt = read(RUN/'receipt.json')
    check(PLAN,receipt['protocol_sha256'])
    check(CACHE/'manifest.json',receipt['cache_sha256'])
    check(RUN/'adapter.pt',receipt['adapter_sha256'])
    check(RUN/'heads.pt',receipt['heads_sha256'])
    entries,manifest = load_cache()
    tail = Tail(manifest)
    adapter = torch.load(RUN/'adapter.pt',map_location='cpu',weights_only=True)
    if adapter['tail_sha256']!=manifest['tail_sha256']:
        raise ValueError('Adapter is not bound to this tail')
    tail.add_lora(adapter['rank'],adapter['alpha'])
    with torch.no_grad():
        for name,p in tail.named_parameters():
            if p.requires_grad:
                p.copy_(adapter['parameters'][name])
    heads = torch.load(RUN/'heads.pt',map_location='cpu',weights_only=True)
    photographs,notes,review_path = images()
    choices = list(plan['labels'].values())
    test = [i for i,r in enumerate(rows) if r['cohort'] not in TRAIN]
    result = []
    with torch.inference_mode():
        for start in range(0,len(test),8):
            batch = test[start:start+8]
            tail_scores = tail([entries[i][0] for i in batch])
            for index,i in enumerate(batch):
                _,final,frozen = entries[i]
                r = rows[i]
                scores = dict(frozen=frozen,tail_lora=tail_scores[index])
                for key,w in heads.items():
                    normalized = ((final-w['mean'])/w['scale']).to(w['weight'].dtype)
                    scores[key+'_head'] = linear(normalized,w['weight'],w['bias'])
                predicted = {name:dict(route=choices[int(v.argmax())],
                    logits={k:float(x) for k,x in zip(choices,v)}) for name,v in scores.items()}
                if r['cohort']=='held_photos':
                    for a in predicted.values():
                        route = a['route']
                        if route=='defer':
                            a.update(visual='deferred',output=None)
                        else:
                            observed = notes[r['group']]['observations'][route]['verdict']
                            visual = observed if route==r['expected'] else 'wrong_route'
                            a.update(visual=visual,own_intent_visual=observed,
                                     output=photographs[r['group']]['results'][route]['path'])
                result.append(dict(r,predictions=predicted))
    summary = {}
    for method in METHODS:
        grouped = {cohort:[r for r in result if r['cohort']==cohort] for cohort in ('held_photos','held_text','r28_regression')}
        summary[method] = {cohort:metrics(items,method) for cohort,items in grouped.items()}
        fresh = grouped['held_photos']+grouped['held_text']
        summary[method]['fresh_all'] = metrics(fresh,method)
        photo = grouped['held_photos']
        ids = {r['id'] for r in photo}
        summary[method]['image_execution'] = dict(language_verdicts=dict(Counter(
            r['predictions'][method]['visual'] for r in photo)),
            both_languages_complete=sum(all(r['predictions'][method]['visual']=='complete' for r in photo if r['id']==cid)
                                        for cid in ids),
            outputs_with_preexisting_visual_failure_grade=sum(
                r['predictions'][method].get('own_intent_visual')=='failed' for r in photo))
    write(EXP/'result.json',dict(at=stamp(),receipt_sha256=sha(RUN/'receipt.json'),
        visual_review_sha256=sha(review_path),summary=summary,rows=result,GPU_hours=0,
        deployed=False,final_acceptance_opened=False))
    print(summary,flush=True)


if __name__=='__main__':
    main()
