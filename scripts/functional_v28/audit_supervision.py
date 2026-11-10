"""Measure exact instruction coverage and sign collisions in the admitted R22 pool."""
from collections import defaultdict
import unicodedata

from functional_v1.common import ROOT,read,sha,stamp,write
from .boards import EXP

PRIOR = ROOT/'experiments/functional-v22-20261107/training-protocol.json'


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def main():
    spec = read(PRIOR)['sources']['train']
    source = ROOT/spec['path']
    if sha(source)!=spec['sha256']:
        raise ValueError('Prior admitted training data changed')
    cases = read(source)
    routes,opposites = defaultdict(list),defaultdict(lambda:defaultdict(list))
    for c in cases:
        decision = c['decision']
        if decision['action']!='edit' or len(decision['parameters'])!=1:
            continue
        tool,value = next(iter(decision['parameters'].items()))
        if not value:
            continue
        sign = '+' if value>0 else '-'
        routes[tool+sign].append(c)
        for lang,text in c['instructions'].items():
            opposites[tool,lang,normalize(text)][sign].append((c['id'],c['group_id']))
    route_summary = {k:dict(semantic_cases=len(v),original_groups=len({c['group_id'] for c in v}),
        unique_literal_requests={lang:len({normalize(c['instructions'][lang]) for c in v}) for lang in ('zh','en')},
        unique_bilingual_request_pairs=len({tuple(normalize(c['instructions'][lang]) for lang in ('zh','en')) for c in v}))
        for k,v in sorted(routes.items())}
    collisions = [dict(tool=tool,language=lang,instruction=text,negative=by_sign['-'],positive=by_sign['+'])
                  for (tool,lang,text),by_sign in sorted(opposites.items()) if set(by_sign)=={'+','-'}]
    write(EXP/'training-intent-audit.json',dict(at=stamp(),source=spec,source_cases=len(cases),
        single_tool_cases=sum(map(len,routes.values())),routes=route_summary,
        exact_same_text_with_opposite_tool_sign=collisions,
        policy='Exact normalized matching only; no inferred semantic labels. Same text can legitimately need different signs on different images. This audit measures corpus diversity and the limit of text-only direction from these labels.',
        GPU_hours=0))
    print(route_summary)
    print('Exact same-text tool polarity collisions:',[(r['tool'],r['language'],r['instruction']) for r in collisions])


if __name__ == '__main__':
    main()
