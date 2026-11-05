"""Anonymous, pixel-deduplicated direct AI review; no automatic success ratings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import textwrap
from .common import ROOT, EXP, EVAL, read, write, append, sha, stamp
from .common import HELDOUT_DATA as DATA


def key(case, prediction):
    spec = json.dumps({k:case[k] for k in ('id','input_sha256','instructions','expected_change','must_preserve')}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((spec+prediction['output_sha256']).encode()).hexdigest()[:16]


def prepare(split, model_ids, name):
    from PIL import Image, ImageDraw, ImageOps
    from functional_v1.curate import font
    cases = read(DATA/f'{split}-final-cases.json')
    lookup = {c['id']:c for c in cases}
    output = EXP/'review'/split/name
    if (output/'manifest.json').exists():
        raise RuntimeError('Preserve prepared review bundles; use a new bundle name')
    output.mkdir(parents=True, exist_ok=True)
    unique, mapping, invalid = {}, [], []
    identities = {}
    for model_id in model_ids:
        folder = EVAL/split/model_id
        identities[model_id] = read(folder/'identity.json')
        rows = [json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==2*len(cases) and len({(r['case_id'],r['language']) for r in rows})==len(rows)
        for row in rows:
            if row['status']!='executed':
                invalid.append({'model_id':model_id,'case_id':row['case_id'],'language':row['language'], 'reason':row.get('error')})
                continue
            assert sha(ROOT/row['output_image'])==row['output_sha256']
            cid = row['case_id']
            item_id = key(lookup[cid], row)
            unique.setdefault(item_id, {'item_id':item_id,'case_id':cid,'image':row['output_image'], 'image_sha256':row['output_sha256']})
            mapping.append({'model_id':model_id,'case_id':cid,'language':row['language'],'item_id':item_id})
    sheets = []
    prior_file=EXP/'review'/split/'codex-observations.jsonl'
    prior={r['item_id']:r for r in map(json.loads,prior_file.read_text().splitlines())} if prior_file.exists() else {}
    already_viewed={item for item,r in prior.items() if (ROOT/r['sheet']).exists() and sha(ROOT/r['sheet'])==r['sheet_sha256']}
    for case in cases:
        entries = sorted([r for r in unique.values() if r['case_id']==case['id'] and r['item_id'] not in already_viewed], key=lambda r:r['item_id'])
        for offset in range(0,len(entries),5):
            subset = entries[offset:offset+5]
            columns=min(3,1+len(subset))
            canvas = Image.new('RGB',(columns*600,115+600*math.ceil((1+len(subset))/columns)),'white')
            d = ImageDraw.Draw(canvas)
            d.text((16,12),case['id']+' | '+case['action'],fill='black',font=font(25))
            for i,line in enumerate(textwrap.wrap(case['instructions']['en'],43*columns)):
                d.text((16,51+i*27),line,fill='black',font=font(22))
            # Same-scale full frames; no model identity, parameters, reference or scores.
            sources = [('INPUT',ROOT/case['image'])]+[(r['item_id'],ROOT/r['image']) for r in subset]
            for i,(label,path) in enumerate(sources):
                x,y=(i%columns)*600,115+(i//columns)*600
                d.text((x+12,y),label,fill='black',font=font(22))
                with Image.open(path) as im:
                    fitted=ImageOps.contain(im.convert('RGB'),(580,550),Image.Resampling.LANCZOS)
                canvas.paste(fitted,(x+10+(580-fitted.width)//2,y+40+(550-fitted.height)//2))
            file=output/f"{case['id']}-{offset//5+1:02d}.jpg"
            canvas.save(file,quality=95,subsampling=0)
            sheets.append({'sheet':str(file.relative_to(ROOT)),'sha256':sha(file),'case_id':case['id'],
                           'items':[r['item_id'] for r in subset]})
    write(output/'manifest.json',{'split':split,'name':name,'created_at':stamp(),
        'cases_sha256':sha(DATA/f'{split}-final-cases.json'),'models':identities,'mapping':mapping,
        'invalid':invalid,'items':list(unique.values()),'sheets':sheets,
        'reused_exact_image_observations':sorted(set(unique)&already_viewed),
        'review_policy':'Actual direct Codex observation. Exact same output for the same bilingual semantic case is reviewed once and mapped back; translations are not independent samples.',
        'no_automatic_completion_ratings':True})
    print(json.dumps({'bundle':str(output),'unique_images':len(unique),'language_outputs':len(mapping),'invalid':len(invalid),'sheets':len(sheets)}))


def record(bundle, observations):
    manifest=read(bundle/'manifest.json')
    items={r['item_id']:r for r in manifest['items']}
    by_item={item:s for s in manifest['sheets'] for item in s['items']}
    destination=EXP/'review'/manifest['split']/'codex-observations.jsonl'
    for row in read(observations):
        row=dict(row)
        if 'item_id' not in row:
            rendered=[item for sheet in manifest['sheets'] if sheet['case_id']==row['case_id'] for item in sheet['items']]
            row['item_id']=rendered[row.pop('index')-1]
        assert row['completion'] in ('completed','partial','failed','uncertain')
        assert row['quality'] in ('acceptable','minor_issue','major_issue','uncertain')
        assert len(row['evidence'])>=20
        item=items[row['item_id']]
        sheet=by_item[row['item_id']]
        assert sha(ROOT/sheet['sheet'])==sheet['sha256']
        append(destination,{**row,'case_id':item['case_id'],'image_sha256':item['image_sha256'],
            'sheet':sheet['sheet'],'sheet_sha256':sheet['sha256'],'bundle':manifest['name'],
            'reviewer':'Codex direct visual review; AI-assisted, not human','observed_at':stamp()})


def summarize(split, model_ids):
    cases={c['id']:c for c in read(DATA/f'{split}-final-cases.json')}
    review_file=EXP/'review'/split/'codex-observations.jsonl'
    observations={r['item_id']:r for r in map(json.loads,review_file.read_text().splitlines())} if review_file.exists() else {}
    summaries={}
    for model_id in model_ids:
        folder=EVAL/split/model_id
        rows=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()))
        scores=[]
        missing=[]
        for row in rows:
            if row['status']!='executed':
                assessment={'completion':'failed','quality':'uncertain','evidence':'No executable edit was produced.'}
                item_id=None
            else:
                item_id=key(cases[row['case_id']],row)
                assessment=observations.get(item_id)
                if assessment is None:
                    missing.append(item_id)
                    continue
                assert assessment['image_sha256']==row['output_sha256']
                assert sha(ROOT/assessment['sheet'])==assessment['sheet_sha256']
            scores.append({'case_id':row['case_id'],'family':row['family'],'language':row['language'],
                           'item_id':item_id,'assessment':assessment})
        grouped=defaultdict(list)
        for r in scores: grouped[r['case_id']].append(r)
        raw_complete=lambda v:len(v)==2 and all(r['assessment']['completion']=='completed' and r['assessment']['quality']=='acceptable' for r in v)
        lookup={(r['case_id'],r['language']):r for r in rows}
        identical_strength=[];excluded=set()
        for cid,c in cases.items():
            if not (c['action'].startswith('style-') and c['action'].endswith('-weak')):continue
            strong=cid[:-5]+'-strong'
            for language in ('zh','en'):
                weakrow=lookup.get((cid,language),{});strongrow=lookup.get((strong,language),{})
                if weakrow.get('status')==strongrow.get('status')=='executed' and weakrow['output_sha256']==strongrow['output_sha256']:
                    identical_strength.append({'weak_case':cid,'strong_case':strong,'language':language,'output_sha256':weakrow['output_sha256']})
                    excluded.update((cid,strong))
        # Frozen case criteria already require a visibly different weak/strong result.
        # An identical render cannot meet that criterion; unequal renders still need direct review.
        complete=lambda v:raw_complete(v) and v[0]['case_id'] not in excluded
        summary={'model_id':model_id,'status':'completed' if not missing and len(scores)==2*len(cases) else 'incomplete',
            'primary_reviewer':'Codex direct visual review; AI-assisted, not human','human_review':False,
            'reviewed_outputs':len(scores),'missing_unique_images':sorted(set(missing)),
            'fully_completed_both_languages':sum(complete(v) for v in grouped.values()),
            'raw_individual_completed_both_languages':sum(raw_complete(v) for v in grouped.values()),
            'identical_strength_pairs':identical_strength,
            'strength_criterion':'Frozen style criteria require distinguishable weak/strong outcomes. Pixel-identical pairs excluded from completed counts; other outcomes require direct visual grading, not parameter error or L1.',
            'photo_groups':len({c['source_id'] for c in cases.values()}),
            'all_tasks_complete_photo_groups':sum(all(complete(grouped.get(cid,[])) for cid,c in cases.items() if c['source_id']==sid) for sid in {c['source_id'] for c in cases.values()}),
            'major_quality_issues':sum(any(r['assessment']['quality']=='major_issue' for r in v) for v in grouped.values()),
            'major_issue_cases':sorted(cid for cid,v in grouped.items() if any(r['assessment']['quality']=='major_issue' for r in v)),
            'per_family':{f:{'completed':sum(complete(v) for v in grouped.values() if v[0]['family']==f),
                             'cases':sum(c['family']==f for c in cases.values())} for f in sorted({c['family'] for c in cases.values()})},
            'per_language':{lang:dict(Counter(r['assessment']['completion'] for r in scores if r['language']==lang)) for lang in ('zh','en')},
            'per_domain':{d:{'completed':sum(complete(v) for cid,v in grouped.items() if cases[cid]['domain']==d), 'cases':sum(c['domain']==d for c in cases.values())} for d in ('portrait','landscape')},
            'selection_uses_L1':False,'cases_sha256':sha(DATA/f'{split}-final-cases.json'),
            'review_sha256':sha(review_file) if review_file.exists() else None}
        write(folder/'direct-functional-summary.json',summary)
        write(folder/'direct-review.json',scores)
        summaries[model_id]=summary
    print(json.dumps({k:{f:v[f] for f in ('status','fully_completed_both_languages','major_quality_issues','missing_unique_images')} for k,v in summaries.items()}))
    return summaries


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['prepare','record','summarize'])
    ap.add_argument('--split',choices=['development','acceptance','natural-use'],default='development')
    ap.add_argument('--models',nargs='+')
    ap.add_argument('--name')
    ap.add_argument('--bundle',type=Path)
    ap.add_argument('--observations',type=Path)
    args=ap.parse_args()
    if args.command=='prepare':prepare(args.split,args.models,args.name)
    elif args.command=='record':record(args.bundle,args.observations)
    else:summarize(args.split,args.models)


if __name__=='__main__':main()
