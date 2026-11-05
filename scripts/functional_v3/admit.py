"""SHA-bound visual admission, explicit label challenges, and matched training pools."""
import copy,hashlib,json,math
from collections import Counter
from .common import ROOT,ROUND2_DATA,ROUND2_EXP,EXP,DATA,PARENT,read,write,sha,stamp


def main():
    import numpy as np
    from functional_v1.editor import load_rgb,edit
    if (DATA/'admission.json').exists():raise RuntimeError('Admitted data already frozen')
    oldcases=read(ROUND2_DATA/'train-final-cases.json');cases={c['id']:copy.deepcopy(c) for c in oldcases}
    manifests={s['case_id']:s for s in read(EXP/'preference-review/manifest.json')['sheets']}
    subset=read(EXP/'preference-review/review-subset.json')['sheets'];observations=read(EXP/'preference-review/observations.json')
    assert {o['case_id'] for o in observations}=={s['case_id'] for s in subset} and len(observations)==48
    prefs=[];challenges=[];reviewed=[]
    def replay(c,entry):
        assert sha(ROOT/entry['image'])==entry['sha256']
        assert np.array_equal(edit(load_rgb(ROOT/c['image']),entry['parameters']),load_rgb(ROOT/entry['image']))
    def preference(c,lang,chosen,rejected,observation,origin):
        replay(c,chosen);replay(c,rejected)
        assert chosen['sha256']!=rejected['sha256'] and chosen['parameters']!=rejected['parameters']
        return {'id':f'pref-{c["id"]}-{lang}-{rejected["sha256"][:8]}','case_id':c['id'],'source_id':c['source_id'],
          'language':lang,'domain':c['domain'],'family':c['family'],'kind':'preference',
          'chosen':chosen['parameters'],'rejected':rejected['parameters'],
          'chosen_image':chosen['image'],'chosen_sha256':chosen['sha256'],
          'rejected_image':rejected['image'],'rejected_sha256':rejected['sha256'],
          'evidence':observation['evidence'],'sheet':observation['sheet'],
          'sheet_sha256':observation['sheet_sha256'],'request_origin':origin}
    for o in observations:
        m=manifests[o['case_id']];assert sha(ROOT/o['sheet'])==o['sheet_sha256']==m['sha256']
        reviewed.append(o['case_id'])
        if o['preferred_index'] is None:continue
        c=cases[o['case_id']];chosen=m['entries'][o['preferred_index']-1];replay(c,chosen)
        for idx in o['rejected_indices']:
            rejected=m['entries'][idx-1]
            if rejected['origin']=='approved_target':
                assert chosen['origin']=='policy'
                challenges.append({'case_id':c['id'],'previous_parameters':c['parameters'],'previous_target':c['target'],'previous_sha256':c['target_sha256'],
                  'replacement_parameters':chosen['parameters'],'replacement_target':chosen['image'],'replacement_sha256':chosen['sha256'],
                  'evidence':o['evidence'],'sheet':o['sheet'],'sheet_sha256':o['sheet_sha256']})
                # Round2 files remain untouched. Only the new round's case snapshot is corrected.
                c.update(parameters=chosen['parameters'],target=chosen['image'],target_sha256=chosen['sha256'],status='round3_direct_visual_label_revision')
                continue
            assert rejected['origin']=='policy'
            for lang in rejected['languages']:
                prefs.append(preference(c,lang,chosen,rejected,o,'AI-authored controlled request'))
    native={c['id']:c for c in read(DATA/'natural-cases.json')};nm={r['case_id']:r for r in read(EXP/'natural-review/manifest.json')}
    natural_obs=read(EXP/'natural-review/observations.json');assert len(natural_obs)==10
    predictions=list(map(json.loads,(EXP/'natural-collection/predictions.jsonl').read_text().splitlines()))
    for o in natural_obs:
        c=copy.deepcopy(native[o['case_id']]);m=nm[c['id']]
        assert sha(ROOT/o['sheet'])==m['sha256']==o['sheet_sha256']
        assert o['chosen_completion']=='completed' and o['chosen_quality']=='acceptable'
        chosen=m['entries'][o['chosen_index']-1];replay(c,chosen)
        c.pop('proposals');c.update(parameters=chosen['parameters'],target=chosen['image'],target_sha256=chosen['sha256'],status='round3_natural_request_direct_visual_approved')
        cases[c['id']]=c
        rejected_sha={m['entries'][r['index']-1]['sha256'] for r in o['rejected_outputs']}
        for pred in predictions:
            if pred['case_id']!=c['id'] or pred['status']!='executed' or pred['output_sha256'] not in rejected_sha:continue
            reject={'parameters':pred['parameters'],'image':pred['output_image'],'sha256':pred['output_sha256']}
            prefs.append(preference(c,pred['language'],chosen,reject,o,'Untouched original GIER EN request / AI ZH translation'))
    assert len({r['id'] for r in prefs})==len(prefs) and len(prefs)>=16
    write(DATA/'label-challenges.json',{'at':stamp(),'previous_round_unchanged':True,'revisions':challenges,
       'interpretation':'New direct pairwise inspection preferred a real policy render over some previous approved targets. Revisions apply only to this new training snapshot, not frozen round2 results.'})
    # Retain all ten native-request positive pairs, revised cases, and ten no-op photo groups.
    mandatory=set(native)|{r['case_id'] for r in challenges}
    for domain in ('portrait','landscape'):
        pool=[c for c in cases.values() if c['domain']==domain and c['action']=='identity']
        pool.sort(key=lambda c:hashlib.sha256((c['id']+'anchor20261107').encode()).hexdigest())
        mandatory.update(c['id'] for c in pool[:5])
    target_cases=max(len(mandatory),math.ceil(len(prefs)/2));selected=set(mandatory)
    pool=sorted((c for c in cases.values() if c['id'] not in selected),key=lambda c:hashlib.sha256((c['id']+'anchor20261107').encode()).hexdigest())
    for c in pool:
        counts=Counter(cases[cid]['domain'] for cid in selected)
        if len(selected)>=target_cases and counts['portrait']==counts['landscape']:break
        desired='portrait' if counts['portrait']<counts['landscape'] else 'landscape'
        if counts['portrait']!=counts['landscape'] and c['domain']!=desired:continue
        selected.add(c['id'])
    assert Counter(cases[cid]['domain'] for cid in selected)['portrait']==Counter(cases[cid]['domain'] for cid in selected)['landscape']
    anchors=[]
    for cid in sorted(selected):
        c=cases[cid];assert c['split']=='train' and c['source_id'].startswith('train-')
        assert sha(ROOT/c['target'])==c['target_sha256']
        replay(c,{'image':c['target'],'sha256':c['target_sha256'],'parameters':c['parameters']})
        for language in ('zh','en'):anchors.append({'id':f'anchor-{cid}-{language}','case_id':cid,'source_id':c['source_id'],'kind':'anchor','language':language,'domain':c['domain'],'family':c['family'],'chosen':c['parameters'],'request_origin':'original natural EN / AI ZH' if cid in native else 'AI-authored controlled request'})
    write(DATA/'cases.json',list(cases.values()));write(DATA/'preferences.json',prefs);write(DATA/'anchors.json',anchors);write(DATA/'training-pool.json',prefs+anchors)
    used={r['case_id'] for r in prefs+anchors}
    accepted={'status':'direct_AI_visual_admitted_not_human','at':stamp(),'preference_rows':len(prefs),'preference_semantic_cases':len({r['case_id'] for r in prefs}),
      'anchor_rows':len(anchors),'total_rows':len(prefs+anchors),'semantic_cases':len(used),'photo_groups':len({cases[cid]['source_id'] for cid in used}),
      'preference_domains':dict(Counter(r['domain'] for r in prefs)),'anchor_domains':dict(Counter(r['domain'] for r in anchors)),
      'used_photo_domains':{d:len({cases[cid]['source_id'] for cid in used if cases[cid]['domain']==d}) for d in ('portrait','landscape')},
      'preference_languages':dict(Counter(r['language'] for r in prefs)),
      'native_cases':10,'native_language_rows_in_anchors':20,'native_photo_domains':{'portrait':5,'landscape':5},
      'controlled_reviewed':48,'controlled_candidate_cases':101,'unreviewed_excluded':53,'controlled_no_clear_preference':sum(o['preferred_index'] is None for o in observations),
      'label_challenges':len(challenges),'pool_sha256':sha(DATA/'training-pool.json'),'cases_sha256':sha(DATA/'cases.json'),
      'review_sha256':sha(EXP/'preference-review/observations.json'),'native_review_sha256':sha(EXP/'natural-review/observations.json'),
      'review_boundary':'Only global tone visible in full-frame sheets; no fine-detail improvement claim. Language variants share semantic/source group.',
      'source_boundary':'All used source groups are from round2 training; development and acceptance never admitted.'}
    write(DATA/'admission.json',accepted)
    protocol={'status':'frozen_before_training','at':stamp(),'parent':str(PARENT.relative_to(ROOT)),'parent_sha256':sha(PARENT/'adapter_model.safetensors'),
      'admission_sha256':sha(DATA/'admission.json'),'pool_sha256':accepted['pool_sha256'],'cases_sha256':accepted['cases_sha256'],
      'training':{'routes':['DPO plus chosen SFT anchor','matched chosen-only SFT'],'epochs':2,'learning_rate':1e-5,'effective_batch':4,'beta':.1,'chosen_sft_weight_on_preference_rows':.1,'anchor_sft_weight':1.,'seed':20261107,'rank':8,'alpha':16,'quantization':'NF4','layers':'language q/v only; vision frozen','reference':'Frozen parent response log probabilities precomputed in identical train dtype and processor; no second live model','rows_each':accepted['total_rows']},
      'selection':'No major damage on controlled development and six natural-development probes; then more both-language natural requests fully qualified, then controlled functional count, then less minor damage, paired blind preference, earlier checkpoint. No L1. These six photos are reused development, not independent user validation.',
      'promotion_gate':'Require improvement over parent on natural requests (>3/6), no new major damage relative to zero-major round1 baseline, controlled functional completion at least81/100. Otherwise fail this correction pilot; no acceptance opening for promotion.',
      'acceptance_status':'closed; lock final models before any inference; final cases hash bound',
      'acceptance_final_cases_sha256':sha(ROUND2_DATA/'acceptance-final-cases.json'),
      'core_code_hashes':read(ROUND2_EXP/'protocol.json')['core_code_hashes'],
      'budget':'Same original16 occupied GPU hours across all rounds; GPUs0/2/3 protect1; no reset.',
      'research_hypothesis':'Rendered outcome preference could penalise over-adjustment better than JSON imitation alone; unproven for this policy.',
      'references':read(EXP/'collection-protocol.json')['references']}
    write(EXP/'training-protocol.json',protocol)
    print(json.dumps(accepted,ensure_ascii=False))

if __name__=='__main__':main()
