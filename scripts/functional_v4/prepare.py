"""Reuse directly reviewed targets and freeze one shared dataset for both input routes."""
import copy,json
from collections import Counter
from .common import *


def main():
    import numpy as np
    from functional_v1.editor import edit,load_rgb
    if (EXP/'protocol.json').exists():raise RuntimeError('Already frozen')
    previous=read(CORRECTED_DATA/'admission.json');assert previous['status']=='direct_AI_visual_admitted_not_human'
    cases=read(CORRECTED_DATA/'cases.json');assert len(cases)==370
    assert len({c['source_id'] for c in cases})==60 and Counter({c['source_id']:c['domain'] for c in cases}.values())=={'portrait':30,'landscape':30}
    for c in cases:
        assert c['split']=='train' and c['source_id'].startswith('train-')
        assert sha(ROOT/c['image'])==c['input_sha256'] and sha(ROOT/c['target'])==c['target_sha256']
        assert np.array_equal(edit(load_rgb(ROOT/c['image']),c['parameters']),load_rgb(ROOT/c['target']))
    used_groups={c['group_id'] for c in cases}
    for split in ('development','acceptance'):
        assert not used_groups & {c['group_id'] for c in read(HELDOUT_DATA/f'{split}-final-cases.json')}
    write(DATA/'cases.json',cases);rows=[]
    for c in cases:
        # Ten original native requests are deliberately oversampled; no new independent photos.
        repeat=10 if c['id'].startswith('natural-') else 1
        for repetition in range(repeat):
            for language in ('zh','en'):rows.append({'id':f'{c["id"]}-{language}-repeat{repetition:02d}','case_id':c['id'],'language':language,'parameters':c['parameters'],'domain':c['domain'],'repeat':repetition,'origin':'original EN / AI ZH translation' if repeat==10 else 'AI-authored controlled request'})
    assert len(rows)==920 and len({(r['case_id'],r['language']) for r in rows})==740
    write(DATA/'train-accepted.json',rows)
    write(DATA/'admission.json',{'at':stamp(),'status':'reused_direct_visual_targets_and_six_recorded_revisions','semantic_cases':370,'source_groups':60,'photo_domains':{'portrait':30,'landscape':30},'language_rows_unique':740,'actual_rows_each_route':920,'domains_by_rows':dict(Counter(r['domain'] for r in rows)),
      'controlled_semantic_cases':360,'native_original_semantic_cases':10,'native_repeated_language_rows':200,'controlled_language_rows':720,
      'native_fraction_by_semantic_cases':10/370,'native_fraction_by_training_rows':200/920,
      'target_review_lineage':'All360 original controlled targets directly inspected in round2, six challenged/revised by round3 direct pair inspection; ten original native-request targets re-reviewed in round3. No unreviewed policy output admitted.',
      'correction_admission_sha256':sha(CORRECTED_DATA/'admission.json'),'label_challenges_sha256':sha(CORRECTED_DATA/'label-challenges.json'),
      'replay':'All370 targets replayed pixel-identical. Targets/canonical photos/review never enter inference inputs.',
      'cases_sha256':sha(DATA/'cases.json'),'dataset_sha256':sha(DATA/'train-accepted.json')})
    p={'status':'targets_frozen_before_training','at':stamp(),'hypothesis':'Current policy often emits similar amplitudes for different image states. Test whether measured low-level properties from the actual visual input improve strength selection beyond the same amount of continued SFT.',
      'routes':['plain','measured'],'parent':str(PARENT.relative_to(ROOT)),'parent_sha256':sha(PARENT/'adapter_model.safetensors'),
      'dataset_sha256':sha(DATA/'train-accepted.json'),'cases_sha256':sha(DATA/'cases.json'),'admission_sha256':sha(DATA/'admission.json'),
      'training':{'epochs':2,'learning_rate':5e-5,'effective_batch':4,'steps':460,'seed':20261108,'rank':8,'alpha':16,'quantization':'NF4','layers':'language q/v; frozen vision','rows_each':920},
      'input_allowlist':['actual input image','unmodified original ZH or EN request','frozen tool specification plus same quality note for both routes','measured route only: deterministic luminance/RGB/saturation statistics computed from the SAME resized image presented to the vision encoder'],
      'input_forbidden':['target/reference/canonical image or its statistics','target parameters','grading/evidence','case/source IDs','dataset task family or action'],
      'code_hashes':{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'scripts/functional_v4/runtime.py',ROOT/'scripts/functional_v1/common.py',ROOT/'scripts/functional_v1/runtime.py',ROOT/'scripts/functional_v1/editor.py']},
      'heldout_hashes':{s:sha(HELDOUT_DATA/f'{s}-final-cases.json') for s in ('development','natural-use','acceptance')},
      'selection':'No severe damage on100 controlled development tasks or six original-request development probes; prioritize both-language complete natural requests, then controlled completion, paired visual preference, earlier checkpoint. No L1 or parameter-error selection.',
      'promotion_gate':{'native_qualified_more_than':3,'controlled_qualified_at_least':81,'major_quality_cases':0},
      'acceptance_policy':'Keep closed during development; lock all final model weights/runtime routes before any final inference. No checkpoint switch after opening.',
      'budget':'Same original16 occupied GPU hours, GPUs0/2/3, protect1. This is a new input hypothesis, not unlimited repeated SFT.'}
    write(EXP/'protocol.json',p);print(json.dumps({'rows':len(rows),'unique_language_rows':740,'source_groups':60,'dataset_sha256':p['dataset_sha256']}))
if __name__=='__main__':main()
