"""Known user-reported regression; freeze literal wording after diagnostic lock."""
from functional_v1.common import ROOT,read,write,sha,stamp


def main():
    exp=ROOT/'experiments/functional-v15-20261106'
    target=exp/'demo/case12-protocol.json';assert not target.exists()
    selected=read(exp/'development-selection.json')
    assert not selected['qualified'] and selected['selected_model_id'] is None
    assert len(selected['models'])==1
    model=next(iter(selected['models'].values()))
    raw=ROOT/'data/sources/gier/GIER.json'
    source=next(r for r in read(raw) if r['input']=='7QSfy_7QSfy.jpg' and r['output']=='7QSfy_5V4XJ.jpg')
    assert 'Please, add brown tone' in source['expert_summary']
    old=ROOT/'experiments/functional-v1-20261104/challenge/functional-current-20261104/identity.json'
    previous=read(old);image=ROOT/'data/images/7QSfy_7QSfy.jpg'
    assert sha(image)==previous['input_sha256']
    write(target,dict(at=stamp(),status='frozen_known_regression_before_new_trial_requests',
        id='historical-case12-brown',image=str(image.relative_to(ROOT)),input_sha256=sha(image),
        instructions=dict(en='Please, add brown tone',zh='请添加棕色色调。'),
        instruction_origin=dict(en='Original GIER expert_summary literal request shown in user screenshot',zh='AI Chinese translation; not an independent original annotation'),
        change='Visible overall brown tone rather than retained green foliage plus mere brighter/yellower enhancement.',
        preserve='Source framing/objects and basic highlight/shadow detail; no broad new clipping or unrelated enhancement.',
        original_record=source,raw_source_sha256=sha(raw),prior_viewed_receipt=str(old.relative_to(ROOT)),prior_viewed_sha256=sha(old),
        model_id=model['model_id'],adapter_sha256=model['adapter_sha256'],editor=model['editor'],
        selection_sha256=sha(exp/'development-selection.json'),used_for_checkpoint_selection=False,
        independent_acceptance=False,training_use=False,
        limitation='One historical source repeatedly viewed since the original screenshot. Not a new independent benchmark, seed replicate, user trial, or proof of general usefulness. No checkpoint/parameter tuning from these results.'))
    print('Frozen literal case12 EN and translated ZH for locked diagnostic model; no new independent test claim.')


if __name__=='__main__':main()
