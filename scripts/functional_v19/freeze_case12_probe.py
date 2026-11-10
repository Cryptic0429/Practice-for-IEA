"""Retain the user's original brown-tone failure as a post-selection regression."""
from functional_v1.common import ROOT,read,write,sha,stamp
EXP=ROOT/'experiments/functional-v19-20261107'

def main():
    target=EXP/'demo/case12-protocol.json';assert not target.exists()
    selection=read(EXP/'development-selection.json');assert len(selection['models'])==1
    mid,meta=next(iter(selection['models'].items()))
    prior=ROOT/'experiments/functional-v15-20261106/demo/case12-protocol.json';old=read(prior)
    assert sha(ROOT/old['image'])==old['input_sha256']
    assert old['instructions']==dict(en='Please, add brown tone',zh='请添加棕色色调。')
    assert sha(ROOT/meta['adapter']/'adapter_model.safetensors')==meta['adapter_sha256']
    record={k:old[k] for k in ('id','image','input_sha256','instructions','instruction_origin','change','preserve')}
    record.update(at=stamp(),status='known_user_failure_regression_frozen_after_checkpoint_choice_before_trial',
        model_id=mid,adapter_sha256=meta['adapter_sha256'],selection_sha256=sha(EXP/'development-selection.json'),
        source_protocol=str(prior.relative_to(ROOT)),source_protocol_sha256=sha(prior),
        used_for_checkpoint_selection=False,independent_acceptance=False,request_training_use=False,
        limitation='Repeatedly viewed historical source from the original user screenshot. Not new independent validation or user approval; no checkpoint, parameter or label changes from this probe.')
    write(target,record);print('Frozen known case12 original literal request,after model choice; no selection or generalization claim.')

if __name__=='__main__':main()
