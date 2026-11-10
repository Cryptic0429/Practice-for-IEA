"""Check exact loader aliases and inherit documented parity without another GPU load."""
import hashlib
import os
import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v13 import runtime as old
from functional_v11 import outcomes as old_contract
from . import runtime, outcomes


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    prior = ROOT/'experiments/functional-v13-20261106'
    old_protocol = read(prior/'training-protocol.json')
    for path,digest in old_protocol['code_hashes'].items(): assert sha(ROOT/path) == digest
    for path,digest in read(runtime.EXP/'editor-only-protocol.json')['code_hashes'].items(): assert sha(ROOT/path) == digest
    names = ['processor','build_batch','verify_origin','create_adapter','transfer_parent','set_trainable','expected_count']
    assert all(getattr(runtime,n) is getattr(old,n) for n in names)
    assert runtime.previous is old and runtime.ROUTES == ('language_merger',)
    assert runtime.PARENT_SHA == old.PARENT_SHA == sha(runtime.PARENT/'adapter_model.safetensors')
    assert runtime.expected_count('language_merger') == 116
    assert all(getattr(outcomes,n) is getattr(old_contract,n) for n in ['parse','input_row','system_prompt','validate_decision','target_text'])
    transfer = read(prior/'runtime-transfer-audit.json')
    parity = read(prior/'initial-function-parity.json')
    assert transfer['runtime_sha256'] == sha(ROOT/'scripts/functional_v13/runtime.py')
    assert parity['status'] == 'passed' and parity['parent_sha256'] == runtime.PARENT_SHA
    assert all(r['max_difference_from_thin'] == 0 for r in parity['records'])
    write(runtime.EXP/'runtime-audit-inheritance.json', dict(at=stamp(), status='passed_exact_loader_alias_inheritance',
        runtime_sha256=sha(ROOT/'scripts/functional_v15/runtime.py'), parent_sha256=runtime.PARENT_SHA,
        exact_aliases=names, underlying_runtime_sha256=sha(ROOT/'scripts/functional_v13/runtime.py'),
        prior_transfer=dict(path=str((prior/'runtime-transfer-audit.json').relative_to(ROOT)),sha256=sha(prior/'runtime-transfer-audit.json'),result=transfer['results'][0]),
        prior_GPU_parity=dict(path=str((prior/'initial-function-parity.json').relative_to(ROOT)),sha256=sha(prior/'initial-function-parity.json'),records=[r for r in parity['records'] if r['route']=='language_merger']),
        new_GPU_load=False,new_forward=False,GPU_hours=0,
        limitation='Transparent reuse of exact R13 loader, adapter construction, tensor transfer, processor and prompt audits. No new R15 parity measurement or quality proof. First actual R15 gradient/optimizer state is checked during training.'))
    rgb = np.array([[[10,30,60],[180,190,210]],[[0,0,0],[255,255,255]]],dtype=np.uint8)
    keep = outcomes.execute(rgb,dict(action='keep'))
    assert keep['status']=='unchanged' and np.array_equal(keep['result'],rgb)
    for reason in outcomes.DECLINE_REASONS:
        result=outcomes.execute(rgb,dict(action='decline',reason=reason))
        assert result['status']=='declined' and result['result'] is None
    decisions=[dict(action='edit',parameters={name:v}) for name in ('exposure','brightness','contrast','highlights','shadows','saturation','temperature','tint','sepia') for v in ((30,) if name=='sepia' else (-30,30))]
    for d in decisions:
        assert outcomes.parse(outcomes.target_text(d)) == d
        assert outcomes.execute(rgb,d)['status']=='edited'
    invalid=['{}','```json\n{"action":"edit","parameters":{"contrast":101}}\n```','```json\n{"action":"edit","parameters":{"contrast":1,"contrast":2}}\n```','```json\n{"action":"edit","parameters":{"exposure":0}}\n```','```json\n{"action":"edit","parameters":{"local_removal":50}}\n```']
    for response in invalid:
        try:outcomes.parse(response)
        except (ValueError,TypeError):pass
        else:raise AssertionError(response)
    write(runtime.EXP/'outcome-contract-check.json',dict(at=stamp(),status='passed',editor=outcomes.EDITOR_VERSION,
        keep_exact_RGB=True,decline_no_image=True,tool_decision_roundtrips=len(decisions),invalid_rejected=len(invalid),
        prompt_sha256=hashlib.sha256(outcomes.system_prompt().encode()).hexdigest(),GPU_hours=0,
        scope='Parser/no-output/identity contract only; actual 925 targets and 44 quality challenge pixels checked separately.'))
    print('Exact runtime aliases and inherited116-tensor transfer/initial parity verified; keep/decline/parser checks passed; zero GPU.')


if __name__=='__main__':main()
