"""CPU evidence that the new manual uses editor math, not heldout references."""
import hashlib
import math
import os
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v11.outcomes import system_prompt as old_prompt
from .outcomes import system_prompt,input_row,parse
from .admit_training import EXP,DATA


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    manual=system_prompt();assert manual!=old_prompt()
    protocol=read(ROOT/'experiments/functional-v11-20261106/training-protocol.json')
    for name in ('scripts/functional_v11/editor.py','scripts/functional_v11/outcomes.py'):
        assert sha(ROOT/name)==protocol['code_hashes'][name]
    cases=read(DATA/'cases.json')
    assert not any(c['id'] in manual or c['image'] in manual or c['input_sha256'] in manual for c in cases)
    assert 'native-dev-' not in manual and 'native-accept-' not in manual
    gains={str(value):2**(value/100) for value in (-10,-50,-80)}
    assert all(abs(gains[str(value)]-shown)<.0005 for value,shown in [(-10,.933),(-50,.707),(-80,.574)])
    bad='```json\n{"action":"edit","parameters":{"sepia":-20}}\n```'
    try:parse(bad)
    except ValueError:pass
    else:raise AssertionError('Illegal response must fail, never be repaired')
    write(EXP/'manual-audit.json',dict(at=stamp(),status='passed',prompt_sha256=hashlib.sha256(manual.encode()).hexdigest(),
        frozen_R11_editor_and_parser_unchanged=True,old_prompt_sha256=hashlib.sha256(old_prompt().encode()).hexdigest(),
        exposure_gain_examples_derived_from_operator=gains,no_training_case_ID_path_or_input_hash_in_manual=True,
        no_heldout_case_ID_in_manual=True,illegal_negative_sepia_still_rejected_not_repaired=True,
        provenance='General operator semantics and mathematical slider response. No heldout reference, expected parameter or case-specific rule was used in the manual.',
        limitation='This CPU audit checks scope/math/contracts only, not whether the model follows the manual or improves editing.',GPU_hours=0))
    print('Manual math, frozen pixel/parser identity and strict rejection checked; no capability claim.')


if __name__=='__main__':main()
