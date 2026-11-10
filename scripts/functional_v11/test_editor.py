"""Directional/zero/positive-regression checks for the changed negative gain."""
import os
import numpy as np
from functional_v1.common import ROOT, write, sha, stamp
from functional_v7.editor_soft import edit as frozen_edit
from functional_v7.outcomes import system_prompt as frozen_prompt
from .editor import edit, bounded_gain
from .outcomes import system_prompt
from .runtime import EXP


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert system_prompt() == frozen_prompt()
    ramp = np.linspace(0, 1, 65537, dtype=np.float64)
    curves = []
    for value in (-100, -60, -20, -1, 0, 1, 20, 60, 100):
        out = bounded_gain(ramp, value)
        assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
        assert (np.diff(out) >= 0).all() and out[0] == 0
        if value < 0:
            assert np.allclose(out, ramp*2**(value/100), rtol=0, atol=1e-14)
            assert out[-1] < 1 and (out[1:] < ramp[1:]).all()
        elif value > 0:
            assert out[-1] == 1 and (out[1:-1] > ramp[1:-1]).all()
        else:
            assert np.array_equal(out, ramp)
        curves.append(dict(slider=value, bounded_monotone=True, white_endpoint=float(out[-1])))
    eps=1e-5
    derivative_left=(.5-bounded_gain(np.array([.5]),-eps)[0])/eps
    derivative_right=(bounded_gain(np.array([.5]),eps)[0]-.5)/eps
    assert abs(derivative_left-derivative_right) < 1e-8
    rgb=np.random.default_rng(20261123).integers(0,256,(137,193,3),dtype=np.uint8)
    assert np.array_equal(edit(rgb,{}), rgb)
    unchanged=0
    for tool in ('exposure','brightness','contrast','highlights','shadows','saturation','temperature','tint','sepia'):
        for value in (-100,-25,0,25,100):
            if (tool in ('exposure','brightness') and value < 0) or (tool=='sepia' and value<0):continue
            for variant in ('bounded-fixed','bounded-adaptive'):
                assert np.array_equal(edit(rgb,{tool:value},variant),frozen_edit(rgb,{tool:value},variant)),(tool,value,variant)
                unchanged+=1
    grey=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    for tool in ('exposure','brightness'):
        for value in (-100,-60,-20,20,60,100):
            output=edit(grey,{tool:value},'bounded-fixed')
            assert (np.diff(output[0,:,0].astype(int))>=0).all()
            assert output[0,0].max()==0
            if value < 0:
                assert output[0,-1].max() < 255 and (output.astype(int)<=grey.astype(int)).all()
            else:assert (output.astype(int)>=grey.astype(int)).all()
    write(EXP/'editor-invariant-tests.json',dict(at=stamp(),status='passed',negative_curves=curves,
        positive_and_other_operator_exact_replays=unchanged,zero_exact=True,small_signal_midgrey_continuity=True,
        prompt_byte_identical_to_R7=True,code_hashes={name:sha(ROOT/'scripts/functional_v11'/name) for name in ('editor.py','outcomes.py','test_editor.py')},
        scope='CPU numerical contract, not visual acceptance or trained capability',GPU_hours=0))
    print('R11 editor direction/zero/continuity and',unchanged,'unchanged operator replays passed.')


if __name__ == '__main__':main()
