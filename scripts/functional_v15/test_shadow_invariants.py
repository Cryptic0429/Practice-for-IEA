"""Numerical operator contracts, separate from visible functional quality."""
import os
import numpy as np
from functional_v1.common import ROOT,write,stamp,sha
from functional_v11.editor import edit as old_edit
from .editor import edit


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    rng=np.random.default_rng(20261129);rgb=rng.integers(0,256,(127,113,3),dtype=np.uint8)
    unchanged=[]
    for key in ('exposure','brightness','contrast','highlights','saturation','temperature','tint','sepia'):
        values=(0,1,20,80,100) if key=='sepia' else (-100,-80,-20,-1,0,1,20,80,100)
        for v in values:
            assert np.array_equal(edit(rgb,{key:v},'bounded-fixed'),old_edit(rgb,{key:v},'bounded-fixed'))
            unchanged.append({key:v})
    # Mixed non-shadow requests must also remain byte-identical.
    for _ in range(30):
        p={k:int(rng.integers(-100,101)) for k in ('exposure','brightness','contrast','highlights','saturation','temperature','tint')}
        p['sepia']=int(rng.integers(0,101));assert np.array_equal(edit(rgb,p,'bounded-fixed'),old_edit(rgb,p,'bounded-fixed'));unchanged.append(p)
    ramp=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    checks=[]
    for v in (-100,-60,-1,0,1,30,60,90,100):
        out=edit(ramp,{'shadows':v},'bounded-fixed')
        assert np.all(out>=ramp) if v>=0 else np.all(out<=ramp)
        assert np.array_equal(out[:,0],ramp[:,0]) and np.array_equal(out[:,-1],ramp[:,-1])
        assert np.all(np.diff(out[0,:,0].astype(int))>=0)
        coloured=edit(rgb,{'shadows':v},'bounded-fixed').astype(float);before=rgb.astype(float)
        for a,b in ((0,1),(0,2),(1,2)):
            # A shared pre-quantization gain gives this exact rounding bound.
            difference=np.abs(coloured[...,a]*before[...,b]-coloured[...,b]*before[...,a])
            assert np.all(difference<=(before[...,a]+before[...,b])/2+2)
        checks.append(v)
    assert np.array_equal(edit(rgb,{'shadows':0},'bounded-fixed'),rgb)
    write(ROOT/'experiments/functional-v15-20261106/shadow-invariant-tests.json',dict(at=stamp(),status='passed',
        no_shadow_exact_previous_editor_cases=len(unchanged),shadow_values_checked=checks,black_white_endpoints_preserved=True,
        positive_lifts_negative_lowers_gray=True,gray_ramp_monotone=True,shared_RGB_gain_quantization_bound_passed=True,
        old_editor_sha256=sha(ROOT/'scripts/functional_v11/editor.py'),new_editor_sha256=sha(ROOT/'scripts/functional_v15/editor.py'),
        scope='Numerical operator invariants only; not image quality, label admission, trained model or heldout performance.',GPU_hours=0))
    print('Prototype numeric contracts passed;unchanged non-shadow cases',len(unchanged))


if __name__=='__main__':main()
