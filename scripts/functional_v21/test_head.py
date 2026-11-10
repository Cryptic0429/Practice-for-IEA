"""Contracts for semantic preservation, learned veto and input-only observations."""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .common import ROOT,EXP,TOOLS,read,write,sha,stamp
from .head import Head,compose,normalize,labels,loss_function,load_head
from .texture import features,NAMES

def main():
    torch.set_num_threads(2);torch.manual_seed(20261205)
    magnitude=torch.full((9,),.6)
    parent={'action':'edit','parameters':{'exposure':-5,'contrast':20,'sepia':85}}
    assert compose(parent,magnitude,torch.tensor(-20.),'magnitude-quality')=={'action':'edit','parameters':{'exposure':-60,'contrast':60,'sepia':60}}
    assert compose(parent,magnitude,torch.tensor(20.),'magnitude')==compose(parent,magnitude,torch.tensor(-20.),'magnitude')
    assert compose(parent,magnitude,torch.tensor(20.),'magnitude-quality')=={'action':'decline','reason':'unreliable_quality'}
    for value in [{'action':'keep'},{'action':'decline','reason':'unsupported_tools'},{'action':'decline','reason':'unreliable_quality'}]:
        for policy in ('magnitude','magnitude-quality'):assert compose(value,magnitude,torch.tensor(20.),policy)==value
    try:compose(parent,torch.zeros(9),torch.tensor(-20.),'magnitude');raise AssertionError('Zero magnitude must fail, not remove a requested tool')
    except ValueError as error:assert str(error)=='head_zero_for_parent_active_tool'
    cases=[dict(decision=d) for d in [parent,{'action':'keep'},{'action':'decline','reason':'unsupported_tools'},{'action':'decline','reason':'unreliable_quality'}] for _ in range(2)]
    targets=labels(cases);assert targets[3].tolist()==[True,True,False,False,False,False,True,True]
    x,mean,std=normalize(torch.randn(8,3584),torch.randn(8,16),torch.randn(8,12))
    net=Head();optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
    loss=loss_function(net(x),*targets,torch.tensor(4.));loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
    optimizer.step();net.eval();expected=net(x)
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        for value in (0,128,255):
            path=folder/f'flat-{value}.png';Image.new('RGB',(32,32),(value,)*3).save(path)
            assert np.array_equal(features(path),np.zeros(12,dtype=np.float32))
        path=folder/'random.png';array=np.random.default_rng(20261205).integers(20,230,size=(32,32,3),dtype=np.uint8)
        Image.fromarray(array).save(path)
        assert features(path)[0]>0 and np.array_equal(features(path),features(path))
        path=folder/'head.pt';torch.save(dict(state_dict=net.state_dict(),mean=mean,std=std,tools=list(TOOLS),texture_names=list(NAMES)),path)
        restored,_=load_head(path);assert all(torch.equal(a,b) for a,b in zip(expected,restored(x)))
    assert not torch.cuda.is_initialized()
    write(EXP/'cpu-preflight.json',dict(at=stamp(),status='passed',GPU_calls=0,
        checks=['parent tool/sign preserved','keep and both parent refusals preserved','shared-weight risk veto on/off',
                'zero-strength active tool is error','supervision mask excludes keep/unsupported risk labels',
                'finite actual gradients/update','save/load exact logits','flat/noisy input residual numerical contracts'],
        source_sha256={name:sha(ROOT/'scripts/functional_v21'/name) for name in ('head.py','texture.py','test_head.py')},
        limitation='Numerical and pipeline contracts, not perceptual quality or practical-use evidence.'))
    print('R21 CPU contracts passed; noGPU andno functional rating.')

if __name__=='__main__':main()
