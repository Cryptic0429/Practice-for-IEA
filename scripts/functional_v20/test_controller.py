"""CPU contracts before any 7B feature extraction or fitting."""
import tempfile
from pathlib import Path
import torch
from .common import EXP, write, sha, stamp, ROOT, TOOLS
from .controller import Controller, ACTIONS, labels, decode, normalize, loss_function, load_controller, predict_cached, js_pair

def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261203)
    cases = [dict(decision={'action':'edit','parameters':{'exposure':-45,'sepia':70}}),
             dict(decision={'action':'keep'}),
             dict(decision={'action':'decline','reason':'unsupported_tools'}),
             dict(decision={'action':'decline','reason':'unreliable_quality'})]
    a,d,m = labels(cases)
    for i, case in enumerate(cases):
        aa = torch.full((4,), -20.); aa[a[i]]=20
        dd = torch.full((9,3), -20.); dd[torch.arange(9),d[i]]=20
        assert decode(aa,dd,m[i]) == case['decision']
    try:
        decode(torch.tensor([10.,0,0,0]), torch.tensor([[10.,0,0]]*9), torch.zeros(9))
        raise AssertionError('Empty edits must fail')
    except ValueError: pass
    paired = [c for c in cases for _ in range(2)]
    a,d,m = labels(paired)
    hidden=torch.randn(8,3584);stats=torch.randn(8,16)
    x,mean,std=normalize(hidden,stats)
    assert torch.equal(mean, normalize(hidden,stats)[1])
    net=Controller();optim=torch.optim.AdamW(net.parameters(),lr=.001)
    loss=loss_function(net(x),a,d,m,torch.ones(9,3));loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
    optim.step();net.eval()
    actual=net(x)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'controller.pt'
        torch.save(dict(state_dict=net.state_dict(),mean=mean,std=std,tools=list(TOOLS),actions=list(ACTIONS),route='last'),path)
        restored,item=load_controller(path)
        assert all(torch.equal(left,right) for left,right in zip(actual,restored(x)))
    same=torch.randn(4,4).repeat_interleave(2,0)
    assert abs(float(js_pair(same)))<1e-7
    assert not torch.cuda.is_initialized()
    write(EXP/'controller-cpu-preflight.json',dict(at=stamp(),status='passed',GPU_calls=0,
          checks=['signed tool and4action roundtrip','empty edit rejected','finite gradients and update',
                  'save/load exact logits','paired language JS zero for identical distributions','TRAIN-only normalizer explicit inputs'],
          implementation_sha256=sha(ROOT/'scripts/functional_v20/controller.py'), functional_quality_claim=False))
    print('R20 CPU contracts passed; not a functional result')

if __name__ == '__main__': main()
