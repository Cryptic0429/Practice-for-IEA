"""Independent causal-mask and hard-negative gradient checks, CPU only."""
import torch
import torch.nn.functional as F
from .objective import sequence_logp,ranking_loss


def main():
    torch.manual_seed(20261129)
    logits=torch.randn(1,7,11,requires_grad=True)
    labels=torch.tensor([[-100,-100,-100,2,5,6,4]])
    value,count=sequence_logp(logits,labels)
    independent=sum(torch.log_softmax(logits[0,i-1],dim=-1)[labels[0,i]] for i in range(3,7))
    assert count==4 and torch.allclose(value,independent,atol=1e-6)
    grad,=torch.autograd.grad(value,logits)
    assert torch.count_nonzero(grad[:,:2])==0 and torch.count_nonzero(grad[:,-1])==0
    chosen=torch.tensor(-8.,requires_grad=True);rejected=torch.tensor(-3.,requires_grad=True)
    loss=ranking_loss(chosen,rejected,31,32)
    good,bad=torch.autograd.grad(loss,(chosen,rejected))
    assert good<0 and bad>0 and torch.isfinite(loss)
    assert ranking_loss(chosen.detach()+1,rejected.detach()-1,31,32)<loss.detach()
    assert ranking_loss(chosen.detach()-1,rejected.detach()+1,31,32)>loss.detach()
    assert torch.allclose(ranking_loss(torch.tensor(-31.),torch.tensor(-32.),31,32),
                          ranking_loss(torch.tensor(-62.),torch.tensor(-32.),62,32))
    assert torch.allclose(-value/count,F.cross_entropy(logits[0,2:6],labels[0,3:7]))
    print('PASS: causal assistant mask, independent CE, chosen/rejected gradient directions, finite ranking objective; no optimizer or GPU.')


if __name__=='__main__':main()
