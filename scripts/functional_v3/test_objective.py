"""Check masking/causal alignment and preference gradient signs against independent CE."""
import torch
import torch.nn.functional as F
from .objective import sequence_logp,dpo_loss


def main():
    torch.manual_seed(73)
    scores=torch.randn(1,9,13,requires_grad=True)
    labels=torch.tensor([[-100,-100,-100,-100,2,4,9,1,6]])
    logp,n=sequence_logp(scores,labels)
    independent=F.cross_entropy(scores[:,:-1].reshape(-1,13),labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
    assert n==5 and torch.allclose(-logp/n,independent)
    g1=torch.autograd.grad(-logp/n,scores,retain_graph=True)[0];g2=torch.autograd.grad(independent,scores)[0]
    assert torch.allclose(g1,g2) and g1[:,:3].abs().sum()==0 and g1[:,3].abs().sum()>0
    chosen=torch.tensor(-8.,requires_grad=True);rejected=torch.tensor(-3.,requires_grad=True)
    loss=dpo_loss(chosen,rejected,-8.,-3.);a,b=torch.autograd.grad(loss,(chosen,rejected))
    assert torch.allclose(loss,torch.log(torch.tensor(2.))) and a<0 and b>0
    improved=dpo_loss(chosen.detach()+1,rejected.detach()-1,-8.,-3.)
    assert improved<loss.detach()
    print('Assistant mask/causal shift, independent CE gradients and DPO preference signs passed')
if __name__=='__main__':main()
