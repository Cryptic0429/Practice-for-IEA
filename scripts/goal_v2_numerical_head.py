"""Structured gates and numeric values on frozen multimodal input features."""
import torch
from goal_v2_json_runtime import TOOLS

class ActionHead(torch.nn.Module):
    def __init__(self,dim,hidden=128,dropout=.1):
        super().__init__()
        self.trunk=torch.nn.Sequential(torch.nn.Linear(dim,hidden),torch.nn.SiLU(),torch.nn.Dropout(dropout))
        self.gates=torch.nn.Linear(hidden,len(TOOLS));self.values=torch.nn.Linear(hidden,len(TOOLS))
    def forward(self,x):
        h=self.trunk(x);return self.gates(h),self.values(h)

def normalized(features,mean,std):return ((features-mean)/std).clamp(-8,8)
def decode(gates,values):
    # Fixed threshold; no reference-based per-image search or output blending.
    return torch.where(gates.sigmoid()>=.5,(values*25).clamp(-100,100),0.)

def objective(gates,values,target):
    active=(target!=0).float()
    reg=torch.nn.functional.smooth_l1_loss(values,target/25,reduction='none')
    return (reg*active).sum()/active.sum().clamp_min(1)+.25*torch.nn.functional.binary_cross_entropy_with_logits(gates,active)

def predict_checkpoint(cp,features):
    x=normalized(torch.as_tensor(features,dtype=torch.float32),cp['mean'],cp['std'])
    if cp.get('model_kind')=='conditional-kernel-ridge':
        k=(x.double()@cp['train_features'].double().T)/x.shape[1]
        gates=cp['gate_mean']+k@cp['gate_coeff']
        values=torch.stack([item['mean']+k[:,item['indices']]@item['coeff'] for item in cp['value_models']],dim=1)
        return torch.where(gates>=.5,values.clamp(-100,100),0.).float()
    model=ActionHead(x.shape[1]);model.load_state_dict(cp['model']);model.eval()
    return decode(*model(x))
