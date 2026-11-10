"""The language model owns semantic choices; the head learns strength and risk."""
import torch
from torch import nn
from functional_v1.common import TOOLS
from functional_v15.outcomes import validate_decision,target_text
from .texture import NAMES

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3612,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,10))

    def forward(self,x):
        y=self.net(x)
        return y[...,:9].sigmoid(),y[...,9]

def normalize(hidden,stats,texture,mean=None,std=None):
    assert hidden.shape[-1]==3584 and stats.shape[-1]==16 and texture.shape[-1]==12
    h=hidden/hidden.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
    x=torch.cat([h,stats,texture],-1)
    if mean is None:mean,std=x.mean(0),x.std(0,unbiased=False).clamp_min(.1)
    assert mean.shape==std.shape==(3612,)
    return (x-mean)/std,mean,std

def labels(cases):
    magnitude=[];active=[];quality=[];supported=[]
    for case in cases:
        d=validate_decision(case['decision']);params=d.get('parameters',{})
        magnitude.append([abs(params.get(k,0))/100 for k in TOOLS])
        active.append([bool(params.get(k,0)) for k in TOOLS])
        quality.append(d.get('reason')=='unreliable_quality')
        supported.append(d['action']=='edit' or d.get('reason')=='unreliable_quality')
    return (torch.tensor(magnitude,dtype=torch.float32),torch.tensor(active,dtype=torch.bool),
            torch.tensor(quality,dtype=torch.float32),torch.tensor(supported,dtype=torch.bool))

def loss_function(outputs,magnitude,active,quality,supported,pos_weight):
    import torch.nn.functional as F
    m,q=outputs
    numerical=F.smooth_l1_loss(m[active],magnitude[active],beta=.1) if active.any() else m.sum()*0
    risk=F.binary_cross_entropy_with_logits(q[supported],quality[supported],pos_weight=pos_weight) if supported.any() else q.sum()*0
    paired=m.reshape(-1,2,9);mask=active.reshape(-1,2,9).any(1)
    mag_pair=(paired[:,0]-paired[:,1]).square()[mask].mean() if mask.any() else m.sum()*0
    probabilities=torch.stack([q.sigmoid(),(-q).sigmoid()],-1).reshape(-1,2,2)
    average=probabilities.mean(1,keepdim=True).clamp_min(1e-8)
    terms=(probabilities*(probabilities.clamp_min(1e-8).log()-average.log())).sum(-1).mean(1)
    supported_pair=supported.reshape(-1,2).all(-1)
    risk_pair=terms[supported_pair].mean() if supported_pair.any() else q.sum()*0
    return 4*numerical+risk+.25*(mag_pair+risk_pair)

def compose(parent,magnitude,risk,policy):
    parent=validate_decision(parent)
    assert policy in ('magnitude','magnitude-quality')
    # Explicit keep and both kinds of language-model refusal always survive.
    if parent['action']!='edit':return parent
    assert magnitude.shape==(9,) and risk.ndim==0
    assert torch.isfinite(magnitude).all() and torch.isfinite(risk)
    if not ((magnitude>=0)&(magnitude<=1)).all():raise ValueError('head_magnitude_out_of_range')
    if policy=='magnitude-quality' and float(risk.sigmoid())>=.5:
        return {'action':'decline','reason':'unreliable_quality'}
    values={}
    for name,parent_value in parent['parameters'].items():
        if not parent_value:continue
        strength=int(round(float(magnitude[TOOLS.index(name)])*100))
        if not strength:raise ValueError('head_zero_for_parent_active_tool')
        values[name]=-strength if parent_value<0 else strength
    # No tool invention, direction inversion or fallback from an empty edit.
    return validate_decision({'action':'edit','parameters':values})

def load_head(path):
    item=torch.load(path,map_location='cpu',weights_only=True)
    assert item['tools']==list(TOOLS) and item['texture_names']==list(NAMES)
    net=Head();net.load_state_dict(item['state_dict'],strict=True);net.eval()
    return net,item

def predict_cached(net,item,cached,texture,parent,policy):
    parent=validate_decision(parent)
    if parent['action']!='edit':return target_text(parent),parent
    x,_,_=normalize(cached['last'][None],cached['stats'][None],texture[None],item['mean'],item['std'])
    with torch.inference_mode():m,q=net(x)
    decision=compose(parent,m[0],q[0],policy)
    return target_text(decision),decision
