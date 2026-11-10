"""A zero-output trainable RGB-statistic bias on every current-image vision token."""
import torch
from torch import nn

class StatBridge(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.first=nn.Linear(16,128)
        self.last=nn.Linear(128,hidden_size)
        nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)

    def forward(self,x):
        assert x.ndim==2 and x.shape==(1,16), 'One current image per request only'
        return self.last(torch.nn.functional.gelu(self.first(x)))

def visual_module(model):
    modules=[m for name,m in model.named_modules() if name=='visual' or name.endswith('.visual')]
    assert len(modules)==1
    return modules[0]

def install_before_peft(base,hidden_size):
    visual=visual_module(base)
    assert not hasattr(visual,'stat_bridge')
    parameter=next(visual.merger.parameters())
    visual.stat_bridge=StatBridge(hidden_size).to(device=parameter.device,dtype=torch.float32)
    visual.functional_photometry=None
    def add_statistics(module,inputs,output):
        current=visual.functional_photometry
        assert current is not None and current.shape==(1,16), 'Missing current-image features; do not silently use zero features'
        assert output.ndim==2 and output.shape[1]==hidden_size
        delta=visual.stat_bridge(current.to(device=output.device,dtype=torch.float32))
        return output+delta.to(dtype=output.dtype)
    visual.merger.register_forward_hook(add_statistics)

def set_features(model,values):
    value=torch.as_tensor(values,dtype=torch.float32).reshape(1,16)
    assert torch.isfinite(value).all()
    visual_module(model).functional_photometry=value
