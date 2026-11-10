"""R20 locations; frozen data and parent remain explicit dependencies."""
from functional_v1.common import ROOT, read, write, append, sha, stamp, GPUJob, TOOLS
EXP = ROOT / 'experiments/functional-v20-20261107'
DATA = ROOT / 'data/training/functional-v20-20261107'
ROUTES = ('last', 'request_mean')
PARENT = ROOT / 'models/from-functional-v11/sft-shadow-correction/seed-01'
PARENT_SHA = '97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90'

def protocol():
    value = read(EXP / 'training-protocol.json')
    for path, digest in value['code_hashes'].items():
        assert sha(ROOT / path) == digest, path
    assert sha(PARENT / 'adapter_model.safetensors') == PARENT_SHA
    return value
