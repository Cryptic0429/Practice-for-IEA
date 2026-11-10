from functional_v1.common import ROOT, read, write, append, sha, stamp, GPUJob, TOOLS
EXP=ROOT/'experiments/functional-v21-20261107'
DATA=ROOT/'data/training/functional-v21-20261107'
PARENT=ROOT/'models/from-functional-v11/sft-shadow-correction/seed-01'
PARENT_SHA='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90'
POLICIES=('magnitude','magnitude-quality')

def protocol():
    value=read(EXP/'training-protocol.json')
    for name,digest in value['code_hashes'].items():assert sha(ROOT/name)==digest,name
    assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
    return value
