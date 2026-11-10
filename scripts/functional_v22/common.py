from functional_v1.common import ROOT,read,write,append,sha,stamp,GPUJob,TOOLS
EXP=ROOT/'experiments/functional-v22-20261107'
DATA=ROOT/'data/training/functional-v22-20261107'
PARENT=ROOT/'models/from-functional-v11/sft-shadow-correction/seed-01'
PARENT_SHA='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90'

def protocol():
    p=read(EXP/'training-protocol.json')
    assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
    assert all(sha(ROOT/n)==h for n,h in p['code_hashes'].items())
    assert sha(DATA/'cases.json')==p['sources']['train']['sha256']
    assert sha(DATA/'admission.json')==p['admission_sha256']
    return p
