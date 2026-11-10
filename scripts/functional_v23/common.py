from functional_v1.common import ROOT,BASE,read,write,append,sha,stamp,GPUJob,BudgetExceeded
EXP=ROOT/'experiments/functional-v23-20261107'
DATA=ROOT/'data/training/functional-v22-20261107'
PARENT=ROOT/'models/from-functional-v11/sft-shadow-correction/seed-01'
PARENT_SHA='97c84f1799fb319799c86515402574b3b585593763e5388eb30d0df3e0d6ea90'

def protocol():
 p=read(EXP/'training-protocol.json')
 assert sha(PARENT/'adapter_model.safetensors')==PARENT_SHA
 assert all(sha(ROOT/n)==h for n,h in p['code_hashes'].items())
 for n,h in p['data_files'].items():assert sha(DATA/n)==h
 return p
