from functional_v1.common import ROOT, GPUJob, read, write, append, sha, stamp, EDITOR_VERSION
NAME = 'functional-v5-20261105'
DATA = ROOT/'data/training'/NAME
EXP = ROOT/'experiments'/NAME
PARENT = ROOT/'models/from-functional-v1/sft-curriculum/seed-01'
HELDOUT_DATA = ROOT/'data/training/functional-v2-20261104'
EVAL = EXP/'evaluation'/EDITOR_VERSION
