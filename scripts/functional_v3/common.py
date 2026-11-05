from functional_v1.common import ROOT, GPUJob, read, write, append, sha, stamp, EDITOR_VERSION
from functional_v2.common import DATA as ROUND2_DATA, EXP as ROUND2_EXP
NAME='functional-v3-20261104'
EXP=ROOT/'experiments'/NAME
DATA=ROOT/'data/training'/NAME
PARENT=ROOT/'models/from-functional-v1/sft-curriculum/seed-01'

EVAL=EXP/'evaluation'/EDITOR_VERSION
