from functional_v1.common import ROOT,GPUJob,read,write,append,sha,stamp,EDITOR_VERSION
from functional_v2.common import DATA as HELDOUT_DATA, EXP as ROUND2_EXP
from functional_v3.common import DATA as CORRECTED_DATA, EXP as ROUND3_EXP
NAME='functional-v4-20261105'
DATA=ROOT/'data/training'/NAME
EXP=ROOT/'experiments'/NAME
EVAL=EXP/'evaluation'/EDITOR_VERSION
PARENT=ROOT/'models/from-functional-v1/sft-curriculum/seed-01'
