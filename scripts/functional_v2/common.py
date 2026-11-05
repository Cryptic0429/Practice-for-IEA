from functional_v1.common import ROOT, GPUJob, BudgetExceeded, read, write, append, sha, stamp, TOOLS, EDITOR_VERSION
NAME='functional-v2-20261104'
DATA=ROOT/'data/training'/NAME
EXP=ROOT/'experiments'/NAME
PARENT=ROOT/'models/from-sft-calibrated/sft-functional/seed-01'
SHARED_LEDGER=ROOT/'experiments/functional-v1-20261104/resources.json'

EVAL=EXP/'evaluation'/EDITOR_VERSION
