"""Frozen R30 candidate grid and file identities."""
from functional_v1.common import ROOT

EXP = ROOT/'experiments/functional-v30-20261107'
DATA = ROOT/'data/training/functional-v30-20261107'
PLAN = EXP/'protocol.json'
RECIPES = dict(linear30=('linear',-30),linear12=('linear',-12),
               shoulder30=('shoulder',-30),shoulder12=('shoulder',-12))
