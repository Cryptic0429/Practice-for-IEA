"""Prepare all predeclared checkpoint images together, without exposing scores."""
import os
from functional_v1.common import read,sha,ROOT
from .runtime import EXP,ROUTES
from .visual_review import configure,decisions
from functional_v6 import visual_review as established


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=read(EXP/'training-protocol.json')
    assert all(sha(ROOT/n)==d for n,d in p['code_hashes'].items())
    for route in ROUTES:
        assert read(EXP/'runs'/f'{route}-seed-{p["training"]["seed"]}'/'status.json')['status']=='completed'
    assert read(EXP/'evaluation-queue.json')['status']=='completed'
    models=[route.replace('_','-')+f'-e{epoch}' for epoch in (1,2) for route in ROUTES]
    configure()
    for split in ('native-development','old-native'):
        established.prepare(split,models,'joint-routes-epochs')
    decisions(models)


if __name__=='__main__':main()
