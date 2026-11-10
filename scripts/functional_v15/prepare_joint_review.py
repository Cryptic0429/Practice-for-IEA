"""Prepare epoch-blind direct review without loss-based checkpoint selection."""
import os
from functional_v1.common import ROOT,read,sha
from .runtime import EXP
from .visual_review import configure,decisions
from functional_v6 import visual_review as established


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert read(EXP/'runs/language_merger-seed-20261127/status.json')['status']=='completed'
    assert read(EXP/'evaluation-queue.json')['status']=='completed'
    protocol=read(EXP/'training-protocol.json')
    assert all(sha(ROOT/path)==h for path,h in protocol['code_hashes'].items())
    models=['language-merger-e1','language-merger-e2']
    configure()
    for split in ('native-development','old-native'):established.prepare(split,models,'joint-epochs')
    decisions(models)


if __name__=='__main__':main()
