"""Review a completed fixed epoch while immutable later training continues."""
import argparse
import os
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v6 import visual_review as established
from .runtime import EXP,ROUTES
from .visual_review import configure,decisions
from .outcomes import EDITOR_VERSION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,choices=[1,2],required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    protocol=read(EXP/'training-protocol.json')
    assert all(sha(ROOT/n)==d for n,d in protocol['code_hashes'].items())
    queue=read(EXP/'evaluation-queue.json');models=[];bindings={}
    for route in ROUTES:
        mid=route.replace('_','-')+f'-e{args.epoch}'
        job=next(j for j in queue['jobs'] if j['model']==mid)
        assert job.get('returncode')==0, 'Wait for the actual evaluation process to finish'
        checkpoint=EXP/'runs'/f'{route}-seed-{protocol["training"]["seed"]}'/f'checkpoint-{501*args.epoch}'
        assert read(checkpoint/'trainer_state.json')['global_step']==501*args.epoch
        assert sha(checkpoint/'adapter_model.safetensors')==job['adapter_sha256']
        for split in ('native-development','old-native','decision-development'):
            path=EXP/'evaluation'/EDITOR_VERSION/split/mid/'identity.json'
            assert read(path)['adapter_sha256']==job['adapter_sha256']
        bindings[mid]=dict(adapter_sha256=job['adapter_sha256'],checkpoint=str(checkpoint.relative_to(ROOT)))
        models.append(mid)
    configure()
    for split in ('native-development','old-native'):
        established.prepare(split,models,f'epoch{args.epoch}-routes')
    decisions(models)
    write(EXP/f'epoch{args.epoch}-review-binding.json',dict(at=stamp(),epoch=args.epoch,models=bindings,
        selection_performed=False,training_protocol_changed=False,all_four_checkpoints_still_required=True,
        policy='Review completed epoch predictions only. Later fixed optimization continues; no learning-rate, data, loss or checkpoint-count change from interim observations. Final selection only after both routes and both epochs are reviewed.'))


if __name__=='__main__':main()
