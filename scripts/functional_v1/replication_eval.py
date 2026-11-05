"""Evaluate the already selected epoch in the second seed, without reselecting it."""
import os
import time
from .common import EXP, read
from .evaluate import predict_all


def main():
    selection = read(EXP/'development-selection.json')
    assert selection['replication_authorized_by_protocol']
    chosen = selection['models'][selection['selected_model_id']]
    start = selection['selected_model_id'].split('-')[1]
    run = EXP/'runs'/f'{start}-seed-20261105'
    adapter=run/f"checkpoint-{chosen['epoch']*110}"
    deadline = time.monotonic()+3600
    while time.monotonic()<deadline:
        if (run/'status.json').exists():
            state=read(run/'status.json')
            if state['status']=='failed': raise RuntimeError(state)
            # The trainer has moved beyond this immutable checkpoint before reading it.
            if state.get('step',0)>chosen['epoch']*110 and (adapter/'adapter_model.safetensors').exists():break
            if state['status']=='completed': break
        time.sleep(20)
    else: raise TimeoutError('Replication training did not finish within one hour')
    predict_all('replication-selected', adapter, 'development', os.environ['CUDA_VISIBLE_DEVICES'])


if __name__=='__main__': main()
