"""Evaluate each fully saved epoch on one GPU after the frozen baselines finish."""
import argparse
import os
import subprocess
import sys
import time

from functional_v1.common import ROOT, read, write, stamp, GPUJob
from .diagnose_editor import EXP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu
    protocol = read(EXP / 'training-protocol.json')
    settings = protocol['training']
    status_path = EXP / 'checkpoint-queue-status.json'
    deadline = time.monotonic() + 7200

    def waiting(reason):
        resource_path = ROOT / 'experiments/functional-v1-20261104/resources.json'
        resource = read(resource_path)
        if GPUJob.consumed(resource) >= resource['limit_gpu_hours'] or time.monotonic() >= deadline:
            write(status_path, dict(at=stamp(), status='stopped_before_next_inference', reason='shared GPU budget or queue time limit', waiting_for=reason))
            raise SystemExit('Budget/time limit reached; queue state preserved')
        write(status_path, dict(at=stamp(), status='waiting_without_GPU_allocation', waiting_for=reason))
        time.sleep(15)

    while True:
        baseline = EXP / 'baseline-queue-status.json'
        state = read(baseline) if baseline.exists() else {}
        if state.get('status') == 'completed_predictions':
            break
        if state.get('status') == 'failed':
            raise RuntimeError('Frozen baseline queue failed; do not skip its controls')
        waiting('frozen baseline queue')
    completed = []
    from safetensors.torch import load_file
    for epoch in range(1, settings['epochs'] + 1):
        step = epoch * settings['steps_per_epoch']
        for route in protocol['routes']:
            run = EXP / 'runs' / f'{route}-seed-{settings["seed"]}'
            checkpoint = run / f'checkpoint-{step}'
            while True:
                run_state = read(run / 'status.json')
                if run_state['status'] in ('failed', 'budget_stopped'):
                    raise RuntimeError('Training stopped: ' + route)
                weights, state_file = checkpoint / 'adapter_model.safetensors', checkpoint / 'trainer_state.json'
                if weights.exists() and state_file.exists() and time.time() - max(weights.stat().st_mtime, state_file.stat().st_mtime) > 2:
                    if read(state_file)['global_step'] == step:
                        tensors = load_file(str(weights), device='cpu')
                        assert len(tensors) == (112 if route == 'language' else 116)
                        del tensors
                        break
                waiting(f'{route} complete epoch {epoch}')
            while int(subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()) >= 512:
                waiting('assigned GPU free; never take over a loaded user trial or another job')
            name = ('language' if route == 'language' else 'language-merger') + '-e' + str(epoch)
            write(status_path, dict(at=stamp(), status='evaluating', model=name, completed=completed))
            with (EXP / ('evaluate-' + name + '.log')).open('ab') as log:
                result = subprocess.run([sys.executable, '-u', '-m', 'functional_v7.evaluate', '--model', name,
                                         '--adapter', str(checkpoint.relative_to(ROOT)), '--gpu', args.gpu],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write(status_path, dict(at=stamp(), status='failed', model=name, returncode=result.returncode, completed=completed))
                raise SystemExit(result.returncode)
            completed.append(name)
    write(status_path, dict(at=stamp(), status='completed_stage1_predictions', models=completed,
                            visual_review_pending=True, functional_improvement_claimed=False))


if __name__ == '__main__':
    main()
