"""Detach one bounded evaluation job without replacing any previous run or log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path('/mnt/sdc/zhc/IEA')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Base', 'A', 'B', 'Official', 'BaseSchema'], required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--run-name', default='full')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu
    assert args.run_name.replace('-', '').replace('_', '').isalnum()
    occupants = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-compute-apps=pid',
                                         '--format=csv,noheader,nounits'], text=True).strip()
    assert not occupants, f'GPU {args.gpu} has existing compute processes: {occupants}'
    free = int(subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 14000, f'GPU {args.gpu} has {free} MiB free'
    out = ROOT / f'outputs/evaluation-v1/{args.split}-{args.model}-{args.run_name}'
    out.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / 'envs/iea/bin/python'), str(ROOT / 'scripts/evaluate_stage1.py'),
               '--model', args.model, '--split', args.split, '--gpu', args.gpu,
               '--limit', str(args.limit), '--run-name', args.run_name]
    with (out / 'worker.log').open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 env=dict(os.environ, PYTHONUNBUFFERED='1'))
    manifest = dict(pid=child.pid, gpu=args.gpu, command=command, output=str(out),
                    launched_at=datetime.now(timezone.utc).isoformat())
    (out / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
