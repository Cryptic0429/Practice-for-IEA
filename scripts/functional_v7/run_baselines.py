"""Sequential frozen new-runtime baselines on one explicitly assigned free GPU."""
import argparse
import os
import subprocess
import sys
from functional_v1.common import ROOT, read, write, sha, stamp
from .diagnose_editor import EXP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu
    specs = [
        ('current-default-new-runtime', 'models/from-sft-calibrated/sft-functional/seed-01',
         '54161bd9c9f2e3aa518572be6cf1dad81f8bebe33d92d31107d64d1f3c4435b0'),
        ('round6-parent-new-runtime', 'models/from-functional-v2/sft-native-coverage/seed-01',
         '39d0e3b91c321ee0abeca8f3260116c76405d371127ba118e04822701e09f66c')]
    path = EXP / 'baseline-queue-status.json'
    for name, adapter, digest in specs:
        assert sha(ROOT / adapter / 'adapter_model.safetensors') == digest
        write(path, dict(at=stamp(), status='running', model=name, gpu=args.gpu))
        with (EXP / ('evaluate-' + name + '.log')).open('ab') as log:
            result = subprocess.run([sys.executable, '-u', '-m', 'functional_v7.evaluate', '--model', name,
                                     '--adapter', adapter, '--gpu', args.gpu], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(path, dict(at=stamp(), status='failed', model=name, returncode=result.returncode))
            raise SystemExit(result.returncode)
    write(path, dict(at=stamp(), status='completed_predictions', models=[s[0] for s in specs],
                     visual_review_pending=True, functional_improvement_claimed=False))


if __name__ == '__main__':
    main()
