"""Download an accepted adapter atomically and verify every transferred file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

LOCAL = Path(__file__).resolve().parents[1]
REMOTE = '/mnt/sdc/zhc/IEA'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'SJTU1']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--model', choices=['A', 'B'], required=True)
    model = cli.parse_args().model
    summary_path = f'{REMOTE}/logs/stage1-{model}-3k/summary.json'
    summary = json.loads(subprocess.check_output(SSH + ['cat ' + shlex.quote(summary_path)], text=True))
    assert summary['status'] == 'completed_and_reload_verified' and summary['global_step'] == 376
    folder = LOCAL / 'artifacts/adapters' / model
    folder.mkdir(parents=True, exist_ok=True)
    verified = []
    for name in ['adapter_model.safetensors', 'adapter_config.json']:
        remote = summary['adapter_path'] + '/' + name
        expected = subprocess.check_output(SSH + ['sha256sum -- ' + shlex.quote(remote)], text=True).split()[0]
        if name.endswith('.safetensors'):
            assert expected == summary['adapter_sha256']
        target = folder / name
        if target.exists():
            assert sha(target) == expected, f'Do not overwrite an existing different artifact: {target}'
        else:
            temporary = folder / (name + '.download')
            assert not temporary.exists(), f'Inspect prior partial transfer: {temporary}'
            subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                            'SJTU1:' + remote, str(temporary)], check=True)
            assert sha(temporary) == expected, f'Transfer checksum mismatch: {name}'
            temporary.replace(target)
        verified.append(dict(file=name, bytes=target.stat().st_size, sha256=expected))
    report = dict(verified_at=datetime.now(timezone.utc).isoformat(), model=model,
                  remote_adapter=summary['adapter_path'], base_model_revision=summary['base_model_revision'],
                  best_checkpoint=summary['best_checkpoint'], files=verified,
                  requires_base_model=True,
                  note='These are LoRA adapter weights, not a standalone full Qwen model.')
    (folder / 'download-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
