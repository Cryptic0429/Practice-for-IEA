"""Record a full-size repair followup; never admits an example automatically."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def record(folder_name, case_id, label, completion, quality, evidence):
    root = Path(__file__).resolve().parents[2]
    assert folder_name in ('label-repairs', 'label-quality-repairs')
    folder = root / 'experiments/functional-v7-20261105' / folder_name
    manifest = json.loads((folder / 'manifest.json').read_text())
    candidate = next(r for r in manifest['records'] if r['case_id'] == case_id and r.get('label') == label)
    for name, digest in [('image', 'image_sha256'), ('input_image', 'input_sha256')]:
        assert hashlib.sha256((root / candidate[name]).read_bytes()).hexdigest() == candidate[digest]
    assert completion in ('completed', 'partial', 'failed', 'uncertain')
    assert quality in ('acceptable', 'minor_issue', 'major_issue', 'uncertain') and evidence.strip()
    target = folder / 'full-resolution-observations.json'
    data = json.loads(target.read_text()) if target.exists() else dict(
        reviewer='Codex direct full-frame/full-size visual AI followup; not human or blinded', observations=[])
    assert not any(r['case_id'] == case_id and r['label'] == label for r in data['observations'])
    data['observations'].append(dict(
        case_id=case_id, label=label, at=dt.datetime.now(dt.timezone.utc).isoformat(),
        completion=completion, quality=quality, evidence=evidence, admitted=False,
        input_image=candidate['input_image'], input_sha256=candidate['input_sha256'],
        image=candidate['image'], image_sha256=candidate['image_sha256']))
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(target)
    print('Recorded repair followup:', case_id, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('folder_name', 'case_id', 'label', 'completion', 'quality', 'evidence'):
        parser.add_argument(key)
    args = parser.parse_args()
    record(**vars(args))
