"""Export anonymous original-size review copies and preserve supplemental notes.

Does not change predictions, pixels, contact-sheet observations or model selection.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/functional-v10-20261106'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(split, bundle):
    from PIL import Image
    protocol = json.loads((EXP / 'evaluation-protocol.json').read_text())
    key = {'native-development': 'native_development', 'old-native': 'old_native_regressions'}[split]
    source = ROOT / protocol['case_files'][key]['path']
    assert sha(source) == protocol['case_files'][key]['sha256']
    cases = {r['id']: r for r in json.loads(source.read_text())}
    folder = EXP / 'review' / split
    manifest = json.loads((folder / bundle / 'manifest.json').read_text())
    raw = {r['item_id']: r for r in map(json.loads, (folder / 'codex-observations.jsonl').read_text().splitlines())}
    destination = folder / bundle / 'full-resolution'
    destination.mkdir(exist_ok=True)
    assert not (destination / 'manifest.json').exists()
    rows = []
    prior_full = folder / 'codex-full-resolution-observations.jsonl'
    already_full = {r['item_id'] for r in map(json.loads, prior_full.read_text().splitlines())} if prior_full.exists() else set()
    for item in manifest['items']:
        note = raw.get(item['item_id'])
        if not note or note.get('prior_final_observation_source') or note['quality'] != 'uncertain' or item['item_id'] in already_full:
            continue
        case = cases[item['case_id']]
        input_path = ROOT / case['image']
        output_path = ROOT / item['image']
        assert sha(input_path) == case['input_sha256']
        assert sha(output_path) == item['image_sha256']
        paths = []
        for label, path in [('INPUT', input_path), (item['item_id'], output_path)]:
            target = destination / f'{case["id"]}-{label}.png'
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                if not target.exists():
                    rgb.save(target)
                with Image.open(target) as check:
                    assert rgb.size == check.size and rgb.tobytes() == check.convert('RGB').tobytes()
                paths.append({'role': label, 'path': str(target.relative_to(ROOT)),
                              'sha256': sha(target), 'size': list(rgb.size),
                              'pixel_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
        rows.append({'item_id': item['item_id'], 'case_id': case['id'],
                     'input_sha256': case['input_sha256'], 'output_sha256': item['image_sha256'],
                     'images': paths})
    (destination / 'manifest.json').write_text(json.dumps({'split': split, 'bundle': bundle,
        'policy': 'Original-size RGB copies, no adjustment or resizing. Identity remains hidden. Separate supplemental notes preserve contact-sheet uncertainty.',
        'items': rows}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'split': split, 'original_size_followups': len(rows),
                      'cases': sorted({r['case_id'] for r in rows})}))


def record(split, bundle, item_id, completion, quality, evidence):
    import datetime as dt
    folder = EXP / 'review' / split
    full = folder / bundle / 'full-resolution'
    manifest = json.loads((full / 'manifest.json').read_text())
    item = next(r for r in manifest['items'] if r['item_id'] == item_id)
    for image in item['images']:
        assert sha(ROOT / image['path']) == image['sha256']
    assert completion in ('C', 'P', 'F', 'U') and quality in ('A', 'm', 'M', 'U')
    assert len(evidence) >= 20
    destination = folder / 'codex-full-resolution-observations.jsonl'
    prior = destination.read_text() if destination.exists() else ''
    assert item_id not in {r['item_id'] for r in map(json.loads, prior.splitlines())}
    row = {**item, 'completion': dict(C='completed', P='partial', F='failed', U='uncertain')[completion],
           'quality': dict(A='acceptable', m='minor_issue', M='major_issue', U='uncertain')[quality],
           'evidence': evidence, 'reviewer': 'Codex direct anonymous original-size AI review; not human',
           'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
           'supplements_raw_contact_sheet': True, 'manifest_sha256': sha(full / 'manifest.json')}
    destination.write_text(prior + json.dumps(row, ensure_ascii=False) + '\n')
    print('Preserved original-size observation:', item_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['native-development', 'old-native'])
    parser.add_argument('--bundle', default='initial-baselines')
    args = parser.parse_args()
    prepare(args.split, args.bundle)
