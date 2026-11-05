"""Decode completed anonymous observations without modifying the frozen notes."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/goal-v2-20261103/blind-review'
M = ROOT / 'metadata/goal-v2-20261103'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    receipt = json.loads((D / 'observations-freeze-receipt.json').read_text())
    assert sha(D / 'observations-before-decoding.json') == receipt['observations_sha256']
    notes = json.loads((D / 'observations-before-decoding.json').read_text())
    assert notes['status'] == 'complete_before_decoding'
    assert not notes['model_identities_decoded'] and len(notes['cases']) == 30
    plan = json.loads((M / 'final-visual-plan.json').read_text())
    assert sha(D / 'final-visual-key.private.json') == plan['key_sha256']
    assert sha(D / 'manifest.json') == receipt['manifest_sha256']
    key = json.loads((D / 'final-visual-key.private.json').read_text())['case_mapping']
    provenance = json.loads((D / 'final-review-provenance.private.json').read_text())['sources']
    manifest = json.loads((D / 'manifest.json').read_text())
    frozen = datetime.datetime.fromisoformat(receipt['frozen_at'])
    for name in ['final-visual-key.private.json', 'final-review-provenance.private.json']:
        assert (D / name).stat().st_mtime >= frozen.timestamp()
    counter = {}
    decoded = []
    for case in notes['cases']:
        spec = next(row for row in manifest['cases'] if row['case'] == case['case'])
        assert case['id'] == spec['id'] and len(case['outputs']) == 7
        for panel in spec['panels']:
            entry = next(row for row in provenance if row['case'] == case['case'] and row['panel'] == panel['label'])
            assert panel['sha256'] == entry['source_sha256']
            if panel['label'].startswith('Output'):
                assert entry['model'] == key[str(case['case'])][panel['label']]
        for row in case['outputs']:
            name = key[str(case['case'])][row['output']]
            counts = counter.setdefault(name, {field: collections.Counter() for field in notes['rubric']})
            for field in counts:
                counts[field][row[field]] += 1
            decoded.append({'case': case['case'], 'id': case['id'], 'instruction_text': case['instruction'],
                            'model': name, **row})
    summary = {
        'status': 'completed_observed_then_decoded',
        'decoded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'observations_sha256': receipt['observations_sha256'],
        'key_sha256': sha(D / 'final-visual-key.private.json'),
        'source_provenance_sha256': sha(D / 'final-review-provenance.private.json'),
        'cases': 30, 'outputs': len(decoded), 'explicit_instruction_cases': 10,
        'generic_or_empty_instruction_cases': 20,
        'models': counter,
        'interpretation': [
            'Compared with strongest original test baseline A, new seeds have2/10 direction-aligned judgments versus1/10; opposite judgments1-2 versus3. This is descriptive AI review, not statistical noninferiority proof.',
            'Excessive-strength judgments fall from16/30 for A to3/30,2/30,4/30 for seeds03,04,05. Some explicit substantial-enhancement requests instead become insufficient.',
            'Selected Seed04 has no major-unrelated-change judgments on10 explicit cases, versus5 for A; generic cases are unassessable for unrelatedness.',
            'B has3/10 aligned judgments, exceeding each new seed2/10; do not claim superiority over every baseline on all instruction types.',
            'Persistent failure cases include night visibility2, brown tone12, outlet removal14, backlit faces24 and soft portrait30. No output achieves local outlet removal or selective colorization23.',
            'Reference match is a qualitative judgment, separate from measured pixelL1; no ranking of general aesthetics is established.'
        ],
        'limits': [
            'AI-assisted anonymous viewing of full-frame1800px9-panel sheets, not independent human review or a formal double-blind user study.',
            'Model identities hidden per case until all210 notes frozen; reviewer knew aggregate campaign results and method hypotheses.',
            'Only10/30 cases have specific requests; generic/empty requests never count as directional passes.',
            'Fine sharpening details cannot be established at sheet scale; uncertainty and partial ratings retained.',
            'Uniform preselected30 source groups; no case substitution, no new test-driven model selection.'
        ]
    }
    assert len(decoded) == 210
    for target, payload in [(D / 'decoded-observations.json', decoded),
                            (D / 'visual-review-summary.json', summary),
                            (M / 'final-visual-review-summary.json', summary)]:
        assert not target.exists()
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'cases': 30, 'outputs': 210, 'models': counter}, ensure_ascii=False))

if __name__ == '__main__':
    main()
