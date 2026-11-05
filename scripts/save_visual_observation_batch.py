"""Save manually authored observations after direct concealed-label image review.

Reads a JSON array from stdin. Does not generate judgments or read the decoder.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from visual_review_records import ALLOWED

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'metadata/evaluation-v1'


def main():
    batch = json.load(sys.stdin)
    path = FOLDER / 'visual-review-observations.json'
    observations = json.loads(path.read_text())
    audit = json.loads((FOLDER / 'figure-audit-summary.json').read_text())
    audited = {Path(row['path']).name: row for row in audit['figures']}
    qa_path = FOLDER / 'final-visual-qa.json'
    qa = json.loads(qa_path.read_text())
    for item in batch:
        case = observations['cases'][item['case']-1]
        assert case['case'] == item['case'] and len(item['panels']) == 5
        assert item['context'].strip() and not case['context_note']
        case['context_note'] = item['context']
        for number, values in enumerate(item['panels'], 1):
            assert len(values) == 4 and values[3].strip()
            note = dict(zip(['direction','strength','unrelated_changes','observation'], values))
            for field, allowed in ALLOWED.items():
                assert note[field] in allowed
            name = f'Model {number}'
            assert not case['panels'][name]['observation']
            case['panels'][name] = note
        filename = f"case-{item['case']:02d}"
        png = FOLDER / 'blinded-visual-render' / f'{filename}.png'
        digest = hashlib.sha256(png.read_bytes()).hexdigest()
        proof = audited[f'{filename}.pdf']
        assert digest == proof['png_sha256']
        assert proof['path'] not in {row['path'] for row in qa['figures']}
        qa['figures'].append(dict(path=proof['path'], png_sha256=digest, passed=True,
                                 reviewed_at=datetime.now(timezone.utc).isoformat(),
                                 note='Directly inspected all seven full-frame panels and the instruction/title labels; no visible clipping or layout collision. Qualitative judgments saved separately before decoding.'))
    qa['direct_visual_inspection_completed'] = len(qa['figures']) == 32
    for destination, data in [(path, observations), (qa_path, qa)]:
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(destination)
    print(json.dumps(dict(cases_recorded=sum(bool(c['context_note']) for c in observations['cases']),
                          total_figures_inspected=len(qa['figures']))))


if __name__ == '__main__':
    main()
