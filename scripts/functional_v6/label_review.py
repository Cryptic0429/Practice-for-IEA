"""Model-hidden training-proposal panels and explicit direct-visual observations."""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from functional_v1.common import ROOT, read, write, append, sha, stamp, validate
from functional_v1.curate import font
from .source_bank import EXP, DATA


def prepare():
    out = EXP/'native-label-review'
    assert not (out/'manifest.json').exists(), 'Keep first review panels and order immutable'
    cases = read(DATA/'native-training-candidates.json')
    routes = {}
    for route in ('base', 'parent'):
        path = EXP/'native-proposals'/route/'predictions.jsonl'
        rows = list(map(json.loads, path.read_text().splitlines()))
        assert len(rows) == len({r['case_id'] for r in rows}) == 100
        assert read(path.with_name('status.json'))['status'] == 'completed'
        routes[route] = {r['case_id']: r for r in rows}
    private, panels = {}, []
    for offset in range(0, len(cases), 3):
        subset = cases[offset:offset+3]
        canvas = Image.new('RGB', (1800, 2220), 'white'); d = ImageDraw.Draw(canvas)
        d.text((12, 10), 'Native TRAINING label candidates | BEFORE + anonymous proposals; no auto-admission', font=font(25), fill='black')
        entries = []
        for i, c in enumerate(subset):
            y = 55+i*715
            d.text((12, y), c['id']+' | '+c['domain'], font=font(23), fill='black')
            for j, line in enumerate(textwrap.wrap(c['instructions']['en'], 123)):
                d.text((12, y+32+j*24), line, font=font(21), fill='black')
            order = ['base', 'parent']
            if int(hashlib.sha256(('v6-label-review'+c['id']).encode()).hexdigest(), 16)%2: order.reverse()
            private[c['id']] = {code: routes[route][c['id']] for code, route in zip(('A', 'B'), order)}
            before = dict(status='executed', output_image=c['image'], output_sha256=c['input_sha256'])
            for j, (label, row) in enumerate([('BEFORE', before)]+list(private[c['id']].items())):
                x = j*600
                d.text((x+12, y+90), label, font=font(23), fill='black')
                if row['status'] != 'executed':
                    d.text((x+15, y+200), 'FAILED / NO EDIT RESULT', font=font(23), fill='red')
                    continue
                path = ROOT/row['output_image']; assert sha(path) == row['output_sha256']
                with Image.open(path) as im:
                    fitted = ImageOps.contain(im.convert('RGB'), (575, 570), Image.Resampling.LANCZOS)
                canvas.paste(fitted, (x+12+(575-fitted.width)//2, y+125+(570-fitted.height)//2))
            entries.append(dict(case_id=c['id'], input_sha256=c['input_sha256'],
                output_sha256={code: r.get('output_sha256') for code, r in private[c['id']].items()},
                instructions=c['instructions']))
        path = out/f'sheet-{offset//3+1:03d}.jpg'; path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=97, subsampling=0)
        panels.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), cases=entries))
    write(out/'private-candidate-key.json', private)
    write(out/'manifest.json', dict(at=stamp(), sheets=panels, cases_sha256=sha(DATA/'native-training-candidates.json'),
        candidate_key_sha256=sha(out/'private-candidate-key.json'), reviewer='Codex direct visual review pending, AI assisted',
        model_identity_hidden=True, parameters_hidden=True, reference_images_used=False,
        display='Whole frames, identical fitting, no colour adjustment, no crop. Source English displayed; bilingual exact translations in case entries. Inspect full saved outputs separately whenever fine detail/quality is uncertain.',
        rule='Do not choose minimum L1 or a route by name. Record candidate-level completion/quality/evidence and only select a visibly completed acceptable result. Neither acceptable means unresolved, never substitute original as successful edit.'))
    print('Anonymous training-proposal panels prepared', len(panels))


def record(path):
    out = EXP/'native-label-review'; manifest = read(out/'manifest.json'); private = read(out/'private-candidate-key.json')
    assert sha(out/'private-candidate-key.json') == manifest['candidate_key_sha256']
    assert sha(DATA/'native-training-candidates.json') == manifest['cases_sha256']
    cases = {c['id']: c for c in read(DATA/'native-training-candidates.json')}
    sheets = {c['case_id']: s for s in manifest['sheets'] for c in s['cases']}
    dest = out/'observations.jsonl'
    done = {r['case_id'] for r in map(json.loads, dest.read_text().splitlines())} if dest.exists() else set()
    for note in read(path):
        cid = note['case_id']; assert cid not in done
        assert note['language_equivalence_checked'] is True
        assert set(note['candidates']) == {'A', 'B'}
        for code, assessment in note['candidates'].items():
            assert assessment['completion'] in ('completed', 'partial', 'failed', 'uncertain')
            assert assessment['quality'] in ('acceptable', 'minor_issue', 'major_issue', 'uncertain')
            assert len(assessment['evidence']) >= 15
            row = private[cid][code]
            if row['status'] == 'executed': assert sha(ROOT/row['output_image']) == row['output_sha256']
            else: assert assessment['completion'] == 'failed'
        selected = note.get('selected_code')
        target = None
        if selected is not None:
            assessment = note['candidates'][selected]
            assert assessment['completion'] == 'completed' and assessment['quality'] == 'acceptable'
            row = private[cid][selected]; assert row['status'] == 'executed'
            target = dict(parameters=validate(row['parameters']), image=row['output_image'], sha256=row['output_sha256'],
                          proposal_route=row['route'], role='directly_reviewed_target_pending_full_dataset_admission')
        sheet = sheets[cid]; assert sha(ROOT/sheet['path']) == sheet['sha256']
        append(dest, dict(note, selected=target, at=stamp(), input_sha256=cases[cid]['input_sha256'],
            sheet=sheet['path'], sheet_sha256=sheet['sha256'], reviewer='Codex direct visual review, AI assisted; not human/user approval'))
        done.add(cid)
    print('Explicit observations recorded; unresolved targets remain outside training')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['prepare', 'record']); ap.add_argument('--notes', type=Path)
    args = ap.parse_args()
    if args.command == 'prepare': prepare()
    else: record(args.notes)
