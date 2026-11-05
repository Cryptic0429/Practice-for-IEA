"""Frozen-base visual screening; training remains gated on direct Codex review."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import textwrap
import traceback
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .common import ROOT, DATA, EXP, GPUJob, read, write, append, sha, stamp, EDITOR_VERSION
from .editor import load_rgb, edit, save, statistics


def font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sheets(selections, cases):
    lookup = {c['id']: c for c in cases}
    output = EXP / 'review/training'
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for offset in range(0, len(selections), 4):
        rows = selections[offset:offset+4]
        canvas = Image.new('RGB', (2000, 1500), 'white')
        draw = ImageDraw.Draw(canvas)
        entries = []
        for index, selection in enumerate(rows):
            case = lookup[selection['case_id']]
            x, y = (index % 2)*1000, (index // 2)*750
            chosen = selection.get('selected')
            title = case['id'] + ' | ' + case['action'] + (' | UNRESOLVED' if not chosen else '')
            draw.text((x+15, y+10), title, fill='black', font=font(21))
            for j, line in enumerate(textwrap.wrap(case['instructions']['en'], 95)):
                draw.text((x+15, y+45+24*j), line, fill='black', font=font(18))
            sources = [('BEFORE', ROOT/case['image'])]
            if chosen:
                sources.append(('AFTER', ROOT/chosen['image']))
            for j, (label, path) in enumerate(sources):
                draw.text((x+15+500*j, y+115), label, fill='black', font=font(18))
                with Image.open(path) as im:
                    fitted = ImageOps.contain(im.convert('RGB'), (480, 580), Image.Resampling.LANCZOS)
                canvas.paste(fitted, (x+10+500*j+(480-fitted.width)//2, y+150+(580-fitted.height)//2))
            entries.append({'case_id': case['id'], 'image_sha256': chosen['sha256'] if chosen else None})
        path = output / f'sheet-{offset//4+1:03d}.jpg'
        canvas.save(path, quality=95, subsampling=0)
        manifest.append({'sheet': str(path.relative_to(ROOT)), 'sha256': sha(path), 'cases': entries})
    write(output/'sheet-manifest.json', manifest)
    return manifest


def screen(gpu, limit):
    cases = read(DATA/'train-cases.json')
    candidates = {r['case_id']: r['candidates'] for r in read(DATA/'candidate-manifest.json')}
    out = EXP/'curation'
    out.mkdir(parents=True, exist_ok=True)
    results = out/'frozen-base-screening.jsonl'
    done = {r['case_id']: r for r in (json.loads(l) for l in results.read_text().splitlines())} if results.exists() else {}
    from .runtime import load_model, processor, predict, REVIEW_SYSTEM
    from .review_runtime import judge
    identity = {'case_manifest_sha256': sha(DATA/'case-manifest.json'),
                'candidate_manifest_sha256': sha(DATA/'candidate-manifest.json'),
                'editor_version': EDITOR_VERSION, 'reviewer': 'frozen original Qwen2.5-VL-7B-Instruct',
                'review_system_sha256': hashlib.sha256(REVIEW_SYSTEM.encode()).hexdigest(),
                'runtime_sha256': sha(__import__('pathlib').Path(__file__).with_name('runtime.py')),
                'review_runtime_sha256': sha(__import__('pathlib').Path(__file__).with_name('review_runtime.py')),
                'original_model_trainable': False, 'model_identity_hidden_from_reviewer': True,
                'references_and_parameters_hidden_from_reviewer': True,
                'human_review': False}
    if (out/'identity.json').exists():
        assert read(out/'identity.json') == identity, 'Screening identity changed; preserve old attempt and start a versioned review'
    else:
        write(out/'identity.json', identity)
    # Carry forward an observation only when the actual before/request and
    # after pixels, frozen reviewer and inspection prompt are all unchanged.
    # This reuses evidence, never changes an uncertain/failed judgment to a pass.
    cache = out/'draft-editor-v1.1'
    cached_reviews, cached_proposals = {}, {}
    if (cache/'identity.json').exists():
        previous = read(cache/'identity.json')
        old_manifest = read(cache/'case-manifest.json')
        compatible = (previous['runtime_sha256'] == identity['runtime_sha256']
                      and previous.get('review_runtime_sha256') == identity['review_runtime_sha256']
                      and previous['review_system_sha256'] == identity['review_system_sha256']
                      and old_manifest['splits']['train']['cases_sha256'] == sha(DATA/'train-cases.json'))
        if compatible:
            for line in (cache/'candidate-observations.jsonl').read_text().splitlines():
                entry = json.loads(line)
                cached_reviews[(entry['case_id'], entry['candidate']['sha256'])] = entry['assessment']
            for line in (cache/'frozen-base-screening.jsonl').read_text().splitlines():
                entry = json.loads(line)
                cached_proposals[entry['case_id']] = entry['base_proposal']
        write(out/'cache-policy.json', {'compatible': compatible,
            'source': str(cache.relative_to(ROOT)), 'identical_output_reviews': len(cached_reviews),
            'case_definitions_unchanged': old_manifest['splits']['train']['cases_sha256']==sha(DATA/'train-cases.json'),
            'policy': 'Exact output SHA plus unchanged case file, frozen model path, review prompt and review runtime. Changed outputs are reviewed again.'})
    with GPUJob(gpu, 'training-label-visual-screen', max_seconds=14400) as budget:
        import torch
        torch.set_num_threads(4)
        model = load_model()
        p = processor()
        count = 0
        for case in cases:
            if case['id'] in done:
                continue
            if limit and count >= limit:
                break
            budget.tick()
            status = {'status': 'running', 'pid': os.getpid(), 'case_id': case['id'],
                      'completed': len(done), 'total': len(cases), 'updated_at': stamp()}
            write(out/'status.json', status)
            proposals = list(candidates[case['id']])
            # A model proposal is recorded independently from the controlled candidates.
            proposal_record = {}
            try:
                prior = cached_proposals.get(case['id'])
                if prior and prior.get('valid'):
                    response, parameters = prior['response'], prior['parameters']
                else:
                    response, parameters = predict(model, p, case, 'en')
                path = DATA/'candidates-v1.2'/case['id']/'base-proposal.png'
                image = edit(load_rgb(ROOT/case['image']), parameters)
                save(path, image)
                proposal_record = {'response': response, 'parameters': parameters, 'valid': True,
                                   'reused_from_previous_frozen_proposal': bool(prior and prior.get('valid'))}
                proposals.append({'candidate_id': 'base-proposal', 'parameters': parameters,
                                  'image': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                  'origin': 'frozen_original_qwen_proposal'})
            except Exception as error:
                proposal_record = {'valid': False, 'error': f'{type(error).__name__}: {error}'}
            evaluations = []
            # Stable randomised order, with no candidate origin or parameter in the judge input.
            ordered = sorted(proposals, key=lambda c: hashlib.sha256((case['id']+c['candidate_id']).encode()).hexdigest())
            for candidate in ordered:
                budget.tick()
                assert sha(ROOT/candidate['image']) == candidate['sha256']
                try:
                    cached = cached_reviews.get((case['id'], candidate['sha256']))
                    if cached:
                        assessment = dict(cached, reused_from_identical_image_review=True)
                    else:
                        assessment = judge(model, p, case, ROOT/candidate['image'])
                except Exception as error:
                    assessment = {'completion': 'uncertain', 'quality': 'uncertain', 'strength': 'uncertain',
                                  'evidence': f'Visual review could not be parsed: {type(error).__name__}: {error}'}
                evaluations.append({'candidate': candidate, 'assessment': assessment})
                append(out/'candidate-observations.jsonl', {'case_id': case['id'], **evaluations[-1]})
            passed = [r for r in evaluations if r['assessment']['completion'] == 'completed'
                      and r['assessment']['quality'] == 'acceptable'
                      and r['assessment']['strength'] == 'appropriate']
            preferred = {'candidate-01': 0, 'candidate-00': 1, 'candidate-02': 2, 'base-proposal': 3}
            chosen = min(passed, key=lambda r: preferred.get(r['candidate']['candidate_id'], 9)) if passed else None
            record = {'case_id': case['id'], 'base_proposal': proposal_record,
                      'evaluations': evaluations, 'selected': chosen['candidate'] if chosen else None,
                      'status': 'awaiting_direct_codex_visual_review' if chosen else 'needs_label_revision',
                      'reviewed_at': stamp()}
            append(results, record)
            done[case['id']] = record
            count += 1
            if len(done) % 4 == 0:
                write(out/'selections.json', [done[c['id']] for c in cases if c['id'] in done])
        selections = [done[c['id']] for c in cases if c['id'] in done]
        write(out/'selections.json', selections)
        sheets(selections, cases)
        write(out/'status.json', {'status': 'awaiting_codex_review' if len(done) == len(cases) else 'partial_screen',
                                  'completed': len(done), 'total': len(cases),
                                  'selected': sum(bool(r['selected']) for r in selections),
                                  'needs_revision': sum(not r['selected'] for r in selections),
                                  'updated_at': stamp()})


def finalize():
    cases = read(DATA/'train-cases.json')
    policy = read(EXP/'review-policy.json') if (EXP/'review-policy.json').exists() else {'primary': 'two_stage_agreement'}
    source = EXP/'review/training/direct-proposals.json' if policy['primary']=='codex_direct_visual' else EXP/'curation/selections.json'
    selections = {r['case_id']: r for r in read(source)}
    review_file = EXP/'review/training/codex-observations.jsonl'
    reviews = {r['case_id']: r for r in (json.loads(l) for l in review_file.read_text().splitlines())}
    assert set(selections) == set(reviews) == {c['id'] for c in cases}
    sheet_manifest = read(EXP/'review/training/sheet-manifest.json')
    sheets_by_case = {c['case_id']: sheet for sheet in sheet_manifest for c in sheet['cases']}
    records = []
    import cv2
    import numpy as np
    for case in cases:
        selection, review = selections[case['id']], reviews[case['id']]
        target = selection['selected']
        assert target, case['id'] + ': no reviewable target label'
        assert review['completion'] == 'completed' and review['quality'] == 'acceptable', case['id']
        assert review['reviewer'] == 'Codex direct visual review; AI-assisted, not human'
        assert review['image_sha256'] == target['sha256'] == sha(ROOT/target['image'])
        assert sha(ROOT/case['image']) == case['input_sha256']
        rgb = load_rgb(ROOT/case['image'])
        if max(rgb.shape[:2]) > 1024:
            scale = 1024/max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (round(rgb.shape[1]*scale), round(rgb.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        assert np.array_equal(edit(rgb, target['parameters']), load_rgb(ROOT/target['image'])), case['id']+': target cannot be replayed'
        if case['action']=='identity':
            assert not any(target['parameters'].values()) and np.array_equal(rgb, load_rgb(ROOT/target['image']))
        sheet = sheets_by_case[case['id']]
        if review.get('sheet_evidence'):
            from .record_direct import case_digest
            assert review['case_spec_sha256'] == case_digest(case)
            assert review['sheet_sha256'] == sha(ROOT/review['sheet_evidence'])
        else:
            assert review['sheet_sha256'] == sheet['sha256'] == sha(ROOT/sheet['sheet'])
        assert review.get('evidence'), 'Visible evidence is mandatory'
        for language in ('zh', 'en'):
            records.append({'id': case['id']+'-'+language, 'case_id': case['id'], 'group_id': case['group_id'],
                            'family': case['family'], 'language': language, 'image': case['image'],
                            'instruction': case['instructions'][language], 'parameters': target['parameters'],
                            'target_image': target['image'], 'target_sha256': target['sha256'],
                            'instruction_origin': case['instruction_origin']})
    assert len(records) == 440
    write(DATA/'train-accepted.json', records)
    write(DATA/'label-acceptance.json', {'status': 'accepted_for_pilot_training', 'accepted_at': stamp(),
        'semantic_cases': 220, 'language_records': 440, 'source_groups': len({r['group_id'] for r in records}),
        'dataset_sha256': sha(DATA/'train-accepted.json'), 'case_manifest_sha256': sha(DATA/'case-manifest.json'),
        'primary_review_policy': policy,
        'selected_target_manifest_sha256': sha(source),
        'frozen_base_review_sha256': sha(EXP/'curation/frozen-base-screening.jsonl') if (EXP/'curation/frozen-base-screening.jsonl').exists() else None,
        'codex_review_sha256': sha(review_file), 'human_review': False,
        'instruction_origins': dict(__import__('collections').Counter(r['instruction_origin'] for r in records)),
        'limits': ['Controlled-task supervision rather than untouched natural GIER requests.',
                   'Codex direct AI visual verification; failed-calibration Qwen screening is advisory only, not a second certification.',
                   'Training contains photographs, old scans, illustrations and graphics; this is not a photography-only dataset.',
                   'Targets inspected at full-frame preview scale, maximum edge 1024; no fine-detail restoration capability is claimed.']})


def direct_proposals():
    """Prepare images for actual inspection; this does not accept any labels."""
    cases = read(DATA/'train-cases.json')
    candidate_rows = read(DATA/'candidate-manifest.json')
    candidates = {r['case_id']: r['candidates'] for r in candidate_rows}
    proposals = []
    for case in cases:
        rows = candidates[case['id']]
        chosen = next((r for r in rows if r['candidate_id']=='candidate-01'), rows[0])
        proposals.append({'case_id': case['id'], 'selected': chosen,
                          'status': 'unaccepted_candidate_awaiting_actual_codex_visual_inspection'})
    path = EXP/'review/training/direct-proposals.json'
    if path.exists():
        raise RuntimeError('Do not overwrite direct-review target choices; revise individual cases with a recorded observation')
    write(path, proposals)
    sheets(proposals, cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['screen', 'finalize', 'sheets', 'direct'])
    ap.add_argument('--gpu', choices=['0','2','3'], default='3')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.command == 'screen':
        screen(args.gpu, args.limit)
    elif args.command == 'finalize':
        finalize()
    elif args.command == 'direct':
        direct_proposals()
    else:
        sheets(read(EXP/'curation/selections.json'), read(DATA/'train-cases.json'))


if __name__ == '__main__':
    main()
