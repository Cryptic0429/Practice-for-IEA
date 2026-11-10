"""Derive final AI assessments without replacing any raw visual observation."""
import argparse
import json
from pathlib import Path

from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v6 import visual_review as established
from .visual_review import configure, EVAL, EXP


def finalize(split, models, name):
    source = EXP / 'review' / split
    raw_path = source / 'codex-observations.jsonl'
    full_path = source / 'codex-full-resolution-observations.jsonl'
    raw = {r['item_id']: r for r in map(json.loads, raw_path.read_text().splitlines())}
    full = {r['item_id']: r for r in map(json.loads, full_path.read_text().splitlines())}
    merged = {k: dict(v) for k, v in raw.items()}
    for key, note in full.items():
        assert key in raw and note['output_sha256'] == raw[key]['image_sha256']
        assert note['input_sha256']
        for image in note['images']:
            assert sha(ROOT / image['path']) == image['sha256']
        manifest = ROOT / note['images'][0]['path']
        assert sha(manifest.parent / 'manifest.json') == note['manifest_sha256']
        merged[key].update({field: note[field] for field in ('completion', 'quality', 'evidence')})
        merged[key]['original_size_supplement'] = note
        merged[key]['raw_contact_sheet_assessment'] = raw[key]

    historical, history_sources = {}, {}
    # Prior final per-prediction reviews already include their native-pixel supplements.
    # Do not turn an old tentative contact-sheet U into an extra conflicting final vote.
    import re
    for path in (ROOT/'experiments').glob('functional-v*-*/evaluation/*/*/*/direct-review.json'):
        match=re.match(r'functional-v(\d+)-',path.relative_to(ROOT/'experiments').parts[0])
        if not match or not 1 <= int(match.group(1)) <= 15:continue
        if path.parent.parent.name not in (split,'natural-use' if split=='old-native' else split):continue
        history_sources[str(path.relative_to(ROOT))]=sha(path)
        for item in read(path):
            key=item.get('item_id');note=item.get('assessment',{})
            if not key or 'image_sha256' not in note:continue
            assert note['item_id']==key
            historical.setdefault(key,[]).append((path,note))
    disputes = []
    for key, fresh in list(merged.items()):
        old = historical.get(key, [])
        if not old:
            continue
        # The item key incorporates exact input hash, bilingual instructions and criteria.
        assert all(note['image_sha256'] == fresh['image_sha256'] for _, note in old)
        fields = [field for field in ('completion', 'quality') if any(note[field] != fresh[field] for _, note in old)]
        if not fields:
            continue
        dispute = dict(item_id=key, case_id=fresh['case_id'], current=fresh,
                       prior=[dict(file=str(p.relative_to(ROOT)), assessment=n) for p, n in old],
                       disagreed_fields=fields)
        disputes.append(dispute)
        revised = dict(fresh)
        revised.update({field: 'uncertain' for field in fields})
        revised.update(adjudication='historical_exact_render_disagreement',
                       evidence='Repeated direct AI observations of this identical input/request/result disagree on ' + ', '.join(fields) + '. All raw observations are preserved; disputed fields stay uncertain. No human adjudication.')
        merged[key] = revised

    destination = EXP / 'review-derivation' / name
    target = destination / 'review' / split
    assert not (target / 'codex-observations.jsonl').exists(), 'Preserve derived reviews; choose a new name'
    target.mkdir(parents=True)
    derived = target / 'codex-observations.jsonl'
    derived.write_text(''.join(json.dumps(n, ensure_ascii=False) + '\n' for n in merged.values()))
    audit = dict(at=stamp(), split=split, name=name, models=models,
                 raw_observations=dict(path=str(raw_path.relative_to(ROOT)), sha256=sha(raw_path), count=len(raw)),
                 original_size_observations=dict(path=str(full_path.relative_to(ROOT)), sha256=sha(full_path), count=len(full)),
                 historical_sources=history_sources, disagreements=disputes,
                 policy='Original-size direct AI notes supplement raw sheet uncertainty; exact input/request/output disagreements remain uncertain. No historical or current raw observation is overwritten. No L1, automatic visual rating or human approval.')
    write(target / 'derivation.json', audit)
    configure()
    established.EXP = destination
    summaries = established.summarize(split, models)
    for model_id, summary in summaries.items():
        folder = EVAL / split / model_id
        identity = read(folder / 'identity.json')
        rows = list(map(json.loads, (folder / 'predictions.jsonl').read_text().splitlines()))
        summary.update(declined_language_outputs=sum(r['status'] == 'declined' for r in rows),
                       declined_semantic_cases=len({r['case_id'] for r in rows if r['status'] == 'declined'}),
                       refusal_policy='Every decline on these feasible tasks is uncompleted; no result image is returned.',
                       original_size_supplement_count=len(full),
                       review_derivation=str((target / 'derivation.json').relative_to(ROOT)),
                       review_derivation_sha256=sha(target / 'derivation.json'),
                       raw_visual_notes_immutable=True,
                       inference_performed=identity.get('kind') not in ('CPU_frozen_parameter_editor_replay','exact_frozen_predictions_reused_no_new_inference'),
                       runtime='Exact prior predictions and pixels reused; no new model inference'
                       if identity.get('kind') in ('CPU_frozen_parameter_editor_replay','exact_frozen_predictions_reused_no_new_inference') else 'functional_v18.runtime')
        write(folder / 'direct-functional-summary.json', summary)
    print(json.dumps(dict(split=split, original_size_supplements=len(full), exact_render_disputes=len(disputes))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['native-development', 'old-native','development'])
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    finalize(args.split, args.models, args.name)
