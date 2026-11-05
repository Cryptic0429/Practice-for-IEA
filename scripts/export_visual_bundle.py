"""Package preselected held-out images without exposing model identity to the reviewer.

Run only after all five complete test runs are fixed. The decoder stays on the
server until the blinded observations have been saved. Raw image bytes are copied;
display-only letterboxing is performed by the separate renderer.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/sdc/zhc/IEA')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--engineering-preview', action='store_true')
    args = cli.parse_args()
    protocol_path = ROOT / 'configs/evaluation-protocol-v1.json'
    p = read(protocol_path)
    if args.engineering_preview:
        # This is an explicitly labelled training-only layout check, not a test result.
        models = ['Base', 'A', 'Official', 'BaseSchema']
        jobs = {m: ROOT / f'outputs/evaluation-v1/train-{m}-engineering-v1' for m in models}
        jobs['Official'] = ROOT / 'outputs/evaluation-v1/train-Official-engineering-v2'
        first = json.loads((jobs['A'] / 'predictions.jsonl').read_text().splitlines()[0])
        ids = [first['id']]
        keys = [dict(id=ids[0], panels={m: m for m in models})]
        out = ROOT / 'outputs/evaluation-v1/engineering-visual-preview'
        expected = 8
    else:
        ids_path = Path(p['visual_ids_path'])
        key_path = ids_path.with_name('visual-blinding-key.json')
        assert sha(ids_path) == p['visual_ids_sha256']
        assert sha(key_path) == p['visual_blinding_key_sha256']
        ids, keys = read(ids_path), read(key_path)
        assert len(ids) == len(keys) == 30 and [k['id'] for k in keys] == ids
        models = list(p['models'])
        jobs = {m: ROOT / f'outputs/evaluation-v1/test-{m}-full' for m in models}
        assert read(ROOT / 'configs/evaluation-models-v1.json')['all_selected_models_verified']
        out = ROOT / 'outputs/evaluation-v1/blinded-visual-bundle'
        expected = 200
    datasets = {}
    split = 'train' if args.engineering_preview else 'test'
    split_path = Path(p['split_directory']) / f'{split}.json'
    split_manifest = read(split_path.with_name('manifest.json'))
    assert sha(split_path) == split_manifest['files'][split]['sha256']
    source_rows = {row['id']: row for row in read(split_path)}
    for model, folder in jobs.items():
        state = read(folder / 'status.json')
        assert state['status'] == 'completed' and state['completed'] == expected, model
        records = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text().splitlines()]
        assert len(records) == len({r['id'] for r in records}) == expected
        if not args.engineering_preview:
            assert read(folder / 'run.json')['identity']['protocol_sha256'] == sha(protocol_path)
        datasets[model] = {r['id']: r for r in records}
    assert not out.exists(), f'Preserve existing bundle: {out}'
    out.mkdir()
    cases, private_provenance = [], []
    for index, key in enumerate(keys, 1):
        row = datasets['A'][key['id']]
        case_dir = out / f'case-{index:02d}'
        case_dir.mkdir()
        paths = [('Original', Path(row['input_image'])), ('Reference', Path(row['reference_image']))]
        paths.extend((panel, Path(datasets[model][key['id']]['scoring']['edited_image_path']))
                     for panel, model in key['panels'].items())
        images = []
        for image_index, (label, source) in enumerate(paths):
            dest = case_dir / f'panel-{image_index:02d}{source.suffix.lower()}'
            shutil.copy2(source, dest)
            assert sha(dest) == sha(source)
            images.append(dict(label=label, path=str(dest.relative_to(out)), sha256=sha(dest)))
            private_provenance.append(dict(case=index, panel=label, source=str(source), sha256=sha(source)))
        user_messages = [message['content'] for message in source_rows[key['id']]['conversations']
                         if message['role'] == 'user']
        assert len(user_messages) == 1
        displayed_instruction = user_messages[0].replace('<image>', '').strip()
        assert displayed_instruction
        cases.append(dict(case=index, id=key['id'], instruction=displayed_instruction,
                          metadata_instruction=row['instruction'],
                          instruction_source='Actual frozen user message, with the image placeholder removed only for display.',
                          images=images))
    public = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  role='training_layout_preview' if args.engineering_preview else 'blinded_test_visual_review',
                  protocol_sha256=sha(protocol_path), case_count=len(cases), cases=cases,
                  image_policy='Byte-identical source copies; full-frame display with identical letterboxing and no contrast, colour or local adjustments.',
                  review_scope='AI-assisted qualitative inspection, not a blinded human user study. Model identities are concealed in the displayed panels, but visual style may reveal them.')
    (out / 'manifest.json').write_text(json.dumps(public, indent=2, ensure_ascii=False) + '\n')
    # Private audit material must not be displayed during the blinded review.
    (out.parent / (out.name + '-private-provenance.json')).write_text(json.dumps(private_provenance, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), case_count=len(cases), role=public['role'])))


if __name__ == '__main__':
    main()
