"""Bind direct anonymous notes to the fixed CPU replay; preserve AI disputes."""
import copy
import json
import os
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v6 import visual_review as established
from .runtime import EXP
from .outcomes import EDITOR_VERSION


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    protocol = read(EXP/'editor-only-protocol.json')
    manifest = read(EXP/'editor-only-review/manifest-readable.json')
    raw = [json.loads(l) for l in (EXP/'editor-control-observations.jsonl').read_text().splitlines()]
    assert len(raw) == 60 and len(manifest['items']) == 15
    notes = {(n['key'], n['label'], n['stage']): n for n in raw}
    assert len(notes) == 60
    cv = dict(C='completed', P='partial', F='failed', U='uncertain')
    qv = dict(A='acceptable', m='minor_issue', M='major_issue', U='uncertain')
    changed = {(i['split'], i['case_id'], lang): i for i in manifest['items'] for lang in i['language_rows']}
    destination = EXP/'evaluation'/EDITOR_VERSION
    assert not (EXP/'editor-control-summary.json').exists()
    summaries, disagreements = {}, []
    for split, source in protocol['sources'].items():
        replay = EXP/'editor-only'/split
        rows = [json.loads(l) for l in (replay/'predictions.jsonl').read_text().splitlines()]
        folder = destination/split/'parent-editor-only'
        folder.mkdir(parents=True, exist_ok=False)
        for name in ('predictions.jsonl', 'identity.json', 'status.json'):
            os.link(replay/name, folder/name)
        previous = (ROOT/source['predictions']).parent
        if split == 'decision-development':
            summary = copy.deepcopy(read(previous/'decision-summary.json'))
            assert all(not r['pixels_changed_by_editor'] for r in rows)
            summary.update(model_id='parent-editor-only', inference_performed=False,
                           exact_prior_summary_sha256=sha(previous/'decision-summary.json'))
            write(folder/'decision-summary.json', summary)
            summaries[split] = summary
            continue
        cases = {c['id']: c for c in read(ROOT/source['cases']['path'])}
        old_reviews = {(r['case_id'], r['language']): r for r in read(previous/'direct-review.json')}
        observations = {}
        for row in rows:
            if row['status'] != 'executed':
                continue
            prior = old_reviews[(row['case_id'], row['language'])]['assessment']
            assert prior['image_sha256'] == row['original_prediction']['output_sha256']
            if not row['pixels_changed_by_editor']:
                assert row['output_sha256'] == prior['image_sha256']
                note = copy.deepcopy(prior)
                note['reuse_source'] = dict(path=str((previous/'direct-review.json').relative_to(ROOT)),
                                           sha256=sha(previous/'direct-review.json'))
            else:
                item = changed[(split, row['case_id'], row['language'])]
                new_label = next(k for k,v in item['anonymous_mapping'].items() if v == 'NEW')
                old_label = next(k for k,v in item['anonymous_mapping'].items() if v == 'OLD')
                native = notes[(item['key'], new_label, 'native')]
                fresh_old = notes[(item['key'], old_label, 'native')]
                assert row['output_sha256'] == item['new_output_sha256']
                for n in (native, fresh_old):
                    assert sha(ROOT/n['sheet']) == n['sheet_sha256']
                    for a in n['assets']: assert sha(ROOT/a['path']) == a['sha256']
                note = dict(item_id=established.key(cases[row['case_id']], row),
                    case_id=row['case_id'], completion=cv[native['completion']], quality=qv[native['quality']],
                    image_sha256=row['output_sha256'], sheet=native['sheet'], sheet_sha256=native['sheet_sha256'],
                    evidence=native['evidence'], original_size_supplement=native,
                    reviewer='Codex direct anonymous AI review; not human/user approval')
                fields = [f for f, table in [('completion',cv), ('quality',qv)]
                          if table[fresh_old[f]] != prior[f]]
                if fields:
                    uncertain = [f for f, table in [('completion',cv), ('quality',qv)]
                                 if f in fields and table[native[f]] == table[fresh_old[f]]]
                    disagreements.append(dict(split=split, case_id=row['case_id'], language=row['language'],
                        prior=prior, fresh_old=fresh_old, fresh_new=native, disagreed_fields=fields,
                        new_fields_kept_uncertain=uncertain,
                        policy='Identical old output received conflicting AI grades. If new and old direct grades agree on that field, do not count it as a confident editor gain.'))
                    for field in uncertain: note[field] = 'uncertain'
                    note['same_old_render_dispute'] = dict(fields=fields, prior=prior, fresh_old=fresh_old)
            observations[note['item_id']] = note
        path = EXP/'review'/split/'codex-observations.jsonl'
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(n,ensure_ascii=False)+'\n' for n in observations.values()))
        established.EXP = EXP
        established.EVAL = destination
        established.case_file = lambda _, s=source: ROOT/s['cases']['path']
        summary = established.summarize(split, ['parent-editor-only'])['parent-editor-only']
        summary.update(inference_performed=False, GPU_hours=0, model_forward_calls=0,
            editor=EDITOR_VERSION, runtime='Exact saved parent decisions; CPU editor replay only',
            declined_language_outputs=sum(r['status']=='declined' for r in rows),
            declined_semantic_cases=len({r['case_id'] for r in rows if r['status']=='declined'}),
            unchanged_reviews_inherited=True, AI_disputes_preserved=True)
        write(folder/'direct-functional-summary.json', summary)
        summaries[split] = summary
    write(EXP/'editor-control-summary.json', dict(at=stamp(), status='completed', summaries=summaries,
        model_forward_calls=0, GPU_hours=0, changed_language_outputs=24, unique_changed_comparisons=15,
        raw_notes_sha256=sha(EXP/'editor-control-observations.jsonl'),
        disagreements=disagreements, original_parent_summaries_preserved=True,
        conclusion='Editor semantics correction is not model learning. Exact-old-render AI disagreements stay explicit; no L1 or automatic image grades. Development is repeatedly used, not an independent test.'))


if __name__ == '__main__': main()
