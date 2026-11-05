"""Explicit revisions after visible failures; this never accepts its own outputs."""
import argparse
from copy import deepcopy
import cv2
from .common import ROOT, DATA, EXP, read, write, append, sha, stamp
from .editor import load_rgb, edit, save, statistics
from .prepare import fingerprint, near
from .curate import sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('changes')
    args = ap.parse_args()
    changes = read(args.changes)
    cases = read(DATA/'train-cases.json')
    lookup = {c['id']:c for c in cases}
    proposals = read(EXP/'review/training/direct-proposals.json')
    by_id = {r['case_id']:r for r in proposals}
    journal = EXP/'review/training/target-revisions.jsonl'
    if (DATA/'train-accepted.json').exists():
        raise RuntimeError('Training dataset already accepted; create a new data version')
    for change in changes:
        cid = change['case_id']
        case = lookup[cid]
        before = {'case':deepcopy(case), 'selection':deepcopy(by_id[cid])}
        if change.get('replacement_record_id'):
            source = next(r for r in read(ROOT/'data/training/quality-v2-20261103/train3k.json')
                          if r['id']==change['replacement_record_id'])
            assert source['extra_info']['instruction'].strip()
            other = [c for split in ('train','development','acceptance')
                     for c in read(DATA/f'{split}-cases.json') if c['id']!=cid]
            assert source['group_id'] not in {c['group_id'] for c in other}
            rgb = load_rgb(source['images'][0])
            fp = fingerprint(cv2.resize(rgb,(256,256)))
            checked = {}
            for existing in other:
                if existing['group_id'] not in checked:
                    checked[existing['group_id']] = fingerprint(cv2.resize(load_rgb(ROOT/existing['image']),(256,256)))
            assert not any(near(fp, f) for f in checked.values()), 'replacement is a near duplicate'
            dest = DATA/'inputs'/f"{source['group_id'][:20]}.png"
            save(dest, rgb)
            case.update(group_id=source['group_id'], image=str(dest.relative_to(ROOT)), input_sha256=sha(dest),
                        original_source_image=source['images'][0], original_source_sha256=sha(source['images'][0]),
                        original_record_id=source['id'], original_request=source['extra_info']['instruction'])
        rgb = load_rgb(ROOT/case['image'])
        if max(rgb.shape[:2])>1024:
            scale = 1024/max(rgb.shape[:2])
            rgb = cv2.resize(rgb,(round(rgb.shape[1]*scale),round(rgb.shape[0]*scale)),interpolation=cv2.INTER_AREA)
        result = edit(rgb, change['parameters'])
        index = sum(__import__('json').loads(l)['case_id']==cid for l in journal.read_text().splitlines())+1 if journal.exists() else 1
        dest = DATA/'candidates-v1.2'/cid/f'revision-{index:02d}.png'
        save(dest, result)
        by_id[cid]['selected'] = {'candidate_id':dest.stem, 'parameters':change['parameters'],
            'image':str(dest.relative_to(ROOT)), 'sha256':sha(dest), 'statistics':statistics(result),
            'visual_preview_max_edge':1024, 'origin':'explicit_revision_after_codex_visible_failure'}
        by_id[cid]['status'] = 'unaccepted_revision_requires_actual_visual_inspection'
        append(journal, {'case_id':cid, 'at':stamp(), 'reason':change['reason'], 'before':before,
                         'after':{'case':case, 'selection':by_id[cid]}})
    write(DATA/'train-cases.json', cases)
    manifest = read(DATA/'case-manifest.json')
    manifest['splits']['train']['cases_sha256'] = sha(DATA/'train-cases.json')
    manifest['train_label_curation_revisions'] = {'journal':str(journal.relative_to(ROOT)), 'sha256':sha(journal)}
    write(DATA/'case-manifest.json', manifest)
    write(EXP/'review/training/direct-proposals.json', proposals)
    sheets(proposals, cases)
    assert len({c['group_id'] for c in cases})==160


if __name__ == '__main__':
    main()
