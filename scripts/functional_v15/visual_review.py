"""Use the established anonymous visual review, plus explicit decision checks."""
import argparse
import json
from pathlib import Path

import numpy as np
from functional_v1.common import ROOT, read, write, sha, stamp
from functional_v1.editor import load_rgb
from functional_v6 import visual_review as established
from .runtime import EXP
from .evaluate import KEYS
from .outcomes import EDITOR_VERSION


EVAL = EXP / 'evaluation' / EDITOR_VERSION


def case_file(split):
    item = read(EXP / 'evaluation-protocol.json')['case_files'][KEYS[split]]
    path = ROOT / item['path']
    assert sha(path) == item['sha256']
    return path


def configure():
    established.EXP = EXP
    established.EVAL = EVAL
    established.case_file = case_file


def summarize(split, models):
    configure()
    summaries = established.summarize(split, models)
    for model_id, summary in summaries.items():
        folder = EVAL / split / model_id
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        summary.update(declined_language_outputs=sum(r['status'] == 'declined' for r in rows),
                       declined_semantic_cases=len({r['case_id'] for r in rows if r['status'] == 'declined'}),
                       refusal_policy='Every refusal on these feasible-task evaluation sets remains uncompleted; no result image is displayed for refusal.',
                       runtime='functional_v15.runtime', editor=EDITOR_VERSION)
        write(folder / 'direct-functional-summary.json', summary)
    return summaries


def decisions(models):
    split = 'decision-development'
    cases = read(case_file(split))
    assert len(cases) == 20
    for model_id in models:
        folder = EVAL / split / model_id
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        lookup = {(r['case_id'], r['language']): r for r in rows}
        assert len(rows) == len(lookup) == 40
        records = []
        for c in cases:
            passed = []
            for language in ('zh', 'en'):
                r = lookup[(c['id'], language)]
                decision = r.get('decision', {})
                valid = decision.get('action') == c['expected_action']
                if c['expected_action'] == 'decline':
                    valid = valid and r['status'] == 'declined' and decision.get('reason') == c['expected_reason']
                    valid = valid and r.get('output_image') is None and r.get('output_sha256') is None
                    valid = valid and not (folder / 'images' / f'{c["id"]}-{language}.png').exists()
                else:
                    valid = valid and r['status'] == 'executed' and r.get('execution_status') == 'unchanged'
                    if valid:
                        valid = np.array_equal(load_rgb(ROOT / c['image']), load_rgb(ROOT / r['output_image']))
                passed.append(bool(valid))
                records.append(dict(case_id=c['id'], language=language, expected_action=c['expected_action'],
                                    observed_action=decision.get('action'), passed=bool(valid)))
            assert len(passed) == 2
        qualified = sum(all(r['passed'] for r in records if r['case_id'] == c['id']) for c in cases)
        summary = dict(at=stamp(), model_id=model_id, status='completed', semantic_cases=20,
                       original_photo_groups=10, passed_both_languages=qualified, language_records=records,
                       assessment='Deterministic decision-schema and exact-pixel/no-output contract checks; not a visual editing score or human judgment.',
                       cases_sha256=sha(case_file(split)))
        write(folder / 'decision-summary.json', summary)
        print(model_id, 'decision controls:', qualified, '/20')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'record', 'summarize', 'decisions'])
    parser.add_argument('--split', choices=list(KEYS), default='native-development')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--name')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--observations', type=Path)
    args = parser.parse_args()
    configure()
    if args.command == 'prepare':
        assert args.split != 'decision-development'
        established.prepare(args.split, args.models, args.name)
    elif args.command == 'record':
        established.record(args.bundle, args.observations)
    elif args.command == 'summarize':
        summarize(args.split, args.models)
    else:
        decisions(args.models)


if __name__ == '__main__':
    main()
