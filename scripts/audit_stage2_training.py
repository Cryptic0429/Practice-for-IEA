"""Audit completed GRPO exposure, reward arithmetic and numeric logs on CPU."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/mnt/sdc/zhc/IEA')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--formal', action='store_true')
    args = parser.parse_args()
    out = ROOT/'outputs/stage2-A-edit-grpo-v1'/args.run_name
    steps = 128 if args.formal else 8
    source = ROOT/'data/stage2/stage2-A-edit-grpo-v1'/('train.json' if args.formal else 'preflight.json')
    rows = json.loads(source.read_text())
    records = [json.loads(line) for line in (out/'candidates.jsonl').read_text().splitlines()]
    metrics = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    verification = json.loads((out/'training-verification.json').read_text())
    assert verification['step'] == steps
    assert len(records) == steps*16 == len(rows)*4
    assert Counter(r['id'] for r in records) == Counter({r['id']: 4 for r in rows})
    assert Counter(r['step'] for r in records) == Counter({s: 16 for s in range(1, steps+1)})
    assert {r['step'] for r in metrics if 'grad_norm' in r} == set(range(1, steps+1))
    assert all(math.isfinite(v) for r in metrics for v in r.values() if isinstance(v, (int, float)))
    groups = []
    errors = Counter()
    for start in range(0, len(records), 4):
        group = records[start:start+4]
        assert len({r['id'] for r in group}) == 1
        assert len({r['step'] for r in group}) == 1
        assert [r['candidate'] for r in group] == [0, 1, 2, 3]
        groups.append(dict(id=group[0]['id'], step=group[0]['step'],
                           distinct_outputs=len({r['response'] for r in group}),
                           distinct_rewards=len({r['reward'] for r in group})))
    assert groups == verification['groups']
    for r in records:
        s, p = r['scoring'], r['parsing']
        assert s['input_readable'] and s['reference_readable'] and not s['metric_error']
        d0, d = s['baseline_metrics']['pixel_L1'], s['edited_metrics']['pixel_L1']
        assert math.isfinite(d0) and math.isfinite(d)
        expected = max(-1., min(1., (d0-d)/max(d0, .01))) if p['valid'] and s['editor_success'] else -2.
        assert r['reward'] == expected
        assert len(r['completion_ids']) == r['output_tokens'] <= 512
        if not p['valid'] or not s['editor_success']:
            errors[str(p.get('error') or s.get('editor_error'))] += 1
    report = dict(passed=True, steps=steps, prompts=len(rows), candidates=len(records),
        informative_groups=sum(g['distinct_rewards'] > 1 for g in groups),
        diverse_output_groups=sum(g['distinct_outputs'] > 1 for g in groups),
        valid_candidates=sum(r['parsing']['valid'] and r['scoring']['editor_success'] for r in records),
        empty_parameters=sum(r['empty_parameters'] for r in records),
        all_zero_parameters=sum(r['all_zero_parameters'] for r in records),
        reward_saturated=sum(r['reward_saturated'] for r in records),
        length_limit_reached=sum(r['output_tokens'] >= 512 for r in records),
        truncated=sum(r['truncated'] for r in records),
        reward_mean=sum(r['reward'] for r in records)/len(records),
        invalid_reasons=dict(errors),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        candidates_sha256=hashlib.sha256((out/'candidates.jsonl').read_bytes()).hexdigest(),
        scope='Full recorded exposure/grouping, reward arithmetic and finite logged metrics; no image-quality improvement claim.')
    target = out/'candidate-audit.json'
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
