"""CPU contract checks for explicit refusal, identity, strict parsing and input leakage."""
import copy
import json
import numpy as np

from functional_v1.common import ROOT, read, write, stamp, sha
from functional_v11.outcomes import parse, target_text, execute, input_row, SCHEMA_VERSION


def main():
    pixels = np.array([[[0, 0, 0], [60, 100, 140], [255, 255, 255]]], dtype=np.uint8)
    decisions = [
        {"action": "edit", "parameters": {"exposure": 20}},
        {"action": "keep"},
        {"action": "decline", "reason": "unreliable_quality"},
        {"action": "decline", "reason": "unsupported_tools"},
    ]
    for decision in decisions:
        assert parse(target_text(decision)) == decision
        result = execute(pixels, decision)
        assert result["schema"] == SCHEMA_VERSION
        if decision["action"] == "decline":
            assert result["status"] == "declined" and result["result"] is None and result["parameters"] is None
        elif decision["action"] == "keep":
            assert result["status"] == "unchanged" and np.array_equal(result["result"], pixels)
            assert result["result"] is not pixels
        else:
            assert result["status"] == "edited" and not np.array_equal(result["result"], pixels)

    invalid = [
        '{}', '{"exposure":20}', '[]', 'null',
        '{"action":"edit","parameters":{}}',
        '{"action":"edit","parameters":{"exposure":0}}',
        '{"action":"edit","parameters":{"exposure":true}}',
        '{"action":"edit","parameters":{"exposure":101}}',
        '{"action":"edit","parameters":{"exposure":NaN}}',
        '{"action":"edit","parameters":{"denoise":40}}',
        '{"action":"edit","parameters":{"sepia":-1}}',
        '{"action":"edit","parameters":{"contrast":1,"contrast":2}}',
        '{"action":"keep","parameters":{}}',
        '{"action":"keep","action":"edit"}',
        '{"action":"decline","reason":"unreliable_quality","parameters":{"exposure":50}}',
        '{"action":"decline","reason":"too_dark"}',
        '{"action":"decline","reason":["unreliable_quality"]}',
        '{"action":"decline"}',
        '{"action":null}',
    ]
    for payload in invalid:
        try:
            parse('```json\n' + payload + '\n```')
        except (ValueError, TypeError):
            continue
        raise AssertionError('Accepted an invalid decision: ' + payload)
    for payload in ['{"action":"keep"}', 'text\n```json\n{"action":"keep"}\n```',
                    '```json\n{"action":"keep"}\n```\n```json\n{"action":"keep"}\n```']:
        try:
            parse(payload)
        except ValueError:
            continue
        raise AssertionError('Accepted invalid wrapper')
    # A technically nonzero decision must not masquerade as an edited result
    # if quantization leaves every output pixel unchanged.
    try:
        execute(pixels, {"action": "edit", "parameters": {"exposure": .000001}})
    except ValueError as error:
        assert str(error) == 'edit_produced_no_pixel_change'
    else:
        raise AssertionError('No-op edit was accepted as success')

    cases = read(ROOT/'data/training/functional-v13-20261106/cases.json')
    for case in cases:
        poisoned = copy.deepcopy(case)
        for name in ('target', 'parameters', 'review', 'expected_change', 'must_preserve', 'canonical_photo'):
            poisoned[name] = 'FORBIDDEN_LABEL_OR_REVIEW_' + name
        for language in ('zh', 'en'):
            row = input_row(case, language)
            assert row == input_row(poisoned, language)
            assert 'FORBIDDEN_LABEL_OR_REVIEW_' not in json.dumps(row)
            assert row['conversations'][1]['content'] == '<image>\n' + case['instructions'][language]

    write(ROOT/'experiments/functional-v13-20261106/outcome-contract-check.json', dict(
        at=stamp(), status='passed', valid_roundtrips=4, invalid_decisions=len(invalid), invalid_wrappers=3,
        no_op_edit_rejected=True, decline_has_no_result=True, keep_is_exact_identity=True,
        input_whitelist_records=len(cases)*2,
        code_hashes={name:sha(ROOT/'scripts/functional_v11'/name) for name in ('outcomes.py','test_outcomes.py','editor.py')},
        scope='CPU contract only, not a trained refusal capability, web integration or functional acceptance',
        new_training_started=False, live_models_or_editor_changed=False))
    print('Explicit-outcome contract passed: no declined pixels, strict schema, exact keep and input-whitelist checks.')


if __name__ == '__main__':
    main()
