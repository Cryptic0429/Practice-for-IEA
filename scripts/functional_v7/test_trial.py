"""Actual local model HTTP parity, no-result decisions and separate agent feedback."""
import hashlib
import json

import requests
from functional_v1.common import ROOT, read, write, stamp, sha
from .visual_review import EXP, EVAL, case_file


def main():
    url = 'http://127.0.0.1:12346'
    catalog = requests.get(url + '/api/models', timeout=15).json()
    assert len(catalog['models']) == 15 and catalog['default'] == 'functional-current-20261104'
    selection = read(EXP / 'development-selection.json')
    cases = {split: {c['id']: c for c in read(case_file(split))}
             for split in ('native-development', 'old-native', 'decision-development')}
    training = read(ROOT / 'data/training/functional-v7-20261105/cases.json')
    records = []
    for model_id, meta in selection['models'].items():
        evaluation_id = meta['evaluation_id']
        offline = {split: {(r['case_id'], r['language']): r for r in map(json.loads,
                   (EVAL / split / evaluation_id / 'predictions.jsonl').read_text().splitlines())}
                   for split in cases}
        controls = list(cases['decision-development'].values())
        decline = next(c for c in controls if c['expected_action'] == 'decline')
        keep = next(c for c in controls if c['expected_action'] == 'keep')
        tasks = [('native-development', cases['native-development'][cid], language)
                 for cid in ('native-dev-006', 'native-dev-001') for language in ('zh', 'en')]
        tasks += [('decision-development', c, language) for c in (decline, keep) for language in ('zh', 'en')]
        invalid = [(split, r) for split in ('native-development', 'old-native') for r in offline[split].values()
                   if r['status'] == 'failed']
        if invalid:
            split, r = invalid[0]
            tasks.append((split, cases[split][r['case_id']], r['language']))
        quality = next(c for c in training if c['id'] == 'native-train-n127')
        tasks.append(('training-quality-decline-check-not-generalization', quality, 'zh'))
        for split, case, language in tasks:
            with (ROOT / case['image']).open('rb') as handle:
                response = requests.post(url + '/api/edit', data={'instruction': case['instructions'][language], 'model_id': model_id},
                                         files={'image': ('source.png', handle, 'image/png')}, timeout=240)
            actual = response.json()
            assert actual.get('request_id'), actual
            request_file = ROOT / 'experiments/functional-v1-20261104/demo/requests' / actual['request_id'] / 'request.json'
            saved = read(request_file)
            saved['source'] = 'agent_round7_HTTP_test_not_user_feedback'
            write(request_file, saved)
            quality_check = split.startswith('training-')
            expected = None if quality_check else offline[split][case['id'], language]
            if expected:
                assert actual['status'] == expected['status'], (actual, expected)
            if actual['status'] == 'executed':
                assert response.status_code == 200
                output = requests.get(url + actual['after'], timeout=30)
                output.raise_for_status()
                digest = hashlib.sha256(output.content).hexdigest()
                if expected:
                    assert actual['parameters'] == expected['parameters']
                    assert digest == expected['output_sha256']
                    assert actual['decision'] == expected['decision']
                assert actual['identity']['adapter_sha256'] == meta['adapter_sha256']
                assert actual['identity']['editor'] == meta['editor']
                result = dict(status='executed', output_sha256=digest, execution_status=actual['execution_status'],
                              offline_byte_identical=expected is not None,
                              quality_decline_obeyed=False if quality_check else None)
            elif actual['status'] == 'declined':
                assert response.status_code == 200 and actual['after'] is None and actual['parameters'] is None
                assert not request_file.with_name('output.png').exists()
                assert requests.get(url + f'/files/{actual["request_id"]}/output.png', timeout=15).status_code == 404
                if expected:
                    assert actual['decision'] == expected['decision']
                assert actual['identity']['adapter_sha256'] == meta['adapter_sha256']
                result = dict(status='declined', no_image_file=True, image_route_status=404,
                              reason_code=actual['reason_code'],
                              quality_decline_obeyed=actual['reason_code'] == 'unreliable_quality' if quality_check else None)
            else:
                assert response.status_code == 422 and expected and actual['detail'] == expected['error']
                assert not request_file.with_name('output.png').exists()
                assert requests.get(url + f'/files/{actual["request_id"]}/output.png', timeout=15).status_code == 404
                result = dict(status='failed', offline_error_identical=True, image_route_status=404)
            assert saved['training_use'] is False
            records.append(dict(model_id=model_id, split=split, case_id=case['id'], language=language,
                                request_id=actual['request_id'], **result))
            write(EXP / 'demo/http-integration-progress.json', dict(at=stamp(), checks=records, user_feedback=False))
        feedback = requests.post(url + '/api/feedback', json={'request_id': actual['request_id'], 'verdict': 'uncertain',
              'comment': 'Agent integration check only; training photo, not independent user usability evidence.'}, timeout=20)
        feedback.raise_for_status()
    requests.post('http://127.0.0.1:18282/unload', timeout=45).raise_for_status()
    assert requests.get(url + '/v7/evaluation-image/native-acceptance/language-e4/native-accept-001/zh/before', timeout=15).status_code == 404
    write(EXP / 'demo/http-integration.json', dict(at=stamp(), status='passed_integration_checks', checks=records,
          models_registered=15, default_unchanged=True, no_external_rewriting=True, agent_feedback_only=True,
          quality_checks_are_training_examples_not_generalization=True,
          quality_decline_checks_passed=sum(r.get('quality_decline_obeyed') is True for r in records),
          acceptance_routes_closed=True, worker_unloaded=True))
    print(json.dumps(dict(actual_requests=len(records), offline_parity=True,
                          quality_training_checks=[r for r in records if r['split'].startswith('training-')],
                          default_unchanged=True, independent_user_feedback=False)))


if __name__ == '__main__':
    main()
