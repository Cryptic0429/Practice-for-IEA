"""CPU HTTP contract tests; no claims of actual model or browser integration."""
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from functional_v1 import web
from functional_v1.common import ROOT, write, stamp, sha
EXP = ROOT / 'experiments/functional-v18-20261107'


def main():
    image = io.BytesIO()
    Image.new('RGB', (32, 32), (70, 110, 150)).save(image, format='PNG')
    payload = image.getvalue()
    checks = []
    spec = dict(id='fixture-new', runtime_family='functional_v18')
    identity = dict(model_id='fixture-new', runtime_family='functional_v18', schema='functional-outcome-v1')
    client = web.app.test_client()

    def submit(instruction='请稍微提亮。'):
        return client.post('/api/edit', data=dict(model_id='fixture-new', instruction=instruction,
                                                image=(io.BytesIO(payload), 'input.png')), content_type='multipart/form-data')

    with tempfile.TemporaryDirectory(prefix='iea-v18-http-contract-') as temporary:
        with patch.object(web, 'REQUESTS', Path(temporary)), patch.object(web, 'resolve', return_value=spec):
            for reason in ('unreliable_quality', 'unsupported_tools'):
                value = dict(status='declined', reason_code=reason, decision=dict(action='decline', reason=reason),
                             output=None, output_sha256=None, parameters=None, identity=identity)
                with patch.object(web.requests, 'post') as post:
                    post.return_value.json.return_value = value
                    response = submit('请移除人物并补全背景。')
                    assert post.called, 'New-schema unsupported decisions must actually call the specified model'
                result = response.get_json()
                assert response.status_code == 200 and result['status'] == 'declined'
                assert result['after'] is None and result['parameters'] is None and result['identity'] == identity
                assert client.get(f'/files/{result["request_id"]}/output.png').status_code == 404
                record = json.loads((Path(temporary) / result['request_id'] / 'request.json').read_text())
                assert record['status'] == 'declined' and record['training_use'] is False
                checks.append('decline_no_image_' + reason)

            def keep_reply(*_, **kwargs):
                source = Path(kwargs['json']['image'])
                output = source.with_name('output.png')
                output.write_bytes(source.read_bytes())
                class Response:
                    def json(self):
                        return dict(status='executed', execution_status='unchanged', decision=dict(action='keep'),
                                    parameters={}, output=str(output), output_sha256=sha(output), identity=identity)
                return Response()

            with patch.object(web.requests, 'post', side_effect=keep_reply):
                response = submit('请保持原图不变。')
            result = response.get_json()
            assert response.status_code == 200 and result['execution_status'] == 'unchanged'
            assert result['message'] == '已按要求保持原图不变。'
            assert client.get(result['before']).data == client.get(result['after']).data
            checks.append('keep_explicit_unchanged_exact_pixels')

            def illegal_decline(*_, **kwargs):
                source = Path(kwargs['json']['image'])
                source.with_name('output.png').write_bytes(source.read_bytes())
                class Response:
                    def json(self):
                        return dict(status='declined', reason_code='unreliable_quality', decision=dict(action='decline', reason='unreliable_quality'),
                                    output=None, output_sha256=None, identity=identity)
                return Response()

            before = set(Path(temporary).iterdir())
            with patch.object(web.requests, 'post', side_effect=illegal_decline):
                response = submit()
            assert response.status_code == 400 and response.get_json()['status'] == 'failed'
            added = set(Path(temporary).iterdir()) - before
            assert len(added) == 1 and not (next(iter(added)) / 'output.png').exists()
            checks.append('decline_with_stale_image_rejected_and_removed')

            with patch.object(web.requests, 'post') as post:
                post.return_value.json.return_value = dict(status='failed', error='invalid_model_json')
                response = submit()
            assert response.status_code == 422 and response.get_json()['status'] == 'failed'
            checks.append('invalid_model_output_not_fallback_success')

        with patch.object(web, 'resolve', return_value=dict(id='fixture-old', runtime_family='functional_v1')):
            with patch.object(web.requests, 'post') as post:
                response = submit('请移除人物并补全背景。')
                assert response.status_code == 422 and response.get_json()['status'] == 'unsupported' and not post.called
        checks.append('legacy_unsupported_precheck_preserved')

    write(EXP / 'trial-http-contract-check.json', dict(at=stamp(), status='passed', checks=checks,
        actual_model_inference=False, browser_interaction=False, GPU_hours=0,
        code_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(web.__file__), Path(__file__)]},
        limitations='Mocked worker transport verifies HTTP/no-output boundary only. Actual trained-model, browser, switching, download and feedback checks remain required.'))
    print('Passed', len(checks), 'HTTP outcome-contract checks; mocked transport, no GPU.')


if __name__ == '__main__':
    main()
