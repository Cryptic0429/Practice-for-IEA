import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('assets', Path(__file__).with_name('assets.py'))
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)

ROOT = Path(__file__).resolve().parents[2]


class VerificationTests(unittest.TestCase):
    def test_verify_passes_without_requiring_gitignored_weights(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = assets.verify(ROOT, require_weights=False)
        self.assertTrue(ok)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['retained_components'], 4)
        self.assertEqual(payload['frozen_code_checks'], 191)
        self.assertEqual(payload['retained_evidence_checks'], 669)

    def test_require_weights_fails_when_binary_weights_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'models').mkdir()
            (root / 'models/mainline.json').write_text(json.dumps({
                'models': [{
                    'id': 'demo',
                    'path': 'models/demo',
                    'alias_paths': [],
                    'files': [{'path': 'models/demo/controller.pt', 'sha256': '0' * 64}],
                }]
            }))
            (root / 'experiments/functional-v15-20261106').mkdir(parents=True)
            (root / 'experiments/functional-v21-20261107').mkdir(parents=True)
            (root / 'experiments/functional-v15-20261106/training-protocol.json').write_text('{"code_hashes": {}}')
            (root / 'experiments/functional-v21-20261107/training-protocol.json').write_text('{"code_hashes": {}}')
            (root / 'reports/mainline-technical-20261113/generated').mkdir(parents=True)
            (root / 'reports/mainline-technical-20261113/generated/evidence.json').write_text('{"sources": {}}')

            with redirect_stdout(io.StringIO()):
                self.assertTrue(assets.verify(root, require_weights=False))
                self.assertFalse(assets.verify(root, require_weights=True))


class CoreLayoutTests(unittest.TestCase):
    def test_core_and_baseline_aliases_exist(self):
        manifest = json.loads((ROOT / 'models/mainline.json').read_text())
        self.assertEqual(manifest['core_pipeline'], 'r15+r21_head')
        self.assertEqual(manifest['runtime_options'][0], 'r15+r21_head')
        self.assertEqual(set(manifest['baselines']), {'r15', 'r1', 'calibrated_sft'})
        self.assertTrue((ROOT / 'scripts/iea_studio/app.py').is_file())
        self.assertTrue((ROOT / 'iea/__init__.py').is_file())
        for model in manifest['models']:
            for alias in model.get('alias_paths', []):
                target = ROOT / alias
                self.assertTrue(target.is_dir(), alias)
                self.assertEqual(target.resolve(), (ROOT / model['path']).resolve())


if __name__ == '__main__':
    unittest.main()
