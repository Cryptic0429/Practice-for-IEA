import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('assets', Path(__file__).with_name('assets.py'))
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)

ROOT = Path(__file__).resolve().parents[2]


class ArchiveSelectionTests(unittest.TestCase):
    def test_prefix_does_not_match_adjacent_model(self):
        manifest = {'entries': [{'path': 'models/r1/a'}, {'path': 'models/r15/b'}]}
        self.assertEqual(assets.selected_entries(manifest, ['models/r1']), [{'path': 'models/r1/a'}])

    def test_unsafe_paths_are_rejected(self):
        for path in ('../secret', '/absolute', 'models/../../secret', 'bad\npath'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                assets.safe_path(path)

    def test_no_match_cannot_trigger_full_restore(self):
        with self.assertRaises(ValueError):
            assets.selected_entries({'entries': [{'path': 'models/r1/a'}]}, ['missing'])


class CoreLayoutTests(unittest.TestCase):
    def test_core_and_baseline_aliases_exist(self):
        manifest = json.loads((ROOT / 'models/mainline.json').read_text())
        self.assertEqual(manifest['core_pipeline'], 'r15+r21_head')
        self.assertEqual(manifest['runtime_options'][0], 'r15+r21_head')
        self.assertEqual(set(manifest['baselines']), {'r15', 'r1', 'calibrated_sft'})
        self.assertTrue((ROOT / 'scripts/iea_studio/app.py').is_file())
        for model in manifest['models']:
            for alias in model.get('alias_paths', []):
                target = ROOT / alias
                self.assertTrue(target.is_dir(), alias)
                self.assertEqual(target.resolve(), (ROOT / model['path']).resolve())


if __name__ == '__main__':
    unittest.main()
