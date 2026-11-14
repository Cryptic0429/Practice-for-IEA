"""Run frozen numerical editor checks in a temporary root, preserving evidence."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from functional_v15 import test_shadow_invariants as frozen


class EditorContractTests(unittest.TestCase):
    def test_shadow_and_unchanged_tool_invariants(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('scripts/functional_v11/editor.py', 'scripts/functional_v15/editor.py'):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project / name, target)
            with patch.object(frozen, 'ROOT', root), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
                frozen.main()
            self.assertTrue((root / 'experiments/functional-v15-20261106/shadow-invariant-tests.json').exists())


if __name__ == '__main__':
    unittest.main()
