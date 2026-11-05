"""Boundary tests for scientific reward and error semantics; no model/GPU needed."""
import unittest
from stage2_reward import image_reward


class RewardTests(unittest.TestCase):
    def score(self, d0, d, valid=True, success=True, readable=True):
        return image_reward(dict(valid=valid), dict(input_readable=readable, reference_readable=True,
                            editor_success=success, baseline_metrics=dict(pixel_L1=d0),
                            edited_metrics=dict(pixel_L1=d)))

    def test_improvement_and_identity(self):
        self.assertAlmostEqual(self.score(.2, .1), .5)
        self.assertEqual(self.score(.2, .2), 0)
        self.assertAlmostEqual(self.score(.2, .3), -.5)

    def test_floor_and_clipping(self):
        self.assertAlmostEqual(self.score(.001, 0), .1)
        self.assertEqual(self.score(.1, .5), -1)
        self.assertEqual(self.score(.1, 0), 1)

    def test_policy_failures(self):
        self.assertEqual(self.score(.2, .2, valid=False), -2)
        self.assertEqual(self.score(.2, .2, success=False), -2)

    def test_data_errors_are_fatal(self):
        with self.assertRaises(RuntimeError):
            self.score(.2, .2, readable=False)
        with self.assertRaises(RuntimeError):
            self.score(float('nan'), .2)


if __name__ == '__main__':
    unittest.main()
