import unittest
import numpy as np
from .common import parse, input_row
from .editor import edit, statistics, remap_old_label


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.tile(np.linspace(30, 225, 64).astype(np.uint8), (48, 1))
        self.gray = np.repeat(self.gray[..., None], 3, axis=2)

    def test_keep_original_preserves_exact_pixels(self):
        for parameters in ({}, {'contrast': 0}, {'temperature': 0, 'brightness': 0}):
            np.testing.assert_array_equal(edit(self.gray, parameters), self.gray)

    def test_contrast_changes_in_the_requested_direction(self):
        for value in (-30, -1, 1, 30):
            changed = edit(self.gray, {'contrast': value}).astype(float).std()
            self.assertGreater(value * (changed - self.gray.astype(float).std()), 0)

    def test_warm_and_cool_are_visually_distinct(self):
        warm = statistics(edit(self.gray, {'temperature': 30}))['warmth']
        cool = statistics(edit(self.gray, {'temperature': -30}))['warmth']
        self.assertGreater(warm, .05)
        self.assertLess(cool, -.05)

    def test_light_controls_follow_both_directions(self):
        for tool in ('exposure','brightness','highlights','shadows'):
            for direction in (-1,1):
                changed = edit(self.gray, {tool: direction*25})
                self.assertGreater(direction*(changed.astype(float).mean()-self.gray.mean()), 0, tool)

    def test_saturation_and_tint_follow_both_directions(self):
        source = self.gray.copy()
        source[..., 0] = np.minimum(source[..., 0].astype(int)+20,255)
        original = statistics(source)
        for direction in (-1,1):
            changed = statistics(edit(source, {'saturation': direction*25}))
            self.assertGreater(direction*(changed['saturation']-original['saturation']), 0)
            tinted = edit(source, {'tint': direction*25})
            self.assertGreater(direction*(tinted[...,1].astype(float).mean()-source[...,1].mean()), 0)

    def test_shadow_lift_does_not_blow_out_white_sky(self):
        source = self.gray.copy()
        source[:, 32:] = 250
        out = edit(source, {'shadows': 35})
        self.assertGreater(out[:, :20].mean(), source[:, :20].mean() + 5)
        self.assertLessEqual(abs(out[:, 32:].mean() - 250), 1)

    def test_shadow_curve_does_not_raise_true_black_floor(self):
        source = self.gray.copy()
        source[:, :8] = 0
        source[:, -8:] = 255
        out = edit(source, {'shadows': 60})
        np.testing.assert_array_equal(out[:, :8], source[:, :8])
        np.testing.assert_array_equal(out[:, -8:], source[:, -8:])

    def test_lowering_highlights_preserves_dark_faces(self):
        source = np.zeros((40,40,3),np.uint8)
        source[:,:20] = 45
        source[:,20:] = 245
        out = edit(source, {'highlights': -40})
        self.assertLess(abs(out[:,:20].astype(float).mean()-45), 1)
        self.assertLess(out[:,20:].mean(), 225)

    def test_old_negative_contrast_label_can_mean_increase(self):
        self.assertGreater(remap_old_label({'contrast': -10})['contrast'], 0)
        self.assertEqual(remap_old_label({'contrast': 0})['contrast'], 0)

    def test_sepia_turns_green_to_brown_without_destroying_white(self):
        source = np.zeros((40, 40, 3), np.uint8)
        source[:, :20] = [45, 135, 55]
        source[:, 20:] = 250
        out = edit(source, {'sepia': 100})
        self.assertGreater(out[:, :20, 0].mean(), out[:, :20, 1].mean())
        self.assertGreater(out[:, :20, 1].mean(), out[:, :20, 2].mean())
        self.assertLessEqual(abs(out[:, 20:].mean() - 250), 1)

    def test_reject_malformed_or_unsupported_outputs(self):
        for response in ('{"brightness":2}', '```json\n{"brightness":true}\n```',
                         '```json\n{"brightness":101}\n```',
                         '```json\n{"brightness":1,"brightness":2}\n```',
                         '```json\n{"remove_object":1}\n```',
                         '```json\n{"brightness":NaN}\n```',
                         'Explanation\n```json\n{}\n```'):
            with self.assertRaises(ValueError):
                parse(response)

    def test_inference_input_does_not_contain_answers(self):
        case = {'image': 'data/images/example.jpg', 'instructions': {'zh': '调暖一点'},
                'parameters': {'temperature': 90}, 'reference': 'secret.jpg',
                'review': 'The correct output is warm.'}
        result = input_row(case, 'zh')
        self.assertEqual(set(result['extra_info']), {'origin_image_path'})
        self.assertEqual([m['role'] for m in result['conversations']], ['system', 'user'])
        self.assertNotIn('secret.jpg', str(result))
        self.assertNotIn('90', str(result))


if __name__ == '__main__':
    unittest.main()
