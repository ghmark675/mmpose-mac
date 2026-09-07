import unittest
from types import SimpleNamespace

import numpy as np

from motionbert import (
    SEQ_LEN,
    coco_to_h36m,
    decode_3d,
    extract_sequence,
    lift_sequence,
    normalize_bbox,
)


class MotionBERTTest(unittest.TestCase):
    def test_coco_mapping_including_synthetic_joints(self):
        points = np.repeat(np.arange(17, dtype=np.float32)[:, None], 2, axis=1)[None]
        expected = [11.5, 12, 14, 16, 11, 13, 15, 8.5, 5.5, 0, 1.5, 5, 7, 9, 6, 8, 10]
        actual = coco_to_h36m(points)
        np.testing.assert_array_equal(actual[0, :, 0], expected)
        np.testing.assert_array_equal(actual[0, :, 1], expected)

    def test_bbox_normalization_is_translation_and_scale_invariant(self):
        points = np.tile([[[200, 300], [300, 500]]], (2, 1, 1)).astype(np.float32)
        boxes = np.array([[100, 100, 500, 700], [100, 100, 500, 700]], np.float32)
        points[1] = points[1] * 2 + [40, 60]
        boxes[1] = boxes[1] * 2 + [40, 60, 40, 60]
        normalized = normalize_bbox(points, boxes)
        np.testing.assert_allclose(normalized[0], normalized[1])
        np.testing.assert_allclose(normalized[0, 0], [528 - 400 / 6, 427 - 400 / 6])
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_bbox(points, np.zeros((2, 4), np.float32))

    def test_centered_window_and_replicate_padding(self):
        encoded = np.arange(300)[:, None, None]
        for index in (0, 120, 121, 150, 298, 299):
            window = extract_sequence(encoded, index)
            self.assertEqual(window.shape, (SEQ_LEN, 1, 1))
            self.assertEqual(window[SEQ_LEN // 2, 0, 0], index)
            np.testing.assert_array_equal(
                window[:, 0, 0], np.clip(np.arange(index - 121, index + 122), 0, 299)
            )
        np.testing.assert_array_equal(
            extract_sequence(encoded[:1], 0), np.zeros((243, 1, 1))
        )

    def test_decode_matches_motionbert_rootrel_and_meter_scale(self):
        output = np.full((17, 3), [0.25, -0.5, 0.75], np.float32)
        output[0] = [9, 8, 7]
        result = decode_3d(output, 1000, 600)
        np.testing.assert_array_equal(result[0], [0, 0, 0])
        np.testing.assert_allclose(result[1:], np.tile([0.5, -1, 1.5], (16, 1)))
        np.testing.assert_array_equal(output[0], [9, 8, 7])

    def test_lift_uses_center_frame_and_optional_bbox_normalization(self):
        windows = []

        def run(_, inputs):
            window = inputs["pose_2d"]
            windows.append(window.copy())
            return [window]

        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="pose_2d")], run=run
        )
        points = np.tile([[[100, 200]]], (3, 17, 1)).astype(np.float32)
        points[1] += [100, 100]
        points[2] += [200, 200]
        result = lift_sequence(session, points, None, 1000, 600, bbox_norm=False)
        self.assertEqual(result.shape, (3, 17, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[:, 0], 0)
        np.testing.assert_allclose(
            result[:, 1], [[-1.6, -0.4, 2], [-1.2, 0, 2], [-0.8, 0.4, 2]], atol=1e-6
        )
        np.testing.assert_array_equal(
            windows[0][0, :122], np.repeat(windows[0][0, :1], 122, axis=0)
        )
        np.testing.assert_array_equal(
            windows[-1][0, 121:], np.repeat(windows[-1][0, -1:], 122, axis=0)
        )
        boxes = np.tile([0, 0, 1000, 600], (3, 1))
        normalized = lift_sequence(session, points, boxes, 1000, 600)
        self.assertFalse(np.allclose(result, normalized))


if __name__ == "__main__":
    unittest.main()
