import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pseudo3d import (
    COCO17_JOINT_NAMES,
    adapt_rtmpose_keypoints,
    estimate_vertical_scale,
    main,
    reconstruct_frame,
    reconstruct_sequence,
    save_keypoints3d,
)


def sample(offset=(0.0, 0.0), scale=1.0):
    # Non-degenerate deterministic COCO-like points, with ankle 16 at (16, 32).
    points = np.stack((np.arange(17), np.arange(17) * 2), axis=1).astype(np.float32)
    return points * np.float32(scale) + np.asarray(offset, dtype=np.float32)


class Pseudo3DTest(unittest.TestCase):
    def test_adapter_shape_dtype_and_mapping(self):
        expected = sample()
        actual = adapt_rtmpose_keypoints({"keypoints": expected.tolist()})
        self.assertEqual(actual.shape, (17, 2))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_raw_simcc_adapter_returns_original_pixels(self):
        simcc_x = np.zeros((17, 40), dtype=np.float32)
        simcc_y = np.zeros((17, 80), dtype=np.float32)
        simcc_x[:, 8] = 1
        simcc_y[:, 12] = 1
        # Crop point (4, 6), followed by x'=2x+10, y'=3y+20.
        inverse_affine = np.array([[2, 0, 10], [0, 3, 20]], dtype=np.float32)
        actual = adapt_rtmpose_keypoints(
            {"simcc_x": simcc_x, "simcc_y": simcc_y},
            inverse_affine=inverse_affine,
        )
        np.testing.assert_array_equal(actual, np.tile([18, 38], (17, 1)))
        self.assertEqual(actual.dtype, np.float32)

    def test_filtered_json_is_expanded_to_coco17(self):
        full = sample()
        indices = [0] + list(range(5, 17))
        actual = adapt_rtmpose_keypoints(
            {"keypoints": full[indices].tolist(), "keypoint_indices": indices}
        )
        np.testing.assert_array_equal(actual[indices], full[indices])
        np.testing.assert_array_equal(actual[1:5], np.tile(full[0], (4, 1)))

    def test_scale_and_frame_formula(self):
        fo = sample(offset=(100, 200), scale=2)
        dtl = sample(offset=(500, 700), scale=4)
        result = reconstruct_frame(fo, dtl)
        fo_centered = fo - fo[16]
        dtl_scaled = (dtl - dtl[16]) * 0.5
        expected = np.stack(
            (
                fo_centered[:, 0],
                fo_centered[:, 1],
                dtl_scaled[:, 0],
            ),
            axis=1,
        ).astype(np.float32)
        self.assertEqual(result.shape, (17, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, expected)
        self.assertAlmostEqual(float(estimate_vertical_scale(fo, dtl)), 0.5)

    def test_vertical_scale_ignores_horizontal_projection_and_translation(self):
        fo = sample(offset=(100, 200), scale=2)
        dtl = sample(offset=(500, 700), scale=4)
        dtl[:, 0] *= 7
        self.assertAlmostEqual(float(estimate_vertical_scale(fo, dtl)), 0.5)
        result = reconstruct_frame(fo, dtl)
        np.testing.assert_allclose(result[:, :2], fo - fo[16])
        np.testing.assert_allclose(result[:, 2], (dtl[:, 0] - dtl[16, 0]) * 0.5)

    def test_sequence_uses_one_scale_and_preserves_fo_y(self):
        fo = [sample(), sample()]
        dtl = [sample(scale=2), sample(scale=2)]
        dtl[1][:, 1] *= 1.1
        points, _ = reconstruct_sequence(enumerate(fo), enumerate(dtl))
        np.testing.assert_array_equal(points[0, :, 2], points[1, :, 2])
        np.testing.assert_array_equal(
            points[:, :, 1], np.stack([p[:, 1] - p[16, 1] for p in fo])
        )

    def test_horizontal_pose_cannot_determine_scale(self):
        pose = sample()
        pose[:, 1] = 100
        with self.assertRaisesRegex(ValueError, "vertical spans"):
            estimate_vertical_scale(pose, pose)

    def test_sequence_fixed_origin_and_indices(self):
        fo0, dtl0 = sample(), sample(scale=2)
        fo1 = fo0 + np.array([10, 20], np.float32)
        dtl1 = dtl0 + np.array([30, 40], np.float32)
        points, indices = reconstruct_sequence(
            [(3, fo0), (8, fo1)], [(3, dtl0), (8, dtl1)]
        )
        np.testing.assert_array_equal(indices, [3, 8])
        # With fixed origins the right ankle records the two-view translation.
        np.testing.assert_allclose(points[1, 16], [10, 20, 15])

    def test_invalid_pair_is_skipped_and_npz_schema(self):
        valid = sample()
        invalid = np.zeros((17, 2), dtype=np.float32)
        points, indices = reconstruct_sequence(
            [(2, invalid), (9, valid)], [(2, invalid), (9, valid)]
        )
        np.testing.assert_array_equal(indices, [9])
        with tempfile.TemporaryDirectory() as directory:
            path = save_keypoints3d(
                Path(directory) / "keypoints3d.npz", points, indices
            )
            with np.load(path) as saved:
                self.assertEqual(saved["keypoints3d"].shape, (1, 17, 3))
                self.assertEqual(saved["keypoints3d"].dtype, np.float32)
                np.testing.assert_array_equal(saved["frame_indices"], [9])
                np.testing.assert_array_equal(saved["joint_names"], COCO17_JOINT_NAMES)

    def test_manual_cli_smoothing_keeps_raw_reconstruction(self):
        poses = [sample() for _ in range(13)]
        poses[6][10, 0] += 35
        frames = [
            {"frame_id": i, "instances": [{"keypoints": pose.tolist()}]}
            for i, pose in enumerate(poses)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "poses.json"
            source.write_text(json.dumps(frames))
            output = root / "result.npz"
            with patch(
                "sys.argv",
                [
                    "pseudo3d.py",
                    str(source),
                    str(source),
                    "-o",
                    str(output),
                    "--smooth-fps",
                    "60",
                ],
            ):
                main()
            expected, ids = reconstruct_sequence(enumerate(poses), enumerate(poses))
            with np.load(root / "result_raw.npz") as raw, np.load(output) as smoothed:
                np.testing.assert_array_equal(raw["keypoints3d"], expected)
                np.testing.assert_array_equal(smoothed["frame_indices"], ids)
                self.assertAlmostEqual(
                    float(smoothed["keypoints3d"][6, 10, 0]),
                    float(expected[6, 10, 0] - 18),
                    places=5,
                )


if __name__ == "__main__":
    unittest.main()
