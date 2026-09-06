import unittest

import numpy as np

from smoothing import DEFAULT_WINDOW_MS, savgol_window_length, smooth_keypoints3d


def coordinates(values):
    return np.broadcast_to(
        np.asarray(values, dtype=np.float32)[:, None, None],
        (len(values), 17, 3),
    ).copy()


class SmoothingTest(unittest.TestCase):
    def test_time_span_tracks_source_fps(self):
        self.assertEqual(DEFAULT_WINDOW_MS, 67.0)
        for fps, expected in ((30, 5), (60, 5), (120, 9), (240, 17)):
            with self.subTest(fps=fps):
                self.assertEqual(savgol_window_length(fps), expected)
        self.assertEqual(savgol_window_length(60, 200), 13)
        self.assertEqual(savgol_window_length(60, 1), 5)

    def test_preserves_quadratic_motion_including_edges(self):
        t = np.arange(41, dtype=np.float32)[:, None, None]
        offsets = np.arange(51, dtype=np.float32).reshape(1, 17, 3)
        points = 100 + offsets + t * (offsets / 8 - 4) + t * t * (offsets / 64)
        before = points.copy()
        indices = np.arange(100, 141)
        result = smooth_keypoints3d(points, indices, fps=120)
        np.testing.assert_allclose(result, points, rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(points, before)
        np.testing.assert_array_equal(indices, np.arange(100, 141))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, points.shape)
        self.assertFalse(np.shares_memory(points, result))

    def test_centered_impulse_matches_known_sg_kernel_without_phase_shift(self):
        points = np.zeros((21, 17, 3), dtype=np.float32)
        points[10, 3, 2] = 1
        result = smooth_keypoints3d(points, np.arange(21), fps=60)
        expected = np.zeros_like(points)
        expected[8:13, 3, 2] = np.array([-3, 12, 17, 12, -3]) / 35
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(np.argmax(result[:, 3, 2]), 10)
        np.testing.assert_allclose(result[:, 3, 2], result[::-1, 3, 2])

    def test_reduces_high_frequency_jitter_while_preserving_linear_motion(self):
        t = np.arange(61, dtype=np.float32)
        trajectory = 30 + t / 2
        jitter = np.where(np.arange(61) % 2 == 0, 1.0, -1.0)
        points = coordinates(trajectory + jitter)
        result = smooth_keypoints3d(points, np.arange(61), fps=60)
        error = result[2:-2, 0, 0] - trajectory[2:-2]
        np.testing.assert_allclose(error, -13 / 35 * jitter[2:-2], atol=2e-6)
        self.assertLess(np.sqrt(np.mean(error ** 2)), 0.4)

    def test_gaps_are_smoothed_independently_without_filling_frames(self):
        first = coordinates(np.array([0, 0, 1, 0, 0], dtype=np.float32) + 1000)
        second = coordinates(np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.float32) - 1000)
        points = np.concatenate((first, second))
        indices = np.r_[np.arange(5), np.arange(100, 107)]
        result = smooth_keypoints3d(points, indices, fps=120)
        expected = np.concatenate((
            smooth_keypoints3d(first, np.arange(5), fps=120),
            smooth_keypoints3d(second, np.arange(7), fps=120),
        ))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, points.shape)
        np.testing.assert_array_equal(indices, np.r_[np.arange(5), np.arange(100, 107)])

    def test_short_segments_remain_unchanged(self):
        values = [5, -4, 3, -2, 1, -6, 7, -8, 9, -10]
        points = coordinates(values)
        # Four independent segments, of lengths 1, 2, 3 and 4.
        indices = [0, 10, 11, 20, 21, 22, 30, 31, 32, 33]
        result = smooth_keypoints3d(points, indices, fps=240)
        np.testing.assert_array_equal(result, points)
        self.assertFalse(np.shares_memory(result, points))

    def test_five_frame_segment_shrinks_window_and_fits_edges(self):
        points = coordinates([0, 0, 1, 0, 0])
        result = smooth_keypoints3d(points, np.arange(5), fps=240)
        expected = coordinates(np.array([-3, 12, 17, 12, -3]) / 35)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_even_segment_shrinks_to_odd_window(self):
        points = coordinates([1, 4, 1, 4, 1, 4])
        result = smooth_keypoints3d(points, np.arange(6), fps=240)
        expected = smooth_keypoints3d(points, np.arange(6), fps=60)
        np.testing.assert_array_equal(result, expected)
        self.assertGreater(np.max(np.abs(result - points)), 0.5)

    def test_empty_sequence(self):
        points = np.empty((0, 17, 3), dtype=np.float64)
        result = smooth_keypoints3d(points, [], fps=60)
        self.assertEqual(result.shape, points.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_rejects_invalid_fps_or_window(self):
        for invalid in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "fps"):
                    savgol_window_length(invalid)
                with self.assertRaisesRegex(ValueError, "window_ms"):
                    savgol_window_length(60, invalid)
                with self.assertRaisesRegex(ValueError, "fps"):
                    smooth_keypoints3d(np.empty((0, 17, 3)), [], fps=invalid)

    def test_rejects_invalid_coordinates_and_frame_indices(self):
        for shape in ((5, 17, 2), (5, 16, 3), (17, 3)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    smooth_keypoints3d(np.zeros(shape), np.arange(5), fps=60)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            points = coordinates([0, 1, 2, 3, 4])
            points[2, 0, 0] = invalid
            with self.subTest(coordinate=invalid):
                with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                    smooth_keypoints3d(points, np.arange(5), fps=60)
        points = coordinates([0, 1, 2, 3, 4])
        for indices in ([0, 1], [[0, 1, 2, 3, 4]], [0, 1, 1, 2, 3],
                        [4, 3, 2, 1, 0], [0, 1, 2, 3, 4.5],
                        [False, True, True, True, True]):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "frame_indices"):
                    smooth_keypoints3d(points, indices, fps=60)


if __name__ == "__main__":
    unittest.main()
