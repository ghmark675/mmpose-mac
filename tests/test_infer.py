import unittest

from infer import video_writer_fps


class VideoWriterFpsTest(unittest.TestCase):
    def test_problematic_camera_rate_is_rounded(self):
        self.assertEqual(video_writer_fps(119793 / 500), 240.0)
        self.assertEqual(video_writer_fps(239.58620689655172), 240.0)

    def test_normal_rates_are_unchanged(self):
        self.assertEqual(video_writer_fps(29.97), 29.97)
        self.assertEqual(video_writer_fps(60.0), 60.0)
        self.assertEqual(video_writer_fps(120.0), 120.0)

    def test_invalid_rate_uses_fallback(self):
        self.assertEqual(video_writer_fps(0.0), 25.0)


if __name__ == "__main__":
    unittest.main()
