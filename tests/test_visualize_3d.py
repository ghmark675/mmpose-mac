import tempfile
import unittest
from pathlib import Path

import numpy as np

from pseudo3d import COCO17_JOINT_NAMES
from visualize_3d import create_viewer_html, load_keypoints3d


class Visualize3DTest(unittest.TestCase):
    def test_load_and_create_self_contained_html(self):
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / "keypoints3d.npz"
            html = Path(directory) / "keypoints3d.html"
            points = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
            np.savez(npz, keypoints3d=points, frame_indices=[4, 9],
                     joint_names=COCO17_JOINT_NAMES)
            loaded, indices, names = load_keypoints3d(npz)
            np.testing.assert_array_equal(loaded, points)
            np.testing.assert_array_equal(indices, [4, 9])
            self.assertEqual(len(names), 17)
            create_viewer_html(npz, html)
            content = html.read_text(encoding="utf-8")
            self.assertIn("原始帧", content)
            self.assertIn('"frameIndices":[4,9]', content)
            self.assertIn("drawAxes()", content)
            self.assertIn("headRadius", content)
            self.assertIn('data-view="fo"', content)
            self.assertIn("view==='dtl'", content)
            self.assertIn("view==='top'", content)
            self.assertIn("x1=-x;y1=z;z2=-y", content)
            self.assertIn("innerHeight-115", content)
            self.assertNotIn("__DATA__", content)
            self.assertNotIn("https://", content)

    def test_empty_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / "empty.npz"
            np.savez(npz, keypoints3d=np.empty((0, 17, 3), np.float32),
                     frame_indices=[], joint_names=COCO17_JOINT_NAMES)
            with self.assertRaisesRegex(ValueError, "no frames"):
                load_keypoints3d(npz)


if __name__ == "__main__":
    unittest.main()
