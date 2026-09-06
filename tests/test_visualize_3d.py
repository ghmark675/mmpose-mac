import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pseudo3d import COCO17_JOINT_NAMES, reconstruct_sequence
from visualize_3d import build_comparison, create_viewer_html, load_keypoints3d


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

    def test_reprojection_restores_pixels_for_both_origin_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
            for _ in range(3):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            base = np.stack((np.arange(17), np.arange(17) * 2), axis=1)
            fo = [base + 10, base + 14]
            dtl = [base * 2 + 5, base * 3 + 8]
            ids = [0, 2]
            paths = [root / "fo.json", root / "dtl.json"]
            for path, poses in zip(paths, (fo, dtl)):
                path.write_text(json.dumps([
                    {"frame_id": i, "instances": [{"keypoints": pose.tolist()}]}
                    for i, pose in zip(ids, poses)
                ]))
            for fixed in (True, False):
                points, indices = reconstruct_sequence(zip(ids, fo), zip(ids, dtl), fixed_origin=fixed)
                result = build_comparison(points, indices, *paths, video, video,
                                          per_frame_origin=not fixed)
                for name, poses in (("fo", fo), ("dtl", dtl)):
                    projected = np.asarray(result[name]["projected"])
                    np.testing.assert_allclose(projected[:, :, 0], np.asarray(poses)[:, :, 0], atol=1e-5)
                    self.assertEqual(len(result[name]["images"]), 2)
                    self.assertEqual(result[name]["fps"], 25)
                fo_origin = np.asarray(fo)[:, 16] if not fixed else fo[0][16]
                np.testing.assert_allclose(result["fo"]["projected"],
                                           points[:, :, :2] + np.asarray(fo_origin).reshape(-1, 1, 2))

    def test_empty_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / "empty.npz"
            np.savez(npz, keypoints3d=np.empty((0, 17, 3), np.float32),
                     frame_indices=[], joint_names=COCO17_JOINT_NAMES)
            with self.assertRaisesRegex(ValueError, "no frames"):
                load_keypoints3d(npz)


if __name__ == "__main__":
    unittest.main()
