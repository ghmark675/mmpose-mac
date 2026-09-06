import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from pipeline_3d import run_pipeline
from pseudo3d import reconstruct_sequence


def prediction(frame_id):
    points = np.stack((np.arange(17), np.arange(17) * 2), axis=1)
    return {"frame_id": frame_id, "instances": [{"keypoints": points.tolist()}]}


class Pipeline3DTest(unittest.TestCase):
    def test_end_to_end_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fo, dtl = root / "fo.mp4", root / "dtl.mp4"
            det, pose = root / "det.onnx", root / "pose.onnx"
            for path in (fo, dtl, det, pose):
                path.touch()
            for path in (fo, dtl):
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
                writer.release()
            args = argparse.Namespace(
                fo_video=fo, dtl_video=dtl, output_dir=root / "result",
                det_model=det, pose_model=pose, config=root / "missing.json",
                det_thr=0.4, kpt_thr=0.3, per_frame_origin=False,
                save_pose_videos=False,
            )
            with patch("pipeline_3d.create_session", side_effect=["det", "pose"]), \
                 patch("pipeline_3d.run_video", side_effect=[[prediction(0)], [prediction(0)]]) as infer:
                paths = run_pipeline(args)

            self.assertEqual(infer.call_count, 2)
            self.assertTrue(paths["fo_json"].is_file())
            self.assertTrue(paths["dtl_json"].is_file())
            self.assertTrue(paths["npz"].is_file())
            self.assertTrue(paths["raw_npz"].is_file())
            self.assertTrue(paths["html"].is_file())
            self.assertNotIn("fo_video", paths)

    def test_smoothing_preserves_raw_data_and_drives_saved_viewer_and_projection(self):
        frames = [prediction(i) for i in range(13)]
        # A known temporal impulse gives an independent check of the 5-frame
        # quadratic SG response: its center coefficient is 17/35.
        frames[6]["instances"][0]["keypoints"][10][0] += 35
        raw_expected, ids = reconstruct_sequence(
            [(f["frame_id"], f["instances"][0]) for f in frames],
            [(f["frame_id"], f["instances"][0]) for f in frames],
        )
        for disabled in (False, True):
            with self.subTest(no_smooth=disabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                video = root / "source.mp4"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                         60, (64, 64))
                for _ in frames:
                    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
                writer.release()
                model = root / "model.onnx"
                model.touch()
                args = argparse.Namespace(
                    fo_video=video, dtl_video=video, output_dir=root / "result",
                    det_model=model, pose_model=model, config=root / "missing.json",
                    det_thr=0.4, kpt_thr=0.3, per_frame_origin=False,
                    save_pose_videos=False, no_smooth=disabled, smooth_window_ms=67,
                )
                with patch("pipeline_3d.create_session"), \
                     patch("pipeline_3d.run_video", side_effect=[frames, frames]):
                    paths = run_pipeline(args)
                with np.load(paths["raw_npz"]) as raw, np.load(paths["npz"]) as saved:
                    np.testing.assert_array_equal(raw["keypoints3d"], raw_expected)
                    np.testing.assert_array_equal(saved["frame_indices"], ids)
                    actual = saved["keypoints3d"].copy()
                expected_center = raw_expected[6, 10, 0] - (0 if disabled else 18)
                self.assertAlmostEqual(float(actual[6, 10, 0]), float(expected_center), places=5)
                if disabled:
                    np.testing.assert_array_equal(actual, raw_expected)

                payload, _ = json.JSONDecoder().raw_decode(
                    paths["html"].read_text().split("const data=", 1)[1]
                )
                display = actual.copy()
                display[:, :, 1] *= -1
                np.testing.assert_allclose(payload["points"], display)
                origin = np.asarray(frames[0]["instances"][0]["keypoints"])[16]
                np.testing.assert_allclose(payload["comparison"]["fo"]["projected"],
                                           actual[:, :, :2] + origin, atol=1e-5)
                np.testing.assert_allclose(payload["comparison"]["dtl"]["projected"],
                                           actual[:, :, [2, 1]] + origin, atol=1e-5)
                for name in ("fo", "dtl"):
                    self.assertEqual(json.loads(paths[f"{name}_json"].read_text()), frames)

    def test_missing_input_fails_before_loading_models(self):
        args = argparse.Namespace(
            fo_video=Path("missing-fo.mp4"), dtl_video=Path("missing-dtl.mp4"),
            output_dir=Path("out"), det_model=Path("det"), pose_model=Path("pose"),
            config=Path("config"), det_thr=0.4, kpt_thr=0.3,
            per_frame_origin=False, save_pose_videos=False,
        )
        with patch("pipeline_3d.create_session") as create:
            with self.assertRaisesRegex(FileNotFoundError, "FO video"):
                run_pipeline(args)
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
