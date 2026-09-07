import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from motionbert import H36M_JOINT_NAMES, H36M_SKELETON
from pipeline_motionbert import run_pipeline, select_person_sequence


def person(bbox, offset=0, joints=17):
    points = np.arange(joints * 2, dtype=np.float32).reshape(joints, 2) + offset
    return {"bbox": bbox, "keypoints": points.tolist()}


class PipelineMotionBERTTest(unittest.TestCase):
    def test_tracking_keeps_identity_and_fills_missing_detections(self):
        first = person([0, 0, 20, 20])
        moved = person([1, 0, 21, 20], 10)
        returned = person([2, 0, 22, 20], 20)
        other = person([100, 100, 140, 140], 100)
        frames = [
            {"frame_id": 5, "instances": []},
            {"frame_id": 7, "instances": [person([100, 100, 105, 105]), first]},
            {"frame_id": 8, "instances": [moved, other]},
            {"frame_id": 9, "instances": []},
            {"frame_id": 10, "instances": [other]},
            {"frame_id": 11, "instances": [other, returned]},
        ]
        points, boxes, indices, detected = select_person_sequence(frames)
        expected = [first, moved, moved, moved, returned]
        np.testing.assert_array_equal(points, [item["keypoints"] for item in expected])
        np.testing.assert_array_equal(boxes, [item["bbox"] for item in expected])
        np.testing.assert_array_equal(indices, [7, 8, 9, 10, 11])
        np.testing.assert_array_equal(detected, [True, True, False, False, True])

    def test_no_person_raises(self):
        with self.assertRaisesRegex(ValueError, "No person"):
            select_person_sequence([{"frame_id": 0, "instances": []}])

    def test_image_and_video_write_complete_h36m_outputs(self):
        for is_image in (True, False):
            with self.subTest(is_image=is_image), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / ("input.png" if is_image else "input.mp4")
                image = np.zeros((64, 96, 3), dtype=np.uint8)
                if is_image:
                    self.assertTrue(cv2.imwrite(str(source), image))
                else:
                    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                                             24, (96, 64))
                    self.assertTrue(writer.isOpened())
                    for _ in range(3):
                        writer.write(image)
                    writer.release()
                model = root / "model.onnx"
                model.touch()
                args = argparse.Namespace(
                    input=source, output_dir=root / "result", det_model=model,
                    pose_model=model, lift_model=model, det_thr=0.4, kpt_thr=0.3,
                    no_bbox_norm=True, save_pose_video=False,
                )
                instance = person([0, 0, 60, 60], joints=26)
                predictions = ([{"frame_id": 0, "instances": [instance]}] if is_image else [
                    {"frame_id": 0, "instances": []},
                    {"frame_id": 1, "instances": [instance]},
                    {"frame_id": 2, "instances": []},
                ])
                ids = [0] if is_image else [1, 2]
                detected = [True] if is_image else [True, False]
                points3d = np.arange(len(ids) * 51, dtype=np.float32).reshape(-1, 17, 3) / 100
                with patch("pipeline_motionbert.create_session", side_effect=["det", "pose"]), \
                     patch("pipeline_motionbert.ort.InferenceSession", return_value="lift"), \
                     patch("pipeline_motionbert.infer", return_value=(image, [instance])), \
                     patch("pipeline_motionbert.run_video", return_value=predictions), \
                     patch("pipeline_motionbert.lift_sequence", return_value=points3d) as lift, \
                     patch("visualize_3d.load_keypoint_config", return_value={
                         "visible_joint_indices": [0], "head_joint": "nose",
                     }):
                    paths = run_pipeline(args)

                self.assertEqual(set(paths), {"json_2d", "json_3d", "npz", "html"}
                                 | ({"image"} if is_image else set()))
                self.assertTrue(all(path.is_file() for path in paths.values()))
                self.assertEqual(json.loads(paths["json_2d"].read_text()), predictions)
                np.testing.assert_array_equal(lift.call_args.args[1],
                                              [instance["keypoints"][:17]] * len(ids))
                self.assertEqual(lift.call_args.args[3:], (96, 64))
                self.assertEqual(lift.call_args.kwargs, {"bbox_norm": False})
                fps = 1.0 if is_image else 24.0
                with np.load(paths["npz"]) as saved:
                    np.testing.assert_array_equal(saved["keypoints3d"], points3d)
                    np.testing.assert_array_equal(saved["frame_indices"], ids)
                    np.testing.assert_array_equal(saved["detected"], detected)
                    np.testing.assert_array_equal(saved["joint_names"], H36M_JOINT_NAMES)
                    np.testing.assert_array_equal(saved["skeleton"], H36M_SKELETON)
                    self.assertEqual(float(saved["fps"]), fps)
                    self.assertEqual(str(saved["units"]), "meters")
                    self.assertEqual(str(saved["coordinate_system"]), "camera")
                    self.assertEqual(str(saved["head_joint"]), "head")
                payload = json.loads(paths["json_3d"].read_text())
                self.assertEqual(payload["units"], "meters")
                self.assertEqual(payload["coordinate_system"], "camera")
                self.assertEqual(payload["joint_names"], list(H36M_JOINT_NAMES))
                self.assertEqual(payload["image_size"], [96, 64])
                self.assertEqual(payload["fps"], fps)
                self.assertEqual([frame["frame_id"] for frame in payload["frames"]], ids)
                self.assertEqual([frame["detected"] for frame in payload["frames"]], detected)
                np.testing.assert_array_equal([f["keypoints_3d"] for f in payload["frames"]], points3d)
                viewer, _ = json.JSONDecoder().raw_decode(
                    paths["html"].read_text().split("const data=", 1)[1])
                self.assertEqual(viewer["visible"], list(range(17)))
                self.assertEqual(viewer["head"], "head")
                self.assertEqual(viewer["fps"], fps)
                self.assertEqual(viewer["frameIndices"], ids)


if __name__ == "__main__":
    unittest.main()
