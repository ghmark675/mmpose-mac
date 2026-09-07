"""RTMDet + RTMPose + MotionBERT image/video to 3-D skeleton pipeline."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort

from infer import IMAGE_SUFFIXES, create_session, infer, run_video
from motionbert import H36M_JOINT_NAMES, H36M_SKELETON, lift_sequence
from visualize_3d import create_viewer_html


ROOT = Path(__file__).resolve().parent


def select_person_sequence(predictions):
    keypoints, bboxes, frame_indices, detected = [], [], [], []
    last = None
    for frame in predictions:
        instances = frame["instances"]
        selected = None
        if instances:
            boxes = np.asarray([item["bbox"] for item in instances], dtype=np.float32)
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            if last is None:
                selected = instances[int(areas.argmax())]
            else:
                box = np.asarray(last["bbox"], dtype=np.float32)
                overlap = np.maximum(0, np.minimum(boxes[:, 2:], box[2:])
                                     - np.maximum(boxes[:, :2], box[:2]))
                intersection = np.prod(overlap, axis=1)
                iou = intersection / np.maximum(
                    areas + np.prod(box[2:] - box[:2]) - intersection, 1e-6
                )
                index = int(iou.argmax())
                if iou[index] > 0.1:
                    selected = instances[index]
        if selected is not None:
            last = selected
        if last is None:
            continue
        keypoints.append(np.asarray(last["keypoints"], dtype=np.float32)[:17, :2])
        bboxes.append(last["bbox"])
        frame_indices.append(frame["frame_id"])
        detected.append(selected is not None)
    if not keypoints:
        raise ValueError("No person detected in the input")
    return (np.stack(keypoints), np.asarray(bboxes, dtype=np.float32),
            np.asarray(frame_indices, dtype=np.int64), np.asarray(detected, dtype=bool))


def run_pipeline(args):
    for path in (args.input, args.det_model, args.pose_model, args.lift_model):
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
    is_image = args.input.suffix.lower() in IMAGE_SUFFIXES
    if is_image:
        image = cv2.imread(str(args.input))
        if image is None:
            raise ValueError(f"Cannot read image: {args.input}")
        height, width = image.shape[:2]
        fps = 1.0
    else:
        capture = cv2.VideoCapture(str(args.input))
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or min(width, height) <= 0:
                raise ValueError(f"Cannot read video: {args.input}")
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot read video FPS: {args.input}")
        finally:
            capture.release()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json_2d": output / "keypoints2d.json",
        "json_3d": output / "keypoints3d.json",
        "npz": output / "keypoints3d.npz",
        "html": output / "index.html",
    }
    print("Loading ONNX models...")
    det_session = create_session(args.det_model)
    pose_session = create_session(args.pose_model)
    lift_session = ort.InferenceSession(str(args.lift_model), providers=["CPUExecutionProvider"])
    if is_image:
        result, instances = infer(det_session, pose_session, image, args.det_thr, args.kpt_thr)
        paths["image"] = output / "pose.jpg"
        if not cv2.imwrite(str(paths["image"]), result):
            raise RuntimeError(f"Cannot write image: {paths['image']}")
        predictions = [{"frame_id": 0, "instances": instances}]
    else:
        if args.save_pose_video:
            paths["video"] = output / "pose.mp4"
        infer_args = SimpleNamespace(input=args.input, det_thr=args.det_thr,
                                     kpt_thr=args.kpt_thr, json=True)
        predictions = run_video(infer_args, det_session, pose_session, paths.get("video"))
    paths["json_2d"].write_text(json.dumps(predictions), encoding="utf-8")
    keypoints, bboxes, frame_indices, detected = select_person_sequence(predictions)
    print(f"Lifting {len(keypoints)} frames with MotionBERT ({int((~detected).sum())} filled)...")
    points = lift_sequence(lift_session, keypoints, bboxes, width, height,
                           bbox_norm=not args.no_bbox_norm)
    np.savez_compressed(
        paths["npz"], keypoints3d=points, frame_indices=frame_indices,
        joint_names=H36M_JOINT_NAMES, skeleton=H36M_SKELETON, fps=fps,
        units="meters", head_joint="head", detected=detected,
        coordinate_system="camera",
    )
    frames = [dict(frame_id=int(i), detected=bool(valid), keypoints_3d=pose.tolist())
              for i, valid, pose in zip(frame_indices, detected, points)]
    paths["json_3d"].write_text(json.dumps(dict(
        units="meters", root_relative=True, joint_names=H36M_JOINT_NAMES.tolist(),
        coordinate_system="camera", image_size=[width, height], fps=fps, frames=frames,
    )), encoding="utf-8")
    create_viewer_html(paths["npz"], paths["html"])
    print(f"Done: {len(points)} frames\nHTML: {paths['html']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="RTMDet + RTMPose + MotionBERT 3-D inference")
    parser.add_argument("input", type=Path, help="image or video containing one main person")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output_motionbert"))
    parser.add_argument("--det-model", type=Path, default=ROOT / "models/rtmdet_m_person.onnx")
    parser.add_argument("--pose-model", type=Path, default=ROOT / "models/rtmpose_l_body8_384x288.onnx")
    parser.add_argument("--lift-model", type=Path, default=ROOT / "models/motionbert_h36m.onnx")
    parser.add_argument("--det-thr", type=float, default=0.4)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--no-bbox-norm", action="store_true", help="use raw pixel coordinates for lifting")
    parser.add_argument("--save-pose-video", action="store_true", help="also save a 2-D pose overlay video")
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
