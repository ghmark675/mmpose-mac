import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DET_SIZE = (640, 640)
POSE_SIZE = (288, 384)
DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
DET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def create_session(path):
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")
    try:
        return ort.InferenceSession(str(path), providers=providers)
    except Exception:
        if providers[0] != "CoreMLExecutionProvider":
            raise
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def detect(session, image, threshold):
    height, width = image.shape[:2]
    scale = min(DET_SIZE[0] / width, DET_SIZE[1] / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    padded = np.full((DET_SIZE[1], DET_SIZE[0], 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    tensor = ((padded.astype(np.float32) - DET_MEAN) / DET_STD).transpose(2, 0, 1)[None]
    dets, labels = session.run(None, {session.get_inputs()[0].name: tensor})

    boxes = []
    for det, label in zip(dets[0], labels[0]):
        if int(label) != 0 or float(det[4]) < threshold:
            continue
        box = det[:4] / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append((box.astype(np.float32), float(det[4])))
    return boxes


def pose(session, image, box):
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    width, height = x2 - x1, y2 - y1
    aspect = POSE_SIZE[0] / POSE_SIZE[1]
    if width > height * aspect:
        height = width / aspect
    else:
        width = height * aspect
    width, height = width * 1.25, height * 1.25
    src = np.float32(
        [
            [center[0] - width / 2, center[1] - height / 2],
            [center[0] - width / 2, center[1] + height / 2],
            [center[0] + width / 2, center[1] - height / 2],
        ]
    )
    dst = np.float32([[0, 0], [0, POSE_SIZE[1] - 1], [POSE_SIZE[0] - 1, 0]])
    matrix = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(image, matrix, POSE_SIZE)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = ((crop - POSE_MEAN) / POSE_STD).transpose(2, 0, 1)[None]
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    values = dict(zip((item.name for item in session.get_outputs()), outputs))
    simcc_x, simcc_y = values["simcc_x"][0], values["simcc_y"][0]
    points = (
        np.stack((simcc_x.argmax(1), simcc_y.argmax(1)), axis=1).astype(np.float32) / 2
    )
    scores = np.sqrt(np.maximum(simcc_x.max(1) * simcc_y.max(1), 0))
    points = cv2.transform(points[None], cv2.invertAffineTransform(matrix))[0]
    return points, scores


def infer(det_session, pose_session, image, det_threshold, kpt_threshold):
    result = image.copy()
    predictions = []
    for box, det_score in detect(det_session, image, det_threshold):
        points, scores = pose(pose_session, image, box)
        predictions.append(
            {
                "keypoints": points.tolist(),
                "keypoint_scores": scores.tolist(),
                "bbox": box.tolist(),
                "bbox_score": det_score,
            }
        )
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(
            result,
            f"person {det_score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 128, 0),
            2,
        )
        for start, end in SKELETON:
            if scores[start] >= kpt_threshold and scores[end] >= kpt_threshold:
                cv2.line(
                    result,
                    tuple(np.rint(points[start]).astype(int)),
                    tuple(np.rint(points[end]).astype(int)),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        for point, score in zip(points, scores):
            if score >= kpt_threshold:
                cv2.circle(
                    result,
                    tuple(np.rint(point).astype(int)),
                    4,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )
    return result, predictions


def run_image(args, det_session, pose_session, output):
    image = cv2.imread(str(args.input))
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.input}")
    result, predictions = infer(
        det_session, pose_session, image, args.det_thr, args.kpt_thr
    )
    if not cv2.imwrite(str(output), result):
        raise RuntimeError(f"Failed to write result: {output}")
    return predictions


def run_video(args, det_session, pose_session, output):
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to read video: {args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to write video: {output}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    index = 0
    predictions = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result, instances = infer(
            det_session, pose_session, frame, args.det_thr, args.kpt_thr
        )
        writer.write(result)
        if args.json is not None:
            predictions.append({"frame_id": index, "instances": instances})
        index += 1
        print(f"\rProcessing: {index}/{frames or '?'}", end="", flush=True)
    print()
    capture.release()
    writer.release()
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="RTMDet + RTMPose image/video inference"
    )
    parser.add_argument("input", type=Path, help="path to an image or video")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--det-model", type=Path, default=Path("models/rtmdet_m_person.onnx")
    )
    parser.add_argument(
        "--pose-model", type=Path, default=Path("models/rtmpose_l_body8_384x288.onnx")
    )
    parser.add_argument("--det-thr", type=float, default=0.4)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument(
        "--json",
        nargs="?",
        const=True,
        type=Path,
        help="export keypoints to JSON, optionally specifying a path",
    )
    args = parser.parse_args()

    is_image = args.input.suffix.lower() in IMAGE_SUFFIXES
    output = args.output or Path(
        f"{args.input.stem}_pose{args.input.suffix if is_image else '.mp4'}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    det_session = create_session(args.det_model)
    pose_session = create_session(args.pose_model)
    if is_image:
        predictions = run_image(args, det_session, pose_session, output)
    else:
        predictions = run_video(args, det_session, pose_session, output)
    if args.json is not None:
        json_output = (
            Path(f"{args.input.stem}_pose.json") if args.json is True else args.json
        )
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with json_output.open("w", encoding="utf-8") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)
        print(f"Keypoints saved to: {json_output}")
    print(f"Result saved to: {output}")


if __name__ == "__main__":
    main()
