import numpy as np


SEQ_LEN = 243
BBOX_CENTER = np.array([528.0, 427.0], dtype=np.float32)
BBOX_SCALE = 400.0
H36M_JOINT_NAMES = np.asarray([
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
])
H36M_SKELETON = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
)


def coco_to_h36m(keypoints):
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (17, 2):
        raise ValueError(f"Expected COCO keypoints with shape (N, 17, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Keypoints contain NaN or infinity")
    result = np.empty_like(points)
    result[:, 0] = (points[:, 11] + points[:, 12]) / 2
    result[:, 8] = (points[:, 5] + points[:, 6]) / 2
    result[:, 7] = (result[:, 0] + result[:, 8]) / 2
    result[:, 10] = (points[:, 1] + points[:, 2]) / 2
    result[:, [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
        points[:, [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
    return result


def normalize_bbox(keypoints, bboxes):
    boxes = np.asarray(bboxes, dtype=np.float32)
    if boxes.shape != (len(keypoints), 4):
        raise ValueError("Expected one xyxy bbox per frame with shape (N, 4)")
    sizes = boxes[:, 2:] - boxes[:, :2]
    if not np.isfinite(boxes).all() or np.any(sizes <= 0):
        raise ValueError("Bboxes must have finite coordinates and positive width/height")
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    scale = sizes.max(axis=1)
    return (keypoints - center[:, None]) / scale[:, None, None] * BBOX_SCALE + BBOX_CENTER


def encode_2d(keypoints, width, height):
    result = np.ones((*keypoints.shape[:-1], 3), dtype=np.float32)
    result[..., :2] = keypoints / width * 2 - np.array([1, height / width], np.float32)
    return result


def extract_sequence(encoded, frame_index):
    indices = np.arange(frame_index - SEQ_LEN // 2, frame_index + SEQ_LEN // 2 + 1)
    return encoded[np.clip(indices, 0, len(encoded) - 1)]


def decode_3d(output, width, height):
    result = np.asarray(output, dtype=np.float32).copy()
    result[0] = 0
    result[:, :2] = (result[:, :2] + np.array([1, height / width], np.float32)) * (width / 2)
    result[:, 2] *= width / 2
    result *= 4
    result -= result[0].copy()
    return result / 1000


def lift_sequence(session, keypoints, bboxes, width, height, bbox_norm=True):
    """Lift COCO pixel poses (N, 17, 2) to root-relative H36M poses in meters."""
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Image width and height must be finite and positive")
    points = coco_to_h36m(keypoints)
    if bbox_norm:
        points = normalize_bbox(points, bboxes)
    encoded = encode_2d(points, width, height)
    poses = np.empty((len(points), 17, 3), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    for index in range(len(points)):
        window = extract_sequence(encoded, index)[None]
        output = session.run(None, {input_name: window})[0]
        if output.shape != (1, SEQ_LEN, 17, 3):
            raise ValueError(f"Expected MotionBERT output (1, {SEQ_LEN}, 17, 3), got {output.shape}")
        if not np.isfinite(output).all():
            raise RuntimeError("MotionBERT output contains NaN or infinity")
        poses[index] = decode_3d(output[0, SEQ_LEN // 2], width, height)
        print(f"\rLifting: {index + 1}/{len(points)}", end="", flush=True)
    if len(points):
        print()
    return poses
