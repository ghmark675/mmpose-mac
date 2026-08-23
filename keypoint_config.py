"""Small shared loader for keypoint filtering and viewer styles."""

import json
from pathlib import Path


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def load_keypoint_config(path="keypoint_config.json"):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    visible = config.get("visible_joints", COCO17_NAMES)
    config["visible_joint_indices"] = [COCO17_NAMES.index(name) for name in visible]
    return config


def filter_predictions(predictions, config):
    """Compact keypoints in infer.py JSON output while recording COCO indices."""
    if config is None:
        return predictions
    indices = config["visible_joint_indices"]
    frames = predictions if isinstance(predictions, list) else [predictions]
    for item in frames:
        instances = item.get("instances", []) if isinstance(item, dict) else []
        if isinstance(item, dict) and "keypoints" in item:
            instances = [item]
        for instance in instances:
            instance["keypoints"] = [instance["keypoints"][i] for i in indices]
            if "keypoint_scores" in instance:
                instance["keypoint_scores"] = [instance["keypoint_scores"][i] for i in indices]
            instance["keypoint_indices"] = indices
    return predictions
