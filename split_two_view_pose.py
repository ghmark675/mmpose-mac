#!/usr/bin/env python3
"""Split left/right pose instances into FO/DTL JSON files."""

import argparse
import copy
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the left pose as FO and the right pose as DTL."
    )
    parser.add_argument("input", type=Path, help="Input *_pose.json")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the input directory")
    return parser.parse_args()


def horizontal_center(instance: dict) -> float:
    """Return a stable horizontal center, preferring bbox over keypoints."""
    bbox = instance.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return (float(bbox[0]) + float(bbox[2])) / 2

    keypoints = instance.get("keypoints", [])
    xs = [float(point[0]) for point in keypoints if isinstance(point, list) and point]
    if not xs:
        raise ValueError(f"Instance has neither a valid bbox nor keypoints: {instance!r}")
    return statistics.median(xs)


def find_split_x(frames: list) -> float:
    """Learn the left/right boundary from frames containing at least two people."""
    left_centers = []
    right_centers = []
    all_centers = []
    for frame in frames:
        centers = sorted(horizontal_center(item) for item in frame.get("instances", []))
        all_centers.extend(centers)
        if len(centers) >= 2:
            left_centers.append(centers[0])
            right_centers.append(centers[-1])

    if left_centers and right_centers:
        return (statistics.median(left_centers) + statistics.median(right_centers)) / 2
    if all_centers:
        return statistics.median(all_centers)
    raise ValueError("No pose instances found")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input.parent

    with args.input.open("r", encoding="utf-8") as file:
        frames = json.load(file)
    if not isinstance(frames, list):
        raise ValueError("Expected the JSON root to be a list of frames")

    split_x = find_split_x(frames)
    views = {"fo": [], "dtl": []}
    missing = {"fo": 0, "dtl": 0}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("instances", []), list):
            raise ValueError(f"Invalid frame structure: {frame!r}")
        instances = sorted(frame.get("instances", []), key=horizontal_center)
        selected = {"fo": None, "dtl": None}
        if len(instances) >= 2:
            selected["fo"], selected["dtl"] = instances[0], instances[-1]
        elif len(instances) == 1:
            view = "fo" if horizontal_center(instances[0]) < split_x else "dtl"
            selected[view] = instances[0]

        for view in ("fo", "dtl"):
            split_frame = copy.deepcopy(frame)
            instance = selected[view]
            split_frame["instances"] = [copy.deepcopy(instance)] if instance else []
            missing[view] += not split_frame["instances"]
            views[view].append(split_frame)

    stem = args.input.stem
    if stem.endswith("_pose"):
        stem = stem[:-5]
    output_dir.mkdir(parents=True, exist_ok=True)
    for view, split_frames in views.items():
        output = output_dir / f"{stem}_{view}_pose.json"
        with output.open("w", encoding="utf-8") as file:
            json.dump(split_frames, file, ensure_ascii=False)
        print(f"{view.upper()}: {output} ({len(split_frames)} frames, {missing[view]} empty)")
    print(f"Learned horizontal split: x={split_x:.2f} (left=FO, right=DTL)")


if __name__ == "__main__":
    main()
