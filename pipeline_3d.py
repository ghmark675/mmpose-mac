#!/usr/bin/env python3
"""End-to-end FO + DTL video to self-contained pseudo-3D HTML pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2

from infer import create_session, run_video
from keypoint_config import filter_predictions, load_keypoint_config
from pseudo3d import load_infer_json, reconstruct_sequence, save_keypoints3d
from smoothing import DEFAULT_WINDOW_MS, savgol_window_length, smooth_keypoints3d
from visualize_3d import build_comparison, create_viewer_html


ROOT = Path(__file__).resolve().parent


def _existing_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _write_predictions(path: Path, predictions: Any, config: dict | None) -> None:
    filtered = filter_predictions(predictions, config)
    with path.open("w", encoding="utf-8") as file:
        json.dump(filtered, file, ensure_ascii=False)


def _read_video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot read video FPS: {path}")
        return float(fps)
    finally:
        capture.release()


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    """Run both views through shared ONNX sessions and build the 3-D viewer."""
    fo_video = _existing_file(args.fo_video, "FO video")
    dtl_video = _existing_file(args.dtl_video, "DTL video")
    det_model = _existing_file(args.det_model, "detector model")
    pose_model = _existing_file(args.pose_model, "pose model")
    config_path = args.config.expanduser().resolve()
    config = load_keypoint_config(config_path)
    smoothing = not getattr(args, "no_smooth", False)
    window_ms = getattr(args, "smooth_window_ms", DEFAULT_WINDOW_MS)
    if smoothing:
        fps = _read_video_fps(fo_video)
        window_frames = savgol_window_length(fps, window_ms)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fo_json": output_dir / "fo_keypoints.json",
        "dtl_json": output_dir / "dtl_keypoints.json",
        "raw_npz": output_dir / "keypoints3d_raw.npz",
        "npz": output_dir / "keypoints3d.npz",
        "html": output_dir / "index.html",
    }
    if args.save_pose_videos:
        paths["fo_video"] = output_dir / "fo_pose.mp4"
        paths["dtl_video"] = output_dir / "dtl_pose.mp4"

    print("Loading ONNX models...")
    det_session = create_session(det_model)
    pose_session = create_session(pose_model)
    common = {
        "det_thr": args.det_thr,
        "kpt_thr": args.kpt_thr,
        "json": True,
    }
    for view, source in (("fo", fo_video), ("dtl", dtl_video)):
        print(f"\n[{view.upper()}] {source}")
        infer_args = SimpleNamespace(input=source, **common)
        predictions = run_video(
            infer_args, det_session, pose_session, paths.get(f"{view}_video")
        )
        _write_predictions(paths[f"{view}_json"], predictions, config)

    points, frame_indices = reconstruct_sequence(
        load_infer_json(paths["fo_json"]),
        load_infer_json(paths["dtl_json"]),
        fixed_origin=not args.per_frame_origin,
    )
    if len(frame_indices) == 0:
        raise RuntimeError(
            "No matching valid FO/DTL frames were reconstructed. "
            "Check that both videos contain a detected person and are frame-aligned."
        )
    save_keypoints3d(paths["raw_npz"], points, frame_indices)
    if smoothing:
        print(
            f"\nSmoothing 3-D: quadratic Savitzky-Golay, "
            f"{window_frames} frames at {fps:g} FPS (short runs use smaller windows)"
        )
        points = smooth_keypoints3d(points, frame_indices, fps=fps, window_ms=window_ms)
    save_keypoints3d(paths["npz"], points, frame_indices)
    comparison = build_comparison(
        points,
        frame_indices,
        paths["fo_json"],
        paths["dtl_json"],
        fo_video,
        dtl_video,
        per_frame_origin=args.per_frame_origin,
    )
    create_viewer_html(paths["npz"], paths["html"], config_path, comparison=comparison)
    print(f"\nDone: {len(frame_indices)} reconstructed frames")
    print(f"HTML: {paths['html']}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert aligned FO and DTL videos into an offline 3-D HTML viewer"
    )
    parser.add_argument("fo_video", type=Path, help="front-on (FO) video")
    parser.add_argument("dtl_video", type=Path, help="down-the-line (DTL) video")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output_3d"),
        help="artifact directory (default: output_3d)",
    )
    parser.add_argument(
        "--det-model", type=Path, default=ROOT / "models/rtmdet_m_person.onnx"
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=ROOT / "models/rtmpose_l_body8_384x288.onnx",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "keypoint_config.json")
    parser.add_argument("--det-thr", type=float, default=0.4)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="disable 3-D temporal smoothing (enabled by default)",
    )
    parser.add_argument(
        "--smooth-window-ms",
        type=float,
        default=DEFAULT_WINDOW_MS,
        help="SG window time span in milliseconds (default: 67; minimum 5 frames)",
    )
    parser.add_argument(
        "--per-frame-origin",
        action="store_true",
        help="center every frame independently instead of retaining translation",
    )
    parser.add_argument(
        "--save-pose-videos",
        action="store_true",
        help="also save FO/DTL videos with 2-D pose overlays",
    )
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
