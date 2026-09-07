"""Basic two-view pseudo-3D reconstruction for COCO-17 RTMPose output.

FO is the front view and DTL is the down-the-line view.  Coordinates are
expressed in pixels; this module intentionally performs no camera calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from smoothing import DEFAULT_WINDOW_MS, savgol_window_length, smooth_keypoints3d


COCO17_JOINT_NAMES = np.asarray(
    [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
)


def _validate_keypoints(points: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2 or points.shape[0] != 17 or points.shape[1] < 2:
        raise ValueError(f"expected keypoints with shape (17, >=2), got {points.shape}")
    points = np.ascontiguousarray(points[:, :2], dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("keypoints contain NaN or infinity")
    return points


def adapt_rtmpose_keypoints(
    output: Any,
    *,
    inverse_affine: np.ndarray | None = None,
    simcc_split_ratio: float = 2.0,
) -> np.ndarray:
    """Adapt one RTMPose result to original-video ``(17, 2) float32`` pixels.

    ``output`` may be the ``(points, scores)`` tuple returned by ``infer.pose``,
    an instance mapping containing ``keypoints``, a keypoint array, or a mapping
    containing raw ``simcc_x``/``simcc_y``. Raw SimCC coordinates are crop-space,
    so ``inverse_affine`` (crop to original image) is required in that case.
    """
    if isinstance(output, Mapping):
        if "keypoints" in output:
            points = output["keypoints"]
            if "keypoint_indices" in output:
                compact = np.asarray(points, dtype=np.float32)
                indices = np.asarray(output["keypoint_indices"], dtype=np.int64)
                if compact.shape != (len(indices), 2):
                    raise ValueError(
                        "filtered keypoints and keypoint_indices do not match"
                    )
                if 0 not in indices:
                    raise ValueError(
                        "filtered keypoints must retain nose (COCO index 0)"
                    )
                # Hidden face slots are irrelevant to scale/reconstruction and
                # share the nose coordinate to preserve the COCO-17 array shape.
                nose = compact[np.flatnonzero(indices == 0)[0]]
                points = np.tile(nose, (17, 1))
                points[indices] = compact
        elif "simcc_x" in output and "simcc_y" in output:
            if inverse_affine is None:
                raise ValueError("inverse_affine is required for raw SimCC output")
            simcc_x = np.asarray(output["simcc_x"])
            simcc_y = np.asarray(output["simcc_y"])
            if simcc_x.ndim == 3:
                simcc_x = simcc_x[0]
            if simcc_y.ndim == 3:
                simcc_y = simcc_y[0]
            points = np.stack((simcc_x.argmax(-1), simcc_y.argmax(-1)), axis=-1)
            points = points.astype(np.float32) / np.float32(simcc_split_ratio)
            affine = np.asarray(inverse_affine, dtype=np.float32)
            if affine.shape != (2, 3):
                raise ValueError("inverse_affine must have shape (2, 3)")
            points = points @ affine[:, :2].T + affine[:, 2]
        else:
            raise ValueError("RTMPose mapping must contain keypoints or SimCC outputs")
    elif isinstance(output, tuple) and len(output) >= 1:
        points = output[0]
    else:
        points = output
    return _validate_keypoints(points)


def estimate_vertical_scale(fo_points: Any, dtl_points: Any) -> np.float32:
    """Fit one DTL-to-FO scale from corresponding body-joint vertical spans."""
    fo = np.asarray(fo_points, dtype=np.float64).reshape(-1, 17, 2)[:, 5:, 1]
    dtl = np.asarray(dtl_points, dtype=np.float64).reshape(-1, 17, 2)[:, 5:, 1]
    if fo.shape != dtl.shape or not np.isfinite(fo).all() or not np.isfinite(dtl).all():
        raise ValueError("scale requires matching finite poses")
    a, b = np.triu_indices(12, 1)
    dy_fo, dy_dtl = fo[:, a] - fo[:, b], dtl[:, a] - dtl[:, b]
    valid = (
        (np.abs(dy_fo) > np.maximum(np.ptp(fo, axis=1)[:, None] * 0.2, 1e-6))
        & (np.abs(dy_dtl) > np.maximum(np.ptp(dtl, axis=1)[:, None] * 0.2, 1e-6))
        & (dy_fo * dy_dtl > 0)
    )
    if not valid.any():
        raise ValueError("no usable corresponding vertical spans")
    x, y = dy_dtl[valid], dy_fo[valid]
    return np.float32(np.dot(x, y) / np.dot(x, x))


def reconstruct_frame(
    fo_output: Any,
    dtl_output: Any,
    *,
    fo_origin: Sequence[float] | None = None,
    dtl_origin: Sequence[float] | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Reconstruct one pseudo-3D frame, optionally using fixed view origins."""
    fo_points = adapt_rtmpose_keypoints(fo_output)
    dtl_points = adapt_rtmpose_keypoints(dtl_output)
    fo_origin_array = fo_points[16] if fo_origin is None else np.asarray(fo_origin)
    dtl_origin_array = dtl_points[16] if dtl_origin is None else np.asarray(dtl_origin)
    if fo_origin_array.shape != (2,) or dtl_origin_array.shape != (2,):
        raise ValueError("origins must each have shape (2,)")

    fo = fo_points - fo_origin_array.astype(np.float32)
    dtl = dtl_points - dtl_origin_array.astype(np.float32)
    if scale is None:
        scale = estimate_vertical_scale(fo_points, dtl_points)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    dtl = dtl * np.float32(scale)
    result = np.stack(
        (fo[:, 0], fo[:, 1], dtl[:, 0]),
        axis=1,
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def reconstruct_sequence(
    fo_frames: Iterable[tuple[int, Any]],
    dtl_frames: Iterable[tuple[int, Any]],
    *,
    fixed_origin: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Align views by frame index and reconstruct all valid frame pairs.

    Invalid/missing pairs are omitted. By default the ankles from the first
    valid pair remain the origins for every output frame, preserving motion.
    """
    fo_by_index = dict(fo_frames)
    dtl_by_index = dict(dtl_frames)
    common_indices = sorted(fo_by_index.keys() & dtl_by_index.keys())
    pairs = []
    valid_indices = []
    for frame_index in common_indices:
        try:
            fo = adapt_rtmpose_keypoints(fo_by_index[frame_index])
            dtl = adapt_rtmpose_keypoints(dtl_by_index[frame_index])
            estimate_vertical_scale(fo, dtl)
        except (TypeError, ValueError, IndexError):
            continue
        pairs.append((fo, dtl))
        valid_indices.append(int(frame_index))

    if not pairs:
        return np.empty((0, 17, 3), dtype=np.float32), np.asarray([], dtype=np.int64)
    fo_poses, dtl_poses = zip(*pairs)
    scale = estimate_vertical_scale(fo_poses, dtl_poses)
    points = [
        reconstruct_frame(
            fo,
            dtl,
            scale=scale,
            fo_origin=pairs[0][0][16] if fixed_origin else None,
            dtl_origin=pairs[0][1][16] if fixed_origin else None,
        )
        for fo, dtl in pairs
    ]
    return np.stack(points), np.asarray(valid_indices, dtype=np.int64)


def save_keypoints3d(path: str | Path, keypoints3d: Any, frame_indices: Any) -> Path:
    """Save the stable pseudo-3D interchange format."""
    points = np.asarray(keypoints3d, dtype=np.float32)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if points.ndim != 3 or points.shape[1:] != (17, 3):
        raise ValueError(f"expected keypoints3d shape (N, 17, 3), got {points.shape}")
    if indices.shape != (points.shape[0],):
        raise ValueError("frame_indices length must match keypoints3d")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        keypoints3d=points,
        frame_indices=indices,
        joint_names=COCO17_JOINT_NAMES,
    )
    return output


def load_infer_json(path: str | Path) -> list[tuple[int, Mapping[str, Any]]]:
    """Load the first detected person per frame from ``infer.py --json``."""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    frames = []
    for fallback_index, item in enumerate(data):
        if isinstance(item, Mapping) and "instances" in item:
            instances = item["instances"]
            if instances:
                frames.append((int(item.get("frame_id", fallback_index)), instances[0]))
        elif isinstance(item, Mapping) and "keypoints" in item:
            frames.append((fallback_index, item))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-view pseudo-3D COCO-17 reconstruction"
    )
    parser.add_argument("fo_json", type=Path, help="FO output from infer.py --json")
    parser.add_argument("dtl_json", type=Path, help="DTL output from infer.py --json")
    parser.add_argument("-o", "--output", type=Path, default=Path("keypoints3d.npz"))
    parser.add_argument(
        "--smooth-fps",
        type=float,
        help="enable 3-D SG smoothing using this source FPS (JSON has no FPS metadata)",
    )
    parser.add_argument(
        "--smooth-window-ms",
        type=float,
        help="SG window time span in milliseconds (default: 67; requires --smooth-fps)",
    )
    parser.add_argument(
        "--per-frame-origin",
        action="store_true",
        help="center every frame independently instead of retaining translation",
    )
    args = parser.parse_args()
    if args.smooth_window_ms is not None and args.smooth_fps is None:
        parser.error("--smooth-window-ms requires --smooth-fps")
    window_ms = (
        DEFAULT_WINDOW_MS if args.smooth_window_ms is None else args.smooth_window_ms
    )
    if args.smooth_fps is not None:
        window_frames = savgol_window_length(args.smooth_fps, window_ms)
    points, indices = reconstruct_sequence(
        load_infer_json(args.fo_json),
        load_infer_json(args.dtl_json),
        fixed_origin=not args.per_frame_origin,
    )
    if args.smooth_fps is not None:
        raw_path = args.output.with_name(f"{args.output.stem}_raw{args.output.suffix}")
        save_keypoints3d(raw_path, points, indices)
        points = smooth_keypoints3d(
            points, indices, fps=args.smooth_fps, window_ms=window_ms
        )
        print(f"Smoothed 3-D with quadratic Savitzky-Golay ({window_frames} frames)")
        print(f"Raw reconstruction: {raw_path}")
    save_keypoints3d(args.output, points, indices)
    print(f"Saved {len(indices)} frames to: {args.output}")


if __name__ == "__main__":
    main()
