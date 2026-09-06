"""Offline temporal smoothing for reconstructed COCO-17 coordinates.

The quadratic Savitzky-Golay filter operates on consecutive source frames.
Missing frames split the sequence; they are never filled or joined together.
Only NumPy is required.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_WINDOW_MS = 67.0


def savgol_window_length(fps: float, window_ms: float = DEFAULT_WINDOW_MS) -> int:
    """Return an odd window of at least five frames for a requested time span.

    The span is measured from the first to the last sample, so a five-frame
    window spans four frame intervals (about 67 ms at 60 FPS).
    """
    fps = float(fps)
    window_ms = float(window_ms)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("window_ms must be finite and positive")
    half_span = fps * (window_ms / 2000.0)
    if not math.isfinite(half_span):
        raise ValueError("fps and window_ms produce an unrepresentable window")
    return max(5, 2 * math.floor(half_span + 0.5) + 1)


def _smooth_segment(points: np.ndarray, window_length: int) -> np.ndarray:
    """Apply quadratic SG with polynomial fits at the two sequence edges."""
    half = window_length // 2
    # Centering and scaling keep the fit well conditioned for long windows.
    x = np.linspace(-1.0, 1.0, window_length)
    design = np.column_stack((np.ones(window_length), x, x * x))
    fit = np.linalg.pinv(design)
    result = np.empty_like(points)

    # At interior frames evaluate each local fit at its center (x = 0).
    windows = np.lib.stride_tricks.sliding_window_view(
        points, window_length, axis=0
    )
    result[half:-half] = np.einsum("ijkw,w->ijk", windows, fit[0])

    # Like SG's conventional 'interp' edge treatment, evaluate one fitted
    # polynomial over the first/last window. No padding or extra frames.
    flattened = points.reshape(len(points), -1)
    result[:half] = (
        design[:half] @ (fit @ flattened[:window_length])
    ).reshape(half, 17, 3)
    result[-half:] = (
        design[-half:] @ (fit @ flattened[-window_length:])
    ).reshape(half, 17, 3)
    return result


def smooth_keypoints3d(
    points: Any,
    frame_indices: Any,
    *,
    fps: float,
    window_ms: float = DEFAULT_WINDOW_MS,
) -> np.ndarray:
    """Return a smoothed ``(N, 17, 3) float32`` copy without changing inputs.

    A centered quadratic Savitzky-Golay filter smooths each joint coordinate
    independently. Gaps in strictly increasing integer ``frame_indices`` split
    the data into independent segments. Each segment uses the largest odd
    window no longer than both the requested window and the segment; segments
    shorter than five frames are unchanged. This is an offline filter that
    uses future frames, not a causal filter for live streaming.
    """
    window_length = savgol_window_length(fps, window_ms)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (17, 3):
        raise ValueError(f"expected keypoints3d shape (N, 17, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("keypoints3d contain NaN or infinity")
    indices = np.asarray(frame_indices)
    if indices.shape != (len(points),):
        raise ValueError("frame_indices must have shape (N,) matching keypoints3d")
    if len(indices) and not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("frame_indices must contain integers")
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("frame_indices must be strictly increasing")

    result = points.astype(np.float32, copy=True)
    if not len(points):
        return result
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(indices) != 1) + 1, [len(points)])
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment_length = int(stop - start)
        if segment_length < 5:
            continue
        segment_window = min(window_length, segment_length)
        if segment_window % 2 == 0:
            segment_window -= 1
        result[start:stop] = _smooth_segment(points[start:stop], segment_window)
    return result
