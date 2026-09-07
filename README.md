# mmpose-mac

RTMDet and RTMPose ONNX inference for images and videos, with optional MotionBERT 3-D lifting.

## Setup

```bash
conda activate mmpose
pip install numpy opencv-python onnxruntime
```

Place the models in `models/`:

```text
models/rtmdet_m_person.onnx
models/rtmpose_l_body8_384x288.onnx
models/motionbert_h36m.onnx  # for the MotionBERT pipeline
```

## Usage

```bash
python infer.py input.jpg
python infer.py input.mp4 -o output.mp4
python infer.py input.jpg --json
python infer.py input.mp4 --json keypoints.json
```

If `keypoint_config.json` exists, JSON export keeps only its `visible_joints`.
The included local config keeps the nose and all body joints while removing
eyes and ears. It is ignored by Git so each machine can customize it.

## Single-view MotionBERT 3-D

```bash
python pipeline_motionbert.py input.jpg -o output_motionbert
python pipeline_motionbert.py input.mp4 -o output_motionbert
python pipeline_motionbert.py input.mp4 -o output_motionbert --save-pose-video
```

Open `output_motionbert/index.html` to rotate, zoom, and play the 3-D skeleton
at the source video FPS. Outputs also include full 2-D detections in
`keypoints2d.json`, and 3-D poses in `keypoints3d.json` and `keypoints3d.npz`.
Images additionally produce `pose.jpg`; `--save-pose-video` adds `pose.mp4`.
The NPZ can also be opened with `python visualize_3d.py output_motionbert/keypoints3d.npz`.

This pipeline targets one main person: it selects the largest detection at the
first detected frame, then matches boxes by IoU (above 0.1). Unmatched frames
reuse the last 2-D pose and have `detected=false` in the 3-D outputs. Leading
frames without a person are skipped; original frame IDs are retained. This
simple matching works best with a single person continuously in view; it does
not provide identity tracking through crossings or large movements.

MotionBERT runs directly in ONNX Runtime on CPU, without installing PyTorch or
MMPose. It uses the full COCO-17 pose, converts it to H36M-17, and infers each
frame from a centered 243-frame window with repeated boundary frames. A single
image is repeated 243 times. Bbox normalization follows the supplied MMPose
reference; `--no-bbox-norm` disables it. Use `--lift-model` to select another
compatible ONNX file. This is an offline pipeline with one MotionBERT inference
per output frame.

The 3-D arrays are `N x 17 x 3`, in H36M joint order, relative to the pelvis,
using camera axes (X right, Y down, Z away from the camera), recorded as
`coordinate_system="camera"`. Values are in meters using the
reference's default scale factor of 4; they are monocular estimates without
camera calibration or global body translation. For FO input, the viewer flips
Y and Z to match the two-view fusion viewer: Y upward, Z toward the FO camera.
DTL and Top therefore use the same axis directions as fusion. This aligns view
directions; single-view depth estimates can still differ from two-view poses.
The viewer uses the H36M skeleton independently of the local COCO keypoint config.
Older MotionBERT NPZ files are recognized by their H36M joint names; regenerate
their HTML without rerunning inference:

```bash
python visualize_3d.py results/hj_mb/keypoints3d.npz -o results/hj_mb/index.html
```

## Two-view pseudo-3D

### One-command pipeline (recommended)

Given frame-aligned front-on (FO) and down-the-line (DTL) videos, run:

```bash
python pipeline_3d.py fo.mp4 dtl.mp4 -o output_3d
```

Open `output_3d/index.html` in any modern browser. The pipeline loads the ONNX
models once, infers both videos, reconstructs matching frame IDs, and creates a
self-contained offline viewer. It also keeps the intermediate 2-D JSON files
and both `keypoints3d_raw.npz` (before smoothing) and `keypoints3d.npz`
(the final result used by the viewer and reprojection) for inspection or reuse.

The pipeline applies quadratic Savitzky-Golay temporal smoothing after 3-D
reconstruction by default. It uses NumPy, with no additional dependencies.
The default window spans approximately 67 ms, converted using the original FO
video FPS: 5 frames at 60 FPS, 9 at 120 FPS, and 17 at 240 FPS. Windows are odd
and at least 5 frames, so the minimum spans more time at lower frame rates.
Filtering is centered in time; sequence ends use local polynomial fits.
Larger windows smooth more strongly and can reduce fast-motion detail.

```bash
python pipeline_3d.py fo.mp4 dtl.mp4 -o output_3d --smooth-window-ms 50
python pipeline_3d.py fo.mp4 dtl.mp4 -o output_3d_raw --no-smooth
```

Missing frame IDs split the sequence into independent runs. Each run uses a
smaller odd window when needed; runs shorter than 5 frames stay unchanged.
No missing frames are filled, and no 2-D smoothing, confidence gating, or
outlier correction is performed. Original 2-D JSON and overlays are retained.
With `--no-smooth`, both NPZ files contain the raw reconstruction.

Add `--save-pose-videos` to keep annotated FO/DTL MP4 files, or
`--per-frame-origin` to remove whole-body translation. The two input videos
must already be synchronized: frame 0 in FO is paired with frame 0 in DTL.

Run `python pipeline_3d.py --help` for model, threshold, config, and output
options.

### Manual stages

Run the existing inference independently for the front (FO) and down-the-line
(DTL) videos, then reconstruct matching frames:

```bash
python infer.py fo.mp4 --json fo.json
python infer.py dtl.mp4 --json dtl.json
python pseudo3d.py fo.json dtl.json -o keypoints3d.npz
```

Manual reconstruction leaves smoothing off unless `--smooth-fps` is supplied,
because the JSON files do not contain source FPS. To smooth existing inference
results without running the models again, supply the original source FPS:

```bash
python pseudo3d.py fo.json dtl.json -o keypoints3d.npz --smooth-fps 240
```

This also saves `keypoints3d_raw.npz`. Add `--smooth-window-ms 50` to adjust the
window. Use the actual input video FPS, rather than a desired playback rate.

The default fixes each view's right ankle from the first valid frame, retaining
whole-body translation. Use `--per-frame-origin` to center every frame at its
current right ankle instead. `keypoints3d.npz` contains `keypoints3d`
(`N x 17 x 3`, float32), original `frame_indices`, and COCO-17 `joint_names`.

Python API:

```python
from pseudo3d import load_infer_json, reconstruct_sequence, save_keypoints3d
from smoothing import smooth_keypoints3d

points, frame_ids = reconstruct_sequence(
    load_infer_json("fo.json"), load_infer_json("dtl.json")
)
save_keypoints3d("keypoints3d_raw.npz", points, frame_ids)
# Optional: use the actual source FPS. reconstruct_sequence itself stays raw.
points = smooth_keypoints3d(points, frame_ids, fps=240, window_ms=67)
save_keypoints3d("keypoints3d.npz", points, frame_ids)
```

Visualize the result in an offline browser viewer (no extra dependencies):

```bash
python visualize_3d.py keypoints3d.npz
```

This writes `keypoints3d.html` next to the NPZ. Open the HTML in a browser,
drag to rotate, use the mouse wheel to zoom, and use the playback controls to
inspect the sequence and its original frame indices. FO, DTL, and Top buttons
switch to fixed orthogonal views. Use `-o viewer.html` to select another output
path.

Viewer colors use `default_bone_color`; individual bones can override it in
`bone_colors` with COCO index pairs such as `"5-6": "#60a5fa"`. The head joint
and size are controlled by `head_joint` and `head_radius` in the same config.

### 3D reprojection check

The pipeline viewer includes synchronized FO/DTL source frames, a green 3D
reprojection overlay, and an optional pink original 2D overlay. Use ‹ / › to
step through valid frame pairs. Playback uses FO source FPS and original frame
spacing; skipped reconstruction frames are omitted. Inputs must be frame-aligned.
JPEG previews (up to 960 pixels per side, without audio) are embedded so the HTML
works offline; long clips produce larger files.

To add the comparison to an existing reconstruction:

```bash
python visualize_3d.py keypoints3d.npz -o comparison.html \
  --fo-json fo.json --dtl-json dtl.json --fo-video fo.mov --dtl-video dtl.mov
```

Add `--per-frame-origin` only if the reconstruction used that option. The
approximate projection restores the current pseudo-3D algorithm's origins and
DTL scale from its original JSON inputs; it is not a calibrated camera model.

Reconstruction keeps X/Y from FO and uses scaled DTL horizontal coordinates for
Z. A single scale is fitted over all valid paired frames using corresponding
body-joint vertical differences (least squares, excluding spans below 20% of
each view's body height and opposite signs). This assumes aligned vertical
axes, orthographic views, fixed camera scale, and no nonuniform image resizing.
Y is not averaged. Regenerate old NPZ files before creating reprojection HTML
with this version; the previous reconstruction used different scale/Y rules.
