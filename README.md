# mmpose-mac

RTMDet and RTMPose ONNX inference for images and videos.

## Setup

```bash
conda activate mmpose
pip install numpy opencv-python onnxruntime
```

Place the models in `models/`:

```text
models/rtmdet_m_person.onnx
models/rtmpose_l_body8_384x288.onnx
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

## Two-view pseudo-3D

Run the existing inference independently for the front (FO) and down-the-line
(DTL) videos, then reconstruct matching frames:

```bash
python infer.py fo.mp4 --json fo.json
python infer.py dtl.mp4 --json dtl.json
python pseudo3d.py fo.json dtl.json -o keypoints3d.npz
```

The default fixes each view's right ankle from the first valid frame, retaining
whole-body translation. Use `--per-frame-origin` to center every frame at its
current right ankle instead. `keypoints3d.npz` contains `keypoints3d`
(`N x 17 x 3`, float32), original `frame_indices`, and COCO-17 `joint_names`.

Python API:

```python
from pseudo3d import load_infer_json, reconstruct_sequence, save_keypoints3d

points, frame_ids = reconstruct_sequence(
    load_infer_json("fo.json"), load_infer_json("dtl.json")
)
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
