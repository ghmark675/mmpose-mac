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
