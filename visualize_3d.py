"""Create a dependency-free interactive HTML viewer for pseudo-3D NPZ files."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from keypoint_config import load_keypoint_config


SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def load_keypoints3d(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate the pseudo-3D NPZ schema without enabling pickle."""
    with np.load(path, allow_pickle=False) as data:
        required = {"keypoints3d", "frame_indices", "joint_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ is missing: {', '.join(sorted(missing))}")
        points = np.asarray(data["keypoints3d"], dtype=np.float32)
        indices = np.asarray(data["frame_indices"], dtype=np.int64)
        names = [str(name) for name in data["joint_names"].tolist()]
    if points.ndim != 3 or points.shape[1:] != (17, 3):
        raise ValueError(f"expected keypoints3d shape (N, 17, 3), got {points.shape}")
    if indices.shape != (points.shape[0],):
        raise ValueError("frame_indices length must match keypoints3d")
    if len(names) != 17:
        raise ValueError("joint_names must contain 17 names")
    if not np.isfinite(points).all():
        raise ValueError("keypoints3d contains NaN or infinity")
    if len(points) == 0:
        raise ValueError("keypoints3d contains no frames")
    return points, indices, names


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )


def build_comparison(points, indices, fo_json, dtl_json, fo_video, dtl_video,
                     *, per_frame_origin=False):
    import cv2
    from pseudo3d import adapt_rtmpose_keypoints, load_infer_json, estimate_vertical_scale

    sources = [dict(load_infer_json(path)) for path in (fo_json, dtl_json)]
    poses = [np.stack([adapt_rtmpose_keypoints(source[int(i)]) for i in indices])
             for source in sources]
    origins = [pose[:, 16] if per_frame_origin else np.repeat(pose[:1, 16], len(indices), axis=0)
               for pose in poses]
    scale = estimate_vertical_scale(*poses)
    projected = [points[:, :, :2] + origins[0][:, None],
                 points[:, :, [2, 1]] / scale + origins[1][:, None]]
    result = {}
    for name, path, original, projection in zip(
            ("fo", "dtl"), (fo_video, dtl_video), poses, projected):
        capture = cv2.VideoCapture(str(path))
        images = []
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot read video FPS: {path}")
            wanted = set(map(int, indices))
            frame_id = 0
            while frame_id <= int(indices[-1]):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Missing video frame {frame_id}: {path}")
                if frame_id in wanted:
                    height, width = image.shape[:2]
                    ratio = min(1, 960 / max(width, height))
                    image = cv2.resize(image, (round(width * ratio), round(height * ratio)))
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise ValueError(f"Cannot encode video frame {frame_id}: {path}")
                    images.append("data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"))
                frame_id += 1
        finally:
            capture.release()
        result[name] = dict(images=images, width=width, height=height, fps=fps,
                            original=original.tolist(), projected=projection.tolist())
    return result


def create_viewer_html(
    npz_path: str | Path, output: str | Path, config_path="keypoint_config.json",
    *, comparison=None
) -> Path:
    """Write a self-contained, offline HTML skeleton viewer."""
    points, frame_indices, joint_names = load_keypoints3d(npz_path)
    # Image coordinates point down. Flip Y once so the viewer uses Y-up.
    display_points = points.copy()
    display_points[:, :, 1] *= -1
    config = load_keypoint_config(config_path) or {}
    payload = {
        "points": display_points.tolist(),
        "comparison": comparison,
        "frameIndices": frame_indices.tolist(),
        "jointNames": joint_names,
        "bones": SKELETON,
        "source": Path(npz_path).name,
        "visible": config.get("visible_joint_indices", list(range(17))),
        "head": config.get("head_joint", "nose"),
        "headRadius": config.get("head_radius", 11),
        "boneColor": config.get("default_bone_color", "#55d6be"),
        "boneColors": config.get("bone_colors", {}),
    }
    html = _HTML_TEMPLATE.replace("__DATA__", _safe_json(payload))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pseudo-3D Skeleton Viewer</title>
<style>
:root{color-scheme:dark;font-family:ui-sans-serif,system-ui,-apple-system,sans-serif}
*{box-sizing:border-box}body{margin:0;background:#090d16;color:#e8eef9;overflow:hidden}
#view{display:block;width:100vw;height:100vh;cursor:grab;touch-action:none}
body.compare #view{width:50vw}#comparison{display:none;position:fixed;left:50%;right:0;top:48px;bottom:120px;padding:12px}
body.compare #comparison{display:flex;flex-direction:column;gap:12px}#photo{position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
#overlay{max-width:100%;max-height:100%;object-fit:contain}label{font-size:13px}#view.drag{cursor:grabbing}.panel{position:fixed;left:18px;right:18px;bottom:18px;
background:#111827e8;border:1px solid #334155;border-radius:12px;padding:12px 14px;
box-shadow:0 10px 35px #0008;backdrop-filter:blur(8px)}
.row{display:flex;align-items:center;gap:12px}.title{position:fixed;left:20px;top:16px;
font-weight:650;text-shadow:0 2px 8px #000}.hint{position:fixed;right:20px;top:16px;
color:#94a3b8;font-size:13px}button,select{border:1px solid #475569;border-radius:8px;
background:#1e293b;color:#e8eef9;padding:7px 11px;font:inherit}button{cursor:pointer}
button.active{background:#2563eb;border-color:#60a5fa}
input[type=range]{flex:1;min-width:80px}.status{min-width:185px;font-variant-numeric:tabular-nums;
font-size:13px;color:#cbd5e1}.source{font-size:12px;color:#64748b;margin-top:7px}
@media(max-width:620px){.hint{display:none}.panel{left:8px;right:8px;bottom:8px}.status{min-width:120px}}
</style></head><body>
<canvas id="view"></canvas><div class="title">Pseudo-3D Skeleton</div>
<div class="hint">拖动旋转 · 滚轮缩放 · 双击复位</div>
<section id="comparison"><div class="row"><select id="camera"><option value="fo">FO 原视频</option><option value="dtl">DTL 原视频</option></select>
<label><input type="checkbox" id="show3d" checked>3D 重投影（绿）</label><label><input type="checkbox" id="show2d">原始 2D（粉）</label></div>
<div id="photo"><canvas id="overlay"></canvas></div><span id="comparisonStatus" class="source"></span></section>
<div class="panel"><div class="row"><button id="play">▶ 播放</button><button id="prev">‹</button><button id="next">›</button>
<button class="view active" data-view="fo">FO</button><button class="view" data-view="dtl">DTL</button><button class="view" data-view="top">Top</button>
<input id="frame" type="range" min="0" value="0" step="1">
<span id="status" class="status"></span><select id="speed">
<option value="0.25">0.25×</option><option value="0.5">0.5×</option>
<option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option>
</select></div><div id="source" class="source"></div></div>
<script>
const data=__DATA__,canvas=document.querySelector('#view'),ctx=canvas.getContext('2d');
const slider=document.querySelector('#frame'),statusEl=document.querySelector('#status');
const playBtn=document.querySelector('#play'),speedEl=document.querySelector('#speed');
slider.max=data.points.length-1;document.querySelector('#source').textContent=data.source+' · '+data.points.length+' 帧';
const plotWidth=()=>data.comparison?innerWidth/2:innerWidth;
let frame=0,playing=false,last=0,acc=0,yaw=0,pitch=0,zoom=1,drag=false,px=0,py=0,view='fo';
const all=data.points.flat(),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
for(const p of all)for(let k=0;k<3;k++){if(p[k]<lo[k])lo[k]=p[k];if(p[k]>hi[k])hi[k]=p[k]}
const center=[0,1,2].map(k=>(lo[k]+hi[k])/2);let extent=1;
for(const p of all)for(let k=0;k<3;k++)extent=Math.max(extent,Math.abs(p[k]-center[k]));
function resize(){const d=devicePixelRatio||1;canvas.width=plotWidth()*d;canvas.height=innerHeight*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function project(p){let x=(p[0]-center[0])/extent,y=(p[1]-center[1])/extent,z=(p[2]-center[2])/extent;
 let x1,y1,z2;if(view==='fo'){x1=x;y1=y;z2=-z}else if(view==='dtl'){x1=z;y1=y;z2=x}else if(view==='top'){x1=-x;y1=z;z2=-y}else{
 let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),z1;x1=cy*x+sy*z;z1=-sy*x+cy*z;y1=cp*y-sp*z1;z2=sp*y+cp*z1}
 const plotHeight=Math.max(200,innerHeight-115);let s=Math.min(plotWidth(),plotHeight)*.38*zoom/(1+z2*.18);return [plotWidth()/2+x1*s,plotHeight/2-y1*s,z2]}
function draw(){ctx.clearRect(0,0,innerWidth,innerHeight);let grad=ctx.createRadialGradient(innerWidth/2,innerHeight/2,0,innerWidth/2,innerHeight/2,Math.max(innerWidth,innerHeight)*.7);
 grad.addColorStop(0,'#17213b');grad.addColorStop(1,'#090d16');ctx.fillStyle=grad;ctx.fillRect(0,0,innerWidth,innerHeight);
 drawAxes();const pts=data.points[frame].map(project);ctx.lineCap='round';ctx.lineWidth=4;
 const visible=new Set(data.visible),head=data.jointNames.indexOf(data.head);
 for(const [a,b] of data.bones){if(!visible.has(a)||!visible.has(b))continue;ctx.beginPath();ctx.moveTo(pts[a][0],pts[a][1]);ctx.lineTo(pts[b][0],pts[b][1]);ctx.strokeStyle=data.boneColors[`${a}-${b}`]||data.boneColors[`${b}-${a}`]||data.boneColor;ctx.stroke()}
 pts.map((p,i)=>({p,i})).filter(v=>visible.has(v.i)).sort((a,b)=>a.p[2]-b.p[2]).forEach(({p,i})=>{const r=i===head?data.headRadius:5;ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);if(i===head){const g=ctx.createRadialGradient(p[0]-r*.3,p[1]-r*.3,1,p[0],p[1],r);g.addColorStop(0,'#fff7cc');g.addColorStop(.35,'#fbbf24');g.addColorStop(1,'#b45309');ctx.fillStyle=g}else ctx.fillStyle='#f472b6';ctx.fill();ctx.strokeStyle='#fff9';ctx.lineWidth=1;ctx.stroke()});
 statusEl.textContent=`序列 ${frame+1}/${data.points.length} · 原始帧 ${data.frameIndices[frame]}`}
function drawAxes(){const length=extent*.55,origin=project([0,0,0]);
 const axes=[[[length,0,0],'#ef4444','X'],[[0,length,0],'#22c55e','Y'],[[0,0,length],'#3b82f6','Z']];
 ctx.lineWidth=2;ctx.font='bold 13px ui-sans-serif,system-ui';
 for(const [end,color,label] of axes){const p=project(end);ctx.beginPath();ctx.moveTo(origin[0],origin[1]);ctx.lineTo(p[0],p[1]);ctx.strokeStyle=color;ctx.stroke();ctx.fillStyle=color;ctx.fillText(label,p[0]+5,p[1]-5)}
 ctx.beginPath();ctx.arc(origin[0],origin[1],3,0,Math.PI*2);ctx.fillStyle='#e2e8f0';ctx.fill()}
const camera=document.querySelector('#camera'),overlay=document.querySelector('#overlay'),overlayCtx=overlay.getContext('2d');
const show3d=document.querySelector('#show3d'),show2d=document.querySelector('#show2d');
let imageRequest=0;
function drawComparison(){if(!data.comparison)return;const source=data.comparison[camera.value],index=frame,request=++imageRequest;
 const image=new Image();image.onload=()=>{if(request!==imageRequest)return;overlay.width=image.width;overlay.height=image.height;overlayCtx.drawImage(image,0,0);
 const sx=image.width/source.width,sy=image.height/source.height,visible=new Set(data.visible);
 function skeleton(points,color){overlayCtx.strokeStyle=color;overlayCtx.fillStyle=color;overlayCtx.lineWidth=2;
 for(const [a,b] of data.bones){if(!visible.has(a)||!visible.has(b))continue;overlayCtx.beginPath();overlayCtx.moveTo(points[a][0]*sx,points[a][1]*sy);overlayCtx.lineTo(points[b][0]*sx,points[b][1]*sy);overlayCtx.stroke()}
 points.forEach((p,i)=>{if(!visible.has(i))return;overlayCtx.beginPath();overlayCtx.arc(p[0]*sx,p[1]*sy,3,0,Math.PI*2);overlayCtx.fill()})}
 if(show2d.checked)skeleton(source.original[index],'#f472b6');if(show3d.checked)skeleton(source.projected[index],'#55d6be');
 document.querySelector('#comparisonStatus').textContent=`原始帧 ${data.frameIndices[index]} · ${(data.frameIndices[index]/source.fps).toFixed(3)} s · 当前算法近似回投`;
 };image.src=source.images[index]}
if(data.comparison){document.body.classList.add('compare');drawComparison()}
camera.onchange=show3d.onchange=show2d.onchange=drawComparison;
function setFrame(v){frame=Math.max(0,Math.min(data.points.length-1,v|0));slider.value=frame;draw();drawComparison()}
function toggle(){playing=!playing;playBtn.textContent=playing?'⏸ 暂停':'▶ 播放';last=performance.now();acc=0;if(playing)requestAnimationFrame(tick)}
function tick(t){if(!playing)return;acc+=(t-last)*Number(speedEl.value);last=t;const fps=data.comparison?data.comparison.fo.fps:30;let duration=()=>1000*(frame+1<data.points.length?data.frameIndices[frame+1]-data.frameIndices[frame]:1)/fps;while(acc>=duration()){acc-=duration();setFrame((frame+1)%data.points.length)}requestAnimationFrame(tick)}
document.querySelector('#prev').onclick=()=>{if(playing)toggle();setFrame(frame-1)};document.querySelector('#next').onclick=()=>{if(playing)toggle();setFrame(frame+1)};
playBtn.onclick=toggle;slider.oninput=()=>setFrame(Number(slider.value));
document.querySelectorAll('.view').forEach(b=>b.onclick=()=>{view=b.dataset.view;if(view==='fo'||view==='dtl'){camera.value=view;drawComparison()}document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x===b));draw()});
canvas.onpointerdown=e=>{if(view==='dtl'){yaw=Math.PI/2;pitch=0}else if(view==='top'){yaw=Math.PI;pitch=Math.PI/2}else if(view==='fo'){yaw=0;pitch=0}view=null;document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));drag=true;px=e.clientX;py=e.clientY;canvas.classList.add('drag');canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-px)*.008;pitch=Math.max(-Math.PI,Math.min(Math.PI,pitch+(e.clientY-py)*.008));px=e.clientX;py=e.clientY;draw()};
canvas.onpointerup=()=>{drag=false;canvas.classList.remove('drag')};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));draw()};
canvas.ondblclick=()=>{view='fo';yaw=0;pitch=0;zoom=1;document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.dataset.view==='fo'));draw()};addEventListener('resize',resize);resize();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pseudo-3D keypoints in a browser")
    parser.add_argument("input", type=Path, help="keypoints3d.npz")
    parser.add_argument("-o", "--output", type=Path, help="output HTML path")
    parser.add_argument("--config", type=Path, default=Path("keypoint_config.json"))
    parser.add_argument("--fo-json", type=Path)
    parser.add_argument("--dtl-json", type=Path)
    parser.add_argument("--fo-video", type=Path)
    parser.add_argument("--dtl-video", type=Path)
    parser.add_argument("--per-frame-origin", action="store_true")
    args = parser.parse_args()
    inputs = (args.fo_json, args.dtl_json, args.fo_video, args.dtl_video)
    if any(inputs) and not all(inputs):
        parser.error("reprojection requires both JSON files and both videos")
    comparison = None
    if all(inputs):
        points, indices, _ = load_keypoints3d(args.input)
        comparison = build_comparison(points, indices, *inputs,
                                      per_frame_origin=args.per_frame_origin)
    output = args.output or args.input.with_suffix(".html")
    output = create_viewer_html(args.input, output, args.config, comparison=comparison).resolve()
    print(f"Viewer saved to: {output}")


if __name__ == "__main__":
    main()
