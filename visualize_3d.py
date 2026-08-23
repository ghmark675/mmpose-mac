"""Create a dependency-free interactive HTML viewer for pseudo-3D NPZ files."""

from __future__ import annotations

import argparse
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


def create_viewer_html(
    npz_path: str | Path, output: str | Path, config_path="keypoint_config.json"
) -> Path:
    """Write a self-contained, offline HTML skeleton viewer."""
    points, frame_indices, joint_names = load_keypoints3d(npz_path)
    # Image coordinates point down. Flip Y once so the viewer uses Y-up.
    display_points = points.copy()
    display_points[:, :, 1] *= -1
    config = load_keypoint_config(config_path) or {}
    payload = {
        "points": display_points.tolist(),
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
#view.drag{cursor:grabbing}.panel{position:fixed;left:18px;right:18px;bottom:18px;
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
<div class="panel"><div class="row"><button id="play">▶ 播放</button>
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
let frame=0,playing=false,last=0,acc=0,yaw=0,pitch=0,zoom=1,drag=false,px=0,py=0,view='fo';
const all=data.points.flat(),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
for(const p of all)for(let k=0;k<3;k++){if(p[k]<lo[k])lo[k]=p[k];if(p[k]>hi[k])hi[k]=p[k]}
const center=[0,1,2].map(k=>(lo[k]+hi[k])/2);let extent=1;
for(const p of all)for(let k=0;k<3;k++)extent=Math.max(extent,Math.abs(p[k]-center[k]));
function resize(){const d=devicePixelRatio||1;canvas.width=innerWidth*d;canvas.height=innerHeight*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function project(p){let x=(p[0]-center[0])/extent,y=(p[1]-center[1])/extent,z=(p[2]-center[2])/extent;
 let x1,y1,z2;if(view==='fo'){x1=x;y1=y;z2=-z}else if(view==='dtl'){x1=z;y1=y;z2=x}else if(view==='top'){x1=-x;y1=z;z2=-y}else{
 let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),z1;x1=cy*x+sy*z;z1=-sy*x+cy*z;y1=cp*y-sp*z1;z2=sp*y+cp*z1}
 const plotHeight=Math.max(200,innerHeight-115);let s=Math.min(innerWidth,plotHeight)*.38*zoom/(1+z2*.18);return [innerWidth/2+x1*s,plotHeight/2-y1*s,z2]}
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
function setFrame(v){frame=Math.max(0,Math.min(data.points.length-1,v|0));slider.value=frame;draw()}
function toggle(){playing=!playing;playBtn.textContent=playing?'⏸ 暂停':'▶ 播放';last=performance.now();acc=0;if(playing)requestAnimationFrame(tick)}
function tick(t){if(!playing)return;acc+=(t-last)*Number(speedEl.value);last=t;while(acc>=1000/30){setFrame((frame+1)%data.points.length);acc-=1000/30}requestAnimationFrame(tick)}
playBtn.onclick=toggle;slider.oninput=()=>setFrame(Number(slider.value));
document.querySelectorAll('.view').forEach(b=>b.onclick=()=>{view=b.dataset.view;document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x===b));draw()});
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
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".html")
    output = create_viewer_html(args.input, output, args.config).resolve()
    print(f"Viewer saved to: {output}")


if __name__ == "__main__":
    main()
