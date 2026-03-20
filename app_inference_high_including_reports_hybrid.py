# -*- coding: utf-8 -*-
import os
import time
import cv2
import gc
import threading
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from torchvision import models, transforms, ops
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image

# ================= Configuration =================

# Paths
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

# [修改] 1. 指向新的混合训练模型
PATCH_MODEL_PATH = "checkpoints_cls/resnet18_hybrid_v2.pth"

# 2. Global Scene Classifiers
GLOBAL_SLOPE_PATH = "checkpoints_global/global_slope.pth"
GLOBAL_LAYER_PATH = "checkpoints_global/global_layer.pth"

# 3. SAM Model (ViT-H)
SAM_CKPT      = "./models/sam_vit_h_4b8939.pth" 
SAM_TYPE      = "vit_h"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Memory & Threading Settings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# 全局排队锁
gpu_lock = threading.Lock()

# [修改] 2. 更新映射表，加入 ID 4 (D/毛球)
# Key 是真实 ID (CSV/Report用), Value 是模型输出索引 (0-6)
LABEL_MAP = {
    0: 0, 
    1: 1, 
    2: 2, 
    3: 3, 
    6: 4, # Stela -> Model Index 4
    7: 5, # Fibrosis -> Model Index 5
    4: 6  # D/毛球 -> Model Index 6 [新增]
}
INV_MAP   = {v: k for k, v in LABEL_MAP.items()}

# [修改] 更新显示名称和颜色
CLASS_NAMES = {
    0: "Others", 
    1: "T", 
    2: "V", 
    3: "I", 
    6: "Stela", 
    7: "Fibrosis", 
    4: "D (毛球)" # [新增]
}
SLOPE_NAMES = {0: "正常 (Straight)", 1: "倾斜 (Inclined)"}
LAYER_NAMES = {0: "真皮上部 (Upper)", 1: "真皮中部 (Mid)", 2: "真皮下部 (Lower)", 3: "皮下脂肪 (Subcutis)"}

# [修改] 增加颜色
COLORS = {
    0: (160, 160, 160), # Gray
    1: (0, 0, 255),     # Red
    2: (0, 255, 0),     # Green
    3: (255, 0, 0),     # Blue
    6: (0, 255, 255),   # Yellow
    7: (255, 0, 255),   # Magenta
    4: (0, 165, 255)    # Orange (毛球) [新增]
}

# Filters
PAD = 16
MIN_AREA = 80
MAX_AREA = 200000
MAX_ASPECT = 6.0
ROUNDNESS_MIN = 0.02
TISSUE_MIN_COMPONENT = 5000
COVERAGE_MIN = 0.60
GLOBAL_FRACTION_MAX = 0.35
EDGE_MIN_STRUCT = 0.015
NMS_IOU_THRESH = 0.45 

# ================= Global Models =================
print(">>> Initializing Models... Please wait.")

# 1. Load Patch Classifier
patch_model = models.resnet18(weights=None)
# [修改] 确保全连接层是 7 类 (len(LABEL_MAP) 现在是 7)
patch_model.fc = nn.Linear(patch_model.fc.in_features, len(LABEL_MAP))
try:
    patch_model.load_state_dict(torch.load(PATCH_MODEL_PATH, map_location=DEVICE))
    patch_model.to(DEVICE).eval()
    print(f"[OK] Patch Classifier loaded (Classes: {len(LABEL_MAP)}).")
except Exception as e:
    print(f"[ERR] Patch Classifier load failed: {e}")
    patch_model = None

# 2. Load Global Slope Model
slope_model = models.resnet18(weights=None)
slope_model.fc = nn.Linear(slope_model.fc.in_features, 2)
try:
    slope_model.load_state_dict(torch.load(GLOBAL_SLOPE_PATH, map_location=DEVICE))
    slope_model.to(DEVICE).eval()
    print("[OK] Global Slope Model loaded.")
except:
    print(f"[WARN] Global Slope Model not found, skipping.")
    slope_model = None

# 3. Load Global Layer Model
layer_model = models.resnet18(weights=None)
layer_model.fc = nn.Linear(layer_model.fc.in_features, 4)
try:
    layer_model.load_state_dict(torch.load(GLOBAL_LAYER_PATH, map_location=DEVICE))
    layer_model.to(DEVICE).eval()
    print("[OK] Global Layer Model loaded.")
except:
    print(f"[WARN] Global Layer Model not found, skipping.")
    layer_model = None

# 4. Load SAM
try:
    print(f"Loading SAM ({SAM_TYPE}) on {DEVICE}...")
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam, points_per_side=64, points_per_batch=16,
        pred_iou_thresh=0.86, stability_score_thresh=0.92,
        min_mask_region_area=80, crop_n_layers=0,
    )
    print(f"[OK] SAM Loaded.")
except Exception as e:
    print(f"[ERR] SAM load failed: {e}")
    mask_generator = None

# Transforms
patch_tfm = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
scene_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= Helper Functions =================

def build_tissue_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    t1 = (S > 25) & (V < 250); t2 = (A > 135)
    tissue = (t1 | t2).astype(np.uint8) * 255
    tissue = cv2.medianBlur(tissue, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    keep = np.zeros_like(tissue)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= TISSUE_MIN_COMPONENT: keep[labels == i] = 255
    return keep

def crop_patch(img, x, y, w, h, pad=PAD):
    H, W = img.shape[:2]
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return img[y0:y1, x0:x1]

def compute_iou_and_iomin(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2
    xx1 = max(b1_x1, b2_x1); yy1 = max(b1_y1, b2_y1)
    xx2 = min(b1_x2, b2_x2); yy2 = min(b1_y2, b2_y2)
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
    inter = w * h
    area1 = w1 * h1; area2 = w2 * h2
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    io_min = inter / (min(area1, area2) + 1e-6)
    return iou, io_min

def apply_custom_nms(candidates):
    if not candidates: return []
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    keep = []
    
    # [修改] 3. 将 ID 4 (毛球) 加入主要结构列表
    # 这些结构会抑制 "Others" (ID 0)
    STRUCTURE_IDS = [1, 2, 3, 6, 7, 4] 
    
    suppressed = [False] * len(candidates)
    
    for i in range(len(candidates)):
        if suppressed[i]: continue
        box_a = candidates[i]
        keep.append(box_a)
        for j in range(i + 1, len(candidates)):
            if suppressed[j]: continue
            box_b = candidates[j]
            iou, io_min = compute_iou_and_iomin(box_a['bbox'], box_b['bbox'])
            is_overlap = iou > 0.1 or io_min > 0.3
            if is_overlap:
                if box_a['class_id'] in STRUCTURE_IDS and box_b['class_id'] == 0:
                    suppressed[j] = True; continue
            if box_a['class_id'] == box_b['class_id']:
                if io_min > 0.6 or iou > 0.45: suppressed[j] = True
            elif box_a['class_id'] in STRUCTURE_IDS and box_b['class_id'] in STRUCTURE_IDS:
                if iou > 0.45: suppressed[j] = True
    return keep

@torch.no_grad()
def process_image(img_path, filename):
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    bgr = cv2.imread(img_path)
    if bgr is None: return None, {}, 0, "Unknown", "Unknown"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Global Scene
    slope_res, layer_res = "未检测", "未检测"
    pil_img = Image.fromarray(rgb)
    scene_tensor = scene_tfm(pil_img).unsqueeze(0).to(DEVICE)
    
    if slope_model:
        slope_res = SLOPE_NAMES.get(torch.argmax(slope_model(scene_tensor), 1).item(), "Unknown")
    if layer_model:
        layer_res = LAYER_NAMES.get(torch.argmax(layer_model(scene_tensor), 1).item(), "Unknown")

    # 2. SAM
    print(f"Running SAM for {filename}...")
    masks = mask_generator.generate(rgb)
    
    tissue_mask = build_tissue_mask(bgr)
    tissue_area = int((tissue_mask > 0).sum())
    candidates = []
    
    for m in masks:
        x, y, w, h = map(int, m["bbox"])
        mask_area = int(m["area"])
        mask_arr = m["segmentation"].astype(np.uint8)
        
        should_keep = True
        if mask_area < MIN_AREA or mask_area > MAX_AREA: should_keep = False
        if should_keep and (max(w,h)/max(1,min(w,h))) > MAX_ASPECT: should_keep = False
        if should_keep:
            cnts, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts: should_keep = False
            elif float(cv2.arcLength(cnts[0], True)) > 0:
                if (4.0 * np.pi * mask_area / (float(cv2.arcLength(cnts[0], True))**2)) < ROUNDNESS_MIN: should_keep = False
        if should_keep and tissue_area > 0 and (mask_area / tissue_area) > GLOBAL_FRACTION_MAX: should_keep = False
        if should_keep:
            tm_resized = cv2.resize(tissue_mask, (mask_arr.shape[1], mask_arr.shape[0]), interpolation=cv2.INTER_NEAREST) if mask_arr.shape != tissue_mask.shape else tissue_mask
            if int(np.count_nonzero((mask_arr > 0) & (tm_resized > 0))) / mask_area < COVERAGE_MIN: should_keep = False
        
        m["segmentation"] = None; m["point_coords"] = None; m["crop_box"] = None
        if not should_keep: continue

        patch = crop_patch(bgr, x, y, w, h)
        if patch.size == 0: continue
        if (float(cv2.Canny(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 50, 150).mean())/255.0) < EDGE_MIN_STRUCT: continue

        # Patch Classification
        patch_input = patch_tfm(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        logits = patch_model(patch_input)
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)
        
        # [逻辑] 将模型索引转回真实 ID (例如 6 -> 4)
        final_id = INV_MAP[pred_idx.item()]
        nms_score = score.item() + (1.0 if final_id != 0 else 0) 
        
        candidates.append({'bbox': [x, y, w, h], 'class_id': final_id, 'score': nms_score, 'display_score': score.item()})

    del masks, tissue_mask
    
    # 3. NMS
    final_detections = apply_custom_nms(candidates)
    
    # 4. Draw & Stats
    overlay = bgr.copy()
    # [修改] 4. 初始化 stats 增加 Key 4
    stats = {1:0, 2:0, 3:0, 6:0, 7:0, 4:0, 0:0, 'total_follicles':0}
    
    for det in final_detections:
        x, y, w, h = det['bbox']
        cid = det['class_id']
        stats[cid] += 1
        
        # [修改] 计算总毛囊数时包括 ID 4
        if cid in [1, 2, 3, 6, 4]: 
            stats['total_follicles'] += 1
        
        color = COLORS.get(cid, (255, 255, 255))
        name = CLASS_NAMES.get(cid, "UNK")
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
        font = 0.4 if cid == 0 else 0.5
        cv2.putText(overlay, f"{name} {det['display_score']:.2f}", (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1 if cid==0 else 2)

    res_filename = "res_" + filename
    cv2.imwrite(os.path.join(RESULT_FOLDER, res_filename), overlay)
    
    del bgr, rgb, overlay, candidates
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return res_filename, stats, len(final_detections), slope_res, layer_res

# ================= Web App =================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Pathology AI Diagnosis</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .upload-box { border: 2px dashed #3498db; padding: 30px; text-align: center; border-radius: 8px; margin-bottom: 20px; background: #ecf0f1; }
        input[type="file"] { margin: 10px 0; }
        button { background-color: #3498db; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; transition: 0.3s; }
        button:hover { background-color: #2980b9; }
        
        .status-bar {
            padding: 10px; margin-bottom: 15px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 14px;
            transition: all 0.5s ease;
        }
        .status-idle { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; } 
        .status-busy { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; } 
        
        .loading { display: none; color: #e67e22; font-weight: bold; margin-top: 10px; }
        
        .result-container { display: flex; gap: 20px; margin-top: 30px; }
        .img-column { flex: 2; }
        .report-column { flex: 1; min-width: 300px; }
        .img-card { border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #fafafa; margin-bottom: 20px; }
        .img-card h3 { text-align: center; margin-top: 0; }
        img { width: 100%; height: auto; border-radius: 4px; }
        .report-box { background: #fff; border: 1px solid #ccc; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); font-family: 'SimSun', 'Songti SC', serif; }
        .report-title { text-align: center; font-size: 18px; font-weight: bold; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 15px; }
        .report-content { font-size: 15px; line-height: 1.8; }
        .count-highlight { font-weight: bold; color: #d35400; }
        .scene-highlight { font-weight: bold; color: #2980b9; }
        .legend { margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; font-size: 12px; }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    </style>
    <script>
        function showLoading() {
            clearInterval(statusInterval);
            document.getElementById('loading-msg').style.display = 'block';
            const statusEl = document.getElementById('server-status');
            statusEl.className = 'status-bar status-busy';
            statusEl.innerHTML = '正在上传并排队中，请不要关闭页面...';
            document.getElementById('upload-btn').disabled = true;
            document.getElementById('upload-btn').innerText = '处理中...';
        }
        function checkServerStatus() {
            fetch('/status').then(response => response.json()).then(data => {
                const statusEl = document.getElementById('server-status');
                if (data.is_busy) {
                    statusEl.className = 'status-bar status-busy';
                    statusEl.innerHTML = '当前服务器繁忙：有其他医生正在进行分析，您可以提交，但需要排队等待。';
                } else {
                    statusEl.className = 'status-bar status-idle';
                    statusEl.innerHTML = '服务器空闲：可以立即进行分析。';
                }
            }).catch(err => console.log('Status check failed', err));
        }
        var statusInterval = setInterval(checkServerStatus, 2000);
        window.onload = checkServerStatus;
    </script>
</head>
<body>
    <div class="container">
        <h1>皮肤病理结构 AI 辅助诊断系统</h1>
        
        <div class="upload-box">
            <div id="server-status" class="status-bar status-idle">正在连接服务器状态...</div>
            <form method="post" enctype="multipart/form-data" onsubmit="showLoading()">
                <h3>上传横切面病理图像</h3>
                <input type="file" name="file" accept=".jpg,.jpeg,.png,.tif" required>
                <br><br>
                <button type="submit" id="upload-btn">开始分析</button>
                <div id="loading-msg" class="loading">正在排队进行 ViT-H 分析 (多用户并发保护开启)，请稍候...</div>
            </form>
        </div>

        {% if result_image %}
        <div class="result-container">
            <div class="img-column">
                <div class="legend">
                    <div class="legend-item"><span class="dot" style="background:red;"></span> T (终毛)</div>
                    <div class="legend-item"><span class="dot" style="background:green;"></span> V (毳毛)</div>
                    <div class="legend-item"><span class="dot" style="background:blue;"></span> I (未定)</div>
                    <div class="legend-item"><span class="dot" style="background:yellow;"></span> Stela</div>
                    <div class="legend-item"><span class="dot" style="background:magenta;"></span> Fibrosis</div>
                    <!-- [修改] 增加前端图例 -->
                    <div class="legend-item"><span class="dot" style="background:orange;"></span> D (毛球)</div>
                    <div class="legend-item"><span class="dot" style="background:gray;"></span> Others</div>
                </div>
                <div class="img-card">
                    <h3>AI 识别结果</h3>
                    <a href="{{ url_for('static_files', filename='results/' + result_image) }}" target="_blank">
                        <img src="{{ url_for('static_files', filename='results/' + result_image) }}" alt="Result">
                    </a>
                </div>
                <div class="img-card">
                    <h3>原始图像</h3>
                    <img src="{{ url_for('static_files', filename='uploads/' + original_image) }}" alt="Original">
                </div>
            </div>
            <div class="report-column">
                <div class="report-box">
                    <div class="report-title">病理切片 AI 辅助读片报告</div>
                    <div class="report-content">
                        <b>基本信息：</b><br>
                        标本类型：横切面<br>
                        切片状态：<span class="scene-highlight">{{ slope_info }}</span><br>
                        视野层次：<span class="scene-highlight">{{ layer_info }}</span><br>
                        <hr>
                        <b>毛囊结构计数统计：</b><br>
                        本视野内共检测到毛囊相关结构 <span class="count-highlight">{{ stats.total_follicles }}</span> 个。<br>
                        <ul>
                            <li><b>生长期终毛 (T):</b> <span class="count-highlight">{{ stats[1] }}</span> 个</li>
                            <li><b>毳毛 (V):</b> <span class="count-highlight">{{ stats[2] }}</span> 个</li>
                            <li><b>退行期/微型化 (Stela):</b> <span class="count-highlight">{{ stats[6] }}</span> 个</li>
                            <li><b>纤维化束 (Fibrosis):</b> <span class="count-highlight">{{ stats[7] }}</span> 个</li>
                            <!-- [修改] 增加报告条目 -->
                            <li><b>毛球结构 (D):</b> <span class="count-highlight">{{ stats[4] }}</span> 个</li>
                            <li><b>未定类毛囊 (I):</b> <span class="count-highlight">{{ stats[3] }}</span> 个</li>
                        </ul>
                        <hr>
                        <b>其他结构 (Others):</b> {{ stats[0] }} 个<br>
                        <br>
                        <b>辅助诊断建议：</b><br>
                        {% if stats[6] > 2 or stats[7] > 0 or stats[2] > stats[1] %}
                        切片显示有微型化毛囊或纤维化迹象，建议结合临床排除雄激素性脱发或瘢痕性脱发可能。
                        {% else %}
                        毛囊结构分布相对正常，未见明显微型化或纤维化主导特征，请结合临床。
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 20px;"><a href="/">分析下一张图片</a></div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        
        if gpu_lock.acquire(blocking=True):
            try:
                filename = secure_filename(file.filename)
                ts = int(time.time()); filename = f"{ts}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                res_filename, stats, total_count, slope_res, layer_res = process_image(filepath, filename)
                
                if res_filename:
                    return render_template_string(HTML_TEMPLATE, 
                                                  original_image=filename, result_image=res_filename,
                                                  stats=stats, count=total_count,
                                                  slope_info=slope_res, layer_info=layer_res)
                else: return "Error processing image."
            except Exception as e:
                print(f"Error: {e}")
                return "Server Error."
            finally:
                gpu_lock.release()
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        else:
            return "Server is busy, please try again later."
    
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def system_status():
    is_busy = gpu_lock.locked()
    return jsonify({"is_busy": is_busy})

@app.route('/static/<path:filename>')
def static_files(filename): return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)