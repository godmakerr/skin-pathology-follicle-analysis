import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# ================================================

import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torchvision import models, transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ================= Configuration =================
IMG_DIR   = "data/raw_images"
NPZ_DIR   = "data/sam_npz"
OUT_DIR   = "pred_focal"

# [修改] 1. 指向新的混合训练模型
MODEL_PATH= "checkpoints_cls/resnet18_hybrid_v2.pth" 

SAM_CKPT  = "/home/user1/models/sam/sam_vit_b_01ec64.pth"
SAM_TYPE  = "vit_b"

# [修改] 2. 更新映射表
# 这里的 Key 是你的真实 ID (CSV里的 ID)，Value 是模型的输出层索引 (0-6)
# 我们定义: D (毛球) 的真实 ID 为 4 (填补了原本空缺的 ID)
LABEL_MAP = {
    0: 0, # Others
    1: 1, # T
    2: 2, # V
    3: 3, # I
    6: 4, # Stela (模型第5个输出)
    7: 5, # Fibrosis (模型第6个输出)
    4: 6  # D/毛球 (模型第7个输出) -> [新增]
}
INV_MAP   = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP) # 现在是 7

# [修改] 3. 增加新类别的颜色
COLORS = {
    0: (160, 160, 160), # Others: Gray
    1: (0, 0, 255),     # T: Red
    2: (0, 255, 0),     # V: Green
    3: (255, 0, 0),     # I: Blue
    6: (0, 255, 255),   # Stela: Yellow
    7: (255, 0, 255),   # Fibrosis: Magenta
    4: (0, 165, 255)    # D/毛球: Orange [新增]
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= Filters (保持不变) =================
PAD = 16
MIN_AREA = 80
MAX_AREA = 200000
MAX_ASPECT = 6.0
ROUNDNESS_MIN = 0.02
TISSUE_MIN_COMPONENT = 5000
COVERAGE_MIN = 0.60         
GLOBAL_FRACTION_MAX = 0.35
BBOX_FRACTION_MAX = 0.30
EDGE_MIN_STRUCT = 0.015     

# ================= Model & Utils =================
inference_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(path):
    print(f"Loading model from {path}...")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # [关键] 必须初始化为 7 类，否则加载权重会报错尺寸不匹配
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model

def ensure_npz(img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    npz_path = os.path.join(NPZ_DIR, f"{name}.npz")
    if os.path.isfile(npz_path):
        return npz_path
    
    os.makedirs(NPZ_DIR, exist_ok=True)
    bgr = cv2.imread(img_path)
    if bgr is None: return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
    gen = SamAutomaticMaskGenerator(sam)
    with torch.no_grad():
        masks = gen.generate(rgb)
        
    H, W = rgb.shape[:2]
    masks = sorted(masks, key=lambda m: m["area"])
    M = len(masks)
    stack = np.zeros((M, H, W), np.uint8)
    boxes = np.zeros((M, 4), np.int32)
    
    for i, m in enumerate(masks):
        stack[i] = m["segmentation"].astype(np.uint8)
        boxes[i] = np.array(list(map(int, m["bbox"])), np.int32)
        
    np.savez_compressed(npz_path, masks=stack, bboxes=boxes, image_name=name, H=H, W=W)
    return npz_path

def crop_patch(img, x, y, w, h, pad=PAD):
    H, W = img.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return img[y0:y1, x0:x1]

def shape_ok(mask, bbox):
    x, y, w, h = bbox
    a = int(mask.sum())
    if a < MIN_AREA or a > MAX_AREA: return False
    aspect = max(w, h) / max(1, min(w, h))
    if aspect > MAX_ASPECT: return False
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        per = float(cv2.arcLength(cnts[0], True))
        if per > 0:
            roundness = 4.0 * np.pi * a / (per * per)
            if roundness < ROUNDNESS_MIN: return False
    return True

def build_tissue_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    t1 = (S > 25) & (V < 250)
    t2 = (A > 135)
    tissue = (t1 | t2).astype(np.uint8) * 255
    
    tissue = cv2.medianBlur(tissue, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    keep = np.zeros_like(tissue)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= TISSUE_MIN_COMPONENT:
            keep[labels == i] = 255
    return keep

def coverage_ok(mask, tissue_mask):
    if mask.shape != tissue_mask.shape:
        tissue_mask = cv2.resize(tissue_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (mask > 0)
    t = (tissue_mask > 0)
    inter = int(np.count_nonzero(m & t))
    total = int(np.count_nonzero(m))
    if total == 0: return False, 0.0
    cov = inter / total
    return cov >= COVERAGE_MIN, cov

def texture_score(patch_bgr):
    g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    edge_density = float(edges.mean())/255.0
    return edge_density

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(MODEL_PATH)
    
    # [修改] 4. 显示用的名称映射，加入 "4: D"
    name_map = {0: "Others", 1:"T", 2:"V", 3:"I", 6:"Stela", 7:"Fibrosis", 4:"D"}

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    
    for ip in imgs:
        if not ip.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            
        name = os.path.splitext(os.path.basename(ip))[0]
        npz_path = ensure_npz(ip)
        if not npz_path: continue
        
        data = np.load(npz_path)
        masks, boxes = data["masks"], data["bboxes"]
        
        base = cv2.imread(ip)
        if base is None: continue
        overlay = base.copy()
        
        tissue = build_tissue_mask(base)
        tissue_area = int((tissue > 0).sum())
        
        rows = []
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            
            if not shape_ok(masks[i], boxes[i]): continue
            
            mask_area = int(masks[i].sum())
            if tissue_area > 0 and (mask_area / tissue_area) > GLOBAL_FRACTION_MAX: continue
            
            ok_cov, cov = coverage_ok(masks[i], tissue)
            if not ok_cov: continue
            
            patch = crop_patch(base, x, y, w, h)
            if patch.size == 0: continue
            
            e_density = texture_score(patch)
            if e_density < EDGE_MIN_STRUCT: continue
            
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            input_tensor = inference_tfm(rgb_patch).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                score, pred_idx = torch.max(probs, dim=1)
            
            # 这里 pred_idx 是 0-6，通过 INV_MAP 转回 真实 ID (如 0, 6, 7, 4)
            final_id = INV_MAP[pred_idx.item()]
            score_val = score.item()
            
            display_name = name_map.get(final_id, str(final_id))
            color = COLORS.get(final_id, (128, 128, 128))

            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            text_str = f"{display_name}:{score_val:.2f}"
            cv2.putText(overlay, text_str, (x, max(0, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            rows.append({
                "image": name, "mask_idx": i, "x": x, "y": y, "w": w, "h": h,
                "pred_id": final_id, "pred_name": display_name, "score": round(score_val, 4),
                "cov_tissue": round(float(cov), 4), "edge_density": round(e_density, 4)
            })

        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_overlay.png"), overlay)
        if rows:
            with open(os.path.join(OUT_DIR, f"{name}_pred.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        print(f"[Done] {name}: Found {len(rows)} objects")

if __name__ == "__main__":
    main()