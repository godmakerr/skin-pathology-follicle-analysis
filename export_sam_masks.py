# export_sam_masks.py
import os, glob
import numpy as np
import cv2
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ================= Configuration =================
IMG_DIR = "data/raw_images2.0"
NPZ_DIR = "data/sam_npz2.0"
VIS_DIR = "data/sam_vis2.0"  
os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

SAM_CKPT = "/home/user1/models/sam/sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"

MAX_LONG_SIDE = None  

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p): return os.path.splitext(p)[1].lower() in IMG_EXTS

def maybe_downscale(rgb):
    if MAX_LONG_SIDE is None: 
        return rgb, 1.0
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_LONG_SIDE:
        return rgb, 1.0
    scale = MAX_LONG_SIDE / float(long_side)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rgb_small, scale

def vis_overlay(rgb, masks_stack):
    overlay = rgb.astype(np.float32).copy()
    np.random.seed(42)
    colors = np.random.rand(masks_stack.shape[0], 3) * 255
    
    for i in range(masks_stack.shape[0]):
        m = masks_stack[i].astype(bool)
        c = colors[i]
        overlay[m] = overlay[m] * 0.5 + c * 0.5
        
    vis = cv2.addWeighted(rgb.astype(np.uint8), 0.6, overlay.astype(np.uint8), 0.4, 0)
    return vis

def to_xywh(bbox):
    x, y, w, h = bbox
    return [int(x), int(y), int(w), int(h)]

def load_sam_on(device:str):
    print(f"Loading SAM ({SAM_TYPE}) on {device}...")
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT)
    sam.to(device=device)
    
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        
        points_per_batch=16,
        
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        
        min_mask_region_area=80,
        
        crop_n_layers=0,
    )
    return sam, gen

def export_one_image(img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    
    npz_out = os.path.join(NPZ_DIR, f"{name}.npz")
    vis_out = os.path.join(VIS_DIR, f"{name}.png")
    
    if os.path.exists(npz_out) and os.path.exists(vis_out):
        print(f"[skip-exist] {name}")
        return
        
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("[skip] cannot read:", img_path); return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]

    img_in, scale = maybe_downscale(img_rgb)

    device_tried = []
    masks_list = None
    
    devices_to_try = ["cuda"] if torch.cuda.is_available() else []
    devices_to_try.append("cpu")

    for device in devices_to_try:
        if device in device_tried: continue
        device_tried.append(device)
        
        try:
            with torch.no_grad():
                sam, gen = load_sam_on(device)
                print(f"Running inference on {name} ({img_in.shape[1]}x{img_in.shape[0]})...")
                masks = gen.generate(img_in)
                
                del gen, sam
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            masks_list = masks
            used_device = device
            break
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"[oom] {name} on {device}. Trying next device...")
            try:
                del gen, sam
            except:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[error] {name} on {device}: {e}")
            continue

    if masks_list is None:
        print(f"[fail] {name}: SAM inference failed.")
        return

    if scale != 1.0:
        print(f"Rescaling masks from {scale}x...")
        masks_up = []
        for m in masks_list:
            seg_small = m["segmentation"].astype(np.uint8)
            seg_up = cv2.resize(seg_small, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            x, y, w, h = to_xywh(m["bbox"])
            x = int(round(x/scale)); y = int(round(y/scale))
            w = int(round(w/scale)); h = int(round(h/scale))
            
            mm = dict(m)
            mm["segmentation"] = seg_up
            mm["bbox"] = [x, y, w, h]
            masks_up.append(mm)
        masks_list = masks_up

    masks_sorted = sorted(masks_list, key=lambda m: m["area"])
    M = len(masks_sorted)
    
    stack = np.zeros((M, H0, W0), dtype=np.uint8)
    areas = np.zeros((M,), dtype=np.int32)
    scores = np.zeros((M,), dtype=np.float32)
    bboxes = np.zeros((M, 4), dtype=np.int32)  # xywh

    for i, m in enumerate(masks_sorted):
        stack[i]  = m["segmentation"].astype(np.uint8)
        areas[i]  = int(m["area"])
        scores[i] = float(m.get("predicted_iou", m.get("stability_score", 0.0)))
        bboxes[i] = np.array(to_xywh(m["bbox"]), dtype=np.int32)

    np.savez_compressed(
        npz_out,
        masks=stack, areas=areas, scores=scores, bboxes=bboxes,
        image_name=os.path.basename(img_path), H=H0, W=W0
    )

    vis = vis_overlay(img_rgb, stack)
    cv2.imwrite(vis_out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"[ok] {name}: masks={M}  device={used_device}  size={W0}x{H0}")

    del masks_list, masks_sorted, stack, areas, scores, bboxes, img_bgr, img_rgb, vis
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    files = sorted([p for p in glob.glob(os.path.join(IMG_DIR, "*")) if is_image(p)])
    if not files:
        print("[info] no images in", IMG_DIR)
        return
    
    print(f"Found {len(files)} images. Starting processing using ViT-H...")
    
    for p in files:
        try:
            export_one_image(p)
        except KeyboardInterrupt:
            print("\n[stop] User interrupted.")
            break
        except Exception as e:
            print("[error-critical]", os.path.basename(p), e)
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()