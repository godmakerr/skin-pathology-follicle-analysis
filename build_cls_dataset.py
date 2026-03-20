# build_cls_dataset.py
import os, csv, glob
import numpy as np
import cv2

IMG_DIR = "data/raw_images"
NPZ_DIR = "data/sam_npz"
CSV_IN  = "labels/labels.csv"
OUT_DIR = "cls_dataset"
OUT_IMG = os.path.join(OUT_DIR, "images")
META_CSV= os.path.join(OUT_DIR, "labels.csv")
os.makedirs(OUT_IMG, exist_ok=True)

CLASS_NAME_TO_ID = {
    "Others": 0,
    "T": 1,
    "V": 2,
    "I": 3,
    "Stela": 6,
    "Fibrosis": 7,
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p): return os.path.splitext(p)[1].lower() in IMG_EXTS

def load_labels(csv_path):
    table = {}
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"labels.csv not found: {csv_path}")
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                img = row.get("image_name", row.get("image"))
                idx = int(row.get("mask_idx", row.get("inst_id", row.get("idx"))))
                name = os.path.basename(img)
                cname = row.get("class_name", row.get("class", row.get("label")))
                if cname is None:
                    continue
                cname = cname.strip()
                if cname.isdigit():
                    cid = int(cname)
                else:
                    cid = CLASS_NAME_TO_ID.get(cname, None)
                    if cid is None:
                        continue
                if cid < 0:
                    continue
                table.setdefault(name, {})[idx] = cid
            except:
                continue
    return table

def safe_crop(img, x, y, w, h, pad=8):
    H, W = img.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return img[y0:y1, x0:x1]

def build_one(npz_path, labels, out_writer):
    z = np.load(npz_path, allow_pickle=False)
    masks  = z["masks"]          # [N,H,W] uint8
    bboxes = z["bboxes"]         # [N,4] xywh
    image  = str(z["image_name"])
    img_path = os.path.join(IMG_DIR, image)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[skip-missing-img] {img_path}")
        return
    H, W = img.shape[:2]
    L = labels.get(image, {})

    kept = 0
    for i in range(masks.shape[0]):
        if i not in L:
            continue
        cid = int(L[i])
        if cid < 0:
            continue
        x, y, w, h = map(int, bboxes[i])
        if w <= 1 or h <= 1:
            # fallback to mask bbox if needed
            ys, xs = np.where(masks[i] > 0)
            if len(xs) == 0:
                continue
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
        patch = safe_crop(img, x, y, w, h, pad=8)
        if patch.size == 0:
            continue
        out_name = f"{os.path.splitext(image)[0]}__{i:04d}.png"
        out_path = os.path.join(OUT_IMG, out_name)
        cv2.imwrite(out_path, patch)
        out_writer.writerow({"path": out_path, "label": cid})
        kept += 1
    print(f"[ok] {os.path.basename(npz_path)} -> {kept} patches")

def main():
    labels = load_labels(CSV_IN)
    npzs = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if not npzs:
        print("[info] no npz found in", NPZ_DIR)
        return
    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        for p in npzs:
            build_one(p, labels, w)
    print("[done] cls_dataset ready:", OUT_DIR)

if __name__ == "__main__":
    main()
