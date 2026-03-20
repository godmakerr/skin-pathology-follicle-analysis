import os, csv, argparse, random
from collections import defaultdict

DEF_CSV = "cls_dataset/labels.csv"
OUT_DIR = "cls_dataset"
SEED = 42

def read_items(csv_path):
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            path = row.get("path") or row.get("img") or row.get("file")
            label = row.get("label")
            image = row.get("image") or row.get("image_name") or ""
            if not path or label is None:
                continue
            label = int(label)
            if not image:
                # fallback: derive image id from filename prefix
                base = os.path.basename(path)
                image = base.split("__")[0] if "__" in base else base.split("_")[0]
            items.append({"path":path, "label":label, "image":image})
    return items

def stratified_group_split(items, val_ratio=0.2, seed=SEED):
    # group by image to avoid leakage
    by_img = defaultdict(list)
    for it in items:
        by_img[it["image"]].append(it)
    imgs = list(by_img.keys())
    random.Random(seed).shuffle(imgs)

    # approximate stratified by class proportion at image-level
    cls_counts = defaultdict(int)
    for it in items: cls_counts[it["label"]] += 1

    target_val = {c: int(round(n * val_ratio)) for c, n in cls_counts.items()}
    cur_val = defaultdict(int)
    val_imgs, train_imgs = set(), set()

    for img in imgs:
        grp = by_img[img]
        # decide whether putting this group into val helps reach targets
        add_to_val_score = 0
        for it in grp:
            c = it["label"]
            if cur_val[c] < target_val[c]:
                add_to_val_score += 1
        # heuristic: prefer val if it improves unmet classes
        if add_to_val_score > 0:
            val_imgs.add(img)
            for it in grp:
                cur_val[it["label"]] += 1
        else:
            train_imgs.add(img)

    train, val = [], []
    for img, grp in by_img.items():
        (val if img in val_imgs else train).extend(grp)
    return train, val

def write_csv(path, items):
    if not items:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","label","image"])
        w.writeheader()
        for it in items:
            w.writerow(it)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEF_CSV, help="Input cls_dataset/labels.csv")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    items = read_items(args.csv)
    if not items:
        raise RuntimeError("No items found in input CSV")

    train, val = stratified_group_split(items, val_ratio=args.val_ratio, seed=args.seed)
    write_csv(os.path.join(args.out_dir, "train.csv"), train)
    write_csv(os.path.join(args.out_dir, "val.csv"),   val)
    print(f"[done] total={len(items)} train={len(train)} val={len(val)} -> {args.out_dir}/train.csv, val.csv")

if __name__ == "__main__":
    main()
