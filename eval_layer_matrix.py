import matplotlib
matplotlib.use('Agg') # 后台绘图
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ================= 配置 =================
CSV_PATH  = "scene_labels.csv"
IMG_DIR   = "data/raw_images"
MODEL_PATH = "checkpoints_global/global_layer.pth"
OUT_IMG   = "layer_confusion_matrix.png"

# 0: Upper, 1: Mid, 2: Lower, 3: Subcutis
CLASSES = ["Upper", "Mid", "Lower", "Subcutis"]
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 预处理 =================
eval_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model.to(DEVICE).eval()
    return model

def parse_label(val):
    if val is None: return None
    val = str(val).strip()
    # 处理分号 (2;3 -> 2)
    if ";" in val:
        val = val.split(";")[0].strip()
    try:
        return int(float(val))
    except:
        return None

def compute_cm(y_true, y_pred, num_classes):
    """手动计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm

def plot_cm(cm, class_names, out_path):
    """只用 Matplotlib 画热力图 (替代 Seaborn)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, 
           yticklabels=class_names,
           title='Confusion Matrix (Layer)',
           ylabel='True Label',
           xlabel='Predicted Label')

    # 在格子里填数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    model = load_model()
    
    y_true = []
    y_pred = []
    
    print("Starting evaluation...")
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    count = 0
    skipped = 0
    
    with torch.no_grad():
        for row in rows:
            # 1. 解析标签
            raw_label = row.get("layer_label", "")
            gt = parse_label(raw_label)
            
            if gt is None or gt < 0 or gt >= NUM_CLASSES:
                skipped += 1
                continue
            
            # 2. 解析图片路径
            filename = row.get("filename", "").strip()
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename = f"{filename}.jpg"
            img_path = os.path.join(IMG_DIR, filename)
            
            if not os.path.exists(img_path):
                skipped += 1
                continue
            
            # 3. 预测
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = eval_tfm(img).unsqueeze(0).to(DEVICE)
                
                outputs = model(input_tensor)
                # Softmax + Argmax
                pred = torch.argmax(outputs, 1).item()
                
                y_true.append(gt)
                y_pred.append(pred)
                count += 1
            except Exception as e:
                print(f"Err: {e}")
                skipped += 1

    print(f"\nProcessed: {count}, Skipped: {skipped}")

    if count == 0:
        print("No valid data.")
        return

    # 4. 计算并画图
    cm = compute_cm(y_true, y_pred, NUM_CLASSES)
    
    # 简单的文本报告
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(cm)
    
    # 画图
    plot_cm(cm, CLASSES, OUT_IMG)
    print(f"\nImage saved to: {OUT_IMG}")

if __name__ == "__main__":
    main()