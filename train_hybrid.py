import os
import csv
import copy
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import models, transforms
import torch.nn.functional as F

# ================= Configuration =================
# 1. 路径设置
CSV_TRAIN = "cls_dataset/train.csv"
CSV_VAL   = "cls_dataset/val.csv"

# 新数据文件夹 (若无则留空 "")
FOLDER_TRAIN_DIR = "new_dataset/train"
FOLDER_VAL_DIR   = "new_dataset/val"

OUT_DIR    = "checkpoints_cls"
MODEL_NAME = "resnet18_hybrid_v2.pth"

# ================= 核心：绝对锁定的映射表 =================
# 这里的逻辑是为了保证和你原来的推理脚本完全兼容
# Index 0-5 的含义绝对不变，只在最后追加 Index 6 (D)

# 1. 类名列表 (顺序很重要，对应模型输出 0,1,2...)
CLASS_NAMES = ["Others", "T", "V", "I", "Stela", "Fibrosis", "D"]
NUM_CLASSES = len(CLASS_NAMES) # 现在是 7

# 2. CSV ID -> 模型 Index 映射
# 原有 ID: 0, 1, 2, 3, 6, 7
# 新增 ID: 4 (假设你在CSV里用4代表毛球，或者CSV里根本没毛球，这行就不生效)
CSV_ID_MAP = {
    0: 0,  # Others
    1: 1,  # T
    2: 2,  # V
    3: 3,  # I
    6: 4,  # Stela (旧ID 6 -> 模型第5个位置)
    7: 5,  # Fibrosis (旧ID 7 -> 模型第6个位置)
    4: 6,  # D/毛球 (假设ID 4 -> 模型第7个位置)
    8: 6   # 备用：万一你CSV里用8代表毛球，也映射到6
}

# 3. 文件夹名关键字 -> 模型 Index 映射
FOLDER_KEYWORD_MAP = {
    "Others": 0, "0_": 0,
    "T": 1,      "1_": 1,
    "V": 2,      "2_": 2,
    "I": 3,      "3_": 3,
    "Stela": 4,  "6_": 4, 
    "Fibrosis": 5,"7_": 5,
    "D": 6,      "Maoqiu": 6, "4_": 6, "8_": 6 # 只要文件名含 D 或 Maoqiu，就是Index 6
}

# 训练参数 (保持原样)
BATCH_SIZE = 32
EPOCHS     = 25
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

# Fix for Segmentation Faults
cv2.setNumThreads(0) 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= Focal Loss (保持原样) =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ================= Datasets =================
class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.items = []
        self.transform = transform
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    lbl = int(row["label"])
                    if lbl in CSV_ID_MAP:
                        self.items.append({
                            "path": row["path"],
                            "label": CSV_ID_MAP[lbl]
                        })
            print(f"[CSV] Loaded {len(self.items)} images from {csv_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        bgr = cv2.imread(item["path"])
        if bgr is None:
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(rgb)
            else:
                img = transforms.ToTensor()(rgb)
        return img, item["label"]
    
    def get_labels(self):
        return [item["label"] for item in self.items]

class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.items = []
        self.transform = transform
        
        if root_dir and os.path.exists(root_dir):
            for folder_name in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder_name)
                if not os.path.isdir(folder_path): continue
                
                # 模糊匹配关键字
                label_idx = -1
                for key, val in FOLDER_KEYWORD_MAP.items():
                    # 比如文件夹叫 "new_D_images"，包含了 "D"，所以匹配到 index 6
                    if key in folder_name: 
                        label_idx = val
                        break
                
                if label_idx == -1:
                    print(f"[Folder] Ignoring unknown folder: {folder_name}")
                    continue
                
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
                for fname in os.listdir(folder_path):
                    if os.path.splitext(fname)[1].lower() in valid_exts:
                        self.items.append({
                            "path": os.path.join(folder_path, fname),
                            "label": label_idx
                        })
            print(f"[Folder] Loaded {len(self.items)} images from {root_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        bgr = cv2.imread(item["path"]) # 保持和CSV一样的读取方式
        if bgr is None:
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(rgb)
            else:
                img = transforms.ToTensor()(rgb)
        return img, item["label"]

    def get_labels(self):
        return [item["label"] for item in self.items]

# ================= Utils =================
def get_transforms():
    # 保持原有的增强参数不变
    train_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tfm, val_tfm

def make_weighted_sampler(datasets_list):
    all_targets = []
    for ds in datasets_list:
        all_targets.extend(ds.get_labels())
    
    if not all_targets: return None

    all_targets = np.array(all_targets)
    # 计算权重时包含所有7个类别
    class_counts = np.bincount(all_targets, minlength=NUM_CLASSES)
    print(f"Total Class Counts (0..6): {class_counts}")
    
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in all_targets]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# ================= Metrics (Modified for 7 Classes) =================
def compute_metrics_manual(all_targets, all_preds):
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int32)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
    
    accuracy = torch.sum(torch.diag(cm)) / torch.sum(cm)
    
    lines = []
    lines.append(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
    lines.append("-" * 55)
    
    f1_sum = 0.0
    valid_classes = 0
    
    for c in range(NUM_CLASSES):
        tp = cm[c, c].item()
        # 简单处理：避免除零
        row_sum = cm[c, :].sum().item()
        col_sum = cm[:, c].sum().item()
        
        precision = tp / (col_sum + 1e-9)
        recall    = tp / (row_sum + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        
        support = row_sum
        lines.append(f"{CLASS_NAMES[c]:<12} {precision:.4f}     {recall:.4f}     {f1:.4f}     {support:<8}")
        
        if support > 0:
            f1_sum += f1
            valid_classes += 1
            
    macro_f1 = f1_sum / max(1, valid_classes)
    return accuracy.item(), macro_f1, "\n".join(lines)

def format_confusion_matrix(cm, class_names):
    lines = []
    header = ["GT\\Pred"] + class_names
    col_width = max(10, max(len(x) for x in header) + 2)

    lines.append("Confusion Matrix (rows=true, cols=pred):")
    lines.append("".join(f"{x:<{col_width}}" for x in header))

    for i, row_name in enumerate(class_names):
        row = [row_name] + [str(int(cm[i, j].item())) for j in range(len(class_names))]
        lines.append("".join(f"{x:<{col_width}}" for x in row))

    return "\n".join(lines)
    
# ================= Loops =================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for imgs, targets in loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = sum([1 for p, t in zip(all_preds, all_targets) if p == t]) / len(all_targets)
    return avg_loss, acc

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, f1_macro, report = compute_metrics_manual(all_targets, all_preds)

    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int32)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    return avg_loss, acc, f1_macro, report, cm

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_tfm, val_tfm = get_transforms()
    
    # 1. 组合训练数据
    print("--- Loading Training Data ---")
    d_csv = CSVDataset(CSV_TRAIN, transform=train_tfm)
    d_fld = FolderDataset(FOLDER_TRAIN_DIR, transform=train_tfm)
    
    train_list = [d for d in [d_csv, d_fld] if len(d) > 0]
    if not train_list:
        print("Error: No training data found.")
        return
        
    full_train_ds = ConcatDataset(train_list)
    sampler = make_weighted_sampler(train_list)
    
    # 2. 组合验证数据
    print("\n--- Loading Validation Data ---")
    v_csv = CSVDataset(CSV_VAL, transform=val_tfm)
    v_fld = FolderDataset(FOLDER_VAL_DIR, transform=val_tfm)
    
    val_list = [d for d in [v_csv, v_fld] if len(d) > 0]
    if not val_list:
        print("Error: No validation data found.")
        return
    full_val_ds = ConcatDataset(val_list)

    # 3. Loaders (保持 num_workers=0 避免 Segmentation Fault)
    train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(full_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model (Change output to 7 classes)
    print(f"\nInitializing ResNet18 for {NUM_CLASSES} classes (0-6)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES) # 输出维度变为 7
    model.to(DEVICE)

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    best_f1 = 0.0
    
    print(f"Start Training on {DEVICE}...")
    for epoch in range(1, EPOCHS+1):
      train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
      val_loss, val_acc, val_f1, report, cm = validate(model, val_loader, criterion)
      
      print(f"Epoch [{epoch}/{EPOCHS}] | Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | Val: Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
      
      if val_f1 > best_f1:
          best_f1 = val_f1
          torch.save(model.state_dict(), os.path.join(OUT_DIR, MODEL_NAME))
          print(f"--> Best Model Saved! F1: {best_f1:.4f}")
          print(report)

    print(f"Done. Saved to {os.path.join(OUT_DIR, MODEL_NAME)}")
    
    best_model_path = os.path.join(OUT_DIR, MODEL_NAME)

    if os.path.exists(best_model_path):
        print("\n=== Final Evaluation with Best Saved Model ===")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model.to(DEVICE)
    
        val_loss, val_acc, val_f1, report, cm = validate(model, val_loader, criterion)
    
        print(f"Best Model | Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
        print(report)
        print()
        print(format_confusion_matrix(cm, CLASS_NAMES))
    else:
        print("Warning: best model file not found, cannot print final confusion matrix.")
    
    print(f"Done. Saved to {best_model_path}")
if __name__ == "__main__":
    main()