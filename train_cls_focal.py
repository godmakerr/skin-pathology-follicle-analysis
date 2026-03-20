import os
import csv
import copy
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import torch.nn.functional as F

# ================= Configuration =================
TRAIN_CSV = "cls_dataset/train.csv"
VAL_CSV   = "cls_dataset/val.csv"
OUT_DIR   = "checkpoints_cls"
MODEL_NAME= "resnet18_focal_best.pth"

# 0:Others, 1:T, 2:V, 3:I, 6:Stela, 7:Fibrosis
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5}
INV_MAP   = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

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

# ================= Focal Loss =================
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

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ================= Dataset =================
class PatchDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                label_original = int(row["label"])
                if label_original in LABEL_MAP:
                    self.items.append({
                        "path": row["path"],
                        "label_idx": LABEL_MAP[label_original],
                        "label_orig": label_original
                    })
        self.transform = transform

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
        return img, item["label_idx"]

# ================= Utils =================
def get_transforms():
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

def make_weighted_sampler(dataset):
    targets = [item["label_idx"] for item in dataset.items]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, class_counts

# ================= Metrics (Manual Implementation) =================
def compute_metrics_manual(all_targets, all_preds, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
    
    accuracy = torch.sum(torch.diag(cm)) / torch.sum(cm)
    
    lines = []
    lines.append(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
    lines.append("-" * 55)
    
    f1_sum = 0.0
    valid_classes = 0
    
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        support = cm[c, :].sum().item()
        
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        
        real_label_id = INV_MAP[c]
        name_map = {0:"Others", 1:"T", 2:"V", 3:"I", 6:"Stela", 7:"Fibrosis"}
        cls_name = name_map.get(real_label_id, str(real_label_id))
        
        lines.append(f"{cls_name:<12} {precision:.4f}     {recall:.4f}     {f1:.4f}     {support:<8}")
        
        if support > 0:
            f1_sum += f1
            valid_classes += 1
            
    macro_f1 = f1_sum / max(1, valid_classes)
    report_text = "\n".join(lines)
    return accuracy.item(), macro_f1, report_text

# ================= Loops =================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

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
    correct = sum([1 for p, t in zip(all_preds, all_targets) if p == t])
    acc = correct / len(all_targets)
    return avg_loss, acc

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
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
    acc, f1_macro, report = compute_metrics_manual(all_targets, all_preds, NUM_CLASSES)
    return avg_loss, acc, f1_macro, report

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    train_tfm, val_tfm = get_transforms()
    train_ds = PatchDataset(TRAIN_CSV, transform=train_tfm)
    val_ds   = PatchDataset(VAL_CSV, transform=val_tfm)
    
    sampler, counts = make_weighted_sampler(train_ds)
    print(f"Train Class Counts (idx order): {counts}")
    
    # FIX: num_workers=0 to prevent Segmentation Fault
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Initializing ResNet18 for {NUM_CLASSES} classes...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.to(DEVICE)

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    best_f1 = 0.0
    
    print("Start Training...")
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, report = validate(model, val_loader, criterion)
        
        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1(Macro): {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(OUT_DIR, MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"--> Best Model Saved! F1: {best_f1:.4f}")
            print(report)

    print(f"Done. Best F1: {best_f1:.4f}. Model saved to {os.path.join(OUT_DIR, MODEL_NAME)}")

if __name__ == "__main__":
    main()