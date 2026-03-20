import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ================= Configuration =================
CSV_PATH = "scene_labels.csv"
IMG_DIR  = "data/raw_images"
OUT_DIR  = "checkpoints_global"
EPOCHS   = 30
BATCH_SIZE = 8
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ================= Dataset =================
class SceneDataset(Dataset):
    def __init__(self, data_list, transform=None, target_col="slope_label"):
        self.data = data_list
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        filename = item["filename"].strip()
        img_path = os.path.join(IMG_DIR, f"{filename}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        raw_label = str(item[self.target_col])
        
        if ";" in raw_label:
            choices = raw_label.split(";")
            label_val = int(random.choice(choices).strip())
        else:
            label_val = int(raw_label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_val

# ================= Training Utils =================
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def filter_data_for_task(all_data, target_col):
    """
    Robust data filtering:
    1. Checks if label exists and is a number (handles "1", "1.0").
    2. Checks if image file exists (with .jpg appended).
    """
    valid_data = []
    skipped = 0
    
    if len(all_data) > 0:
        print(f"[Debug] CSV Headers: {list(all_data[0].keys())}")
    
    for i, row in enumerate(all_data):
        val = row.get(target_col, "")
        if val is None: val = ""
        val = str(val).strip()
        
        is_multi = ";" in val
        is_number = False
        try:
            float(val)
            is_number = True
        except:
            pass
            
        if (not is_number) and (not is_multi):
            skipped += 1
            continue

        if is_number:
            row[target_col] = str(int(float(val)))
        
        raw_name = row.get("filename", "").strip()
        full_img_name = f"{raw_name}.jpg"
        if not os.path.exists(os.path.join(IMG_DIR, full_img_name)):
            if i < 3: print(f"[Warn] Missing image: {full_img_name}")
            skipped += 1
            continue
            
        valid_data.append(row)
        
    print(f"Task [{target_col}]: Used {len(valid_data)} items, Skipped {skipped} items.")
    return valid_data

def train_task(task_name, target_col, num_classes, all_data_raw):
    print(f"\n>>> Preparing Task: {task_name} (Classes: {num_classes})")
    
    task_data = filter_data_for_task(all_data_raw, target_col)
    
    if len(task_data) < 2:
        print(f"[Error] Not enough data for {task_name}, skipping...")
        return

    # MANUAL SPLIT instead of sklearn
    random.shuffle(task_data)
    split_idx = int(len(task_data) * 0.8) # 80% train, 20% val
    train_data = task_data[:split_idx]
    val_data = task_data[split_idx:]
    
    print(f"Split: Train={len(train_data)}, Val={len(val_data)}")

    # Setup Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)
    
    # Loaders
    ds_train = SceneDataset(train_data, transform=get_transforms(True), target_col=target_col)
    ds_val   = SceneDataset(val_data, transform=get_transforms(False), target_col=target_col)
    
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0; correct = 0; total = 0
        
        for imgs, lbls in dl_train:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)
        
        train_acc = correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for imgs, lbls in dl_val:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == lbls).sum().item()
                val_total += lbls.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"global_{task_name}.pth"))
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        
    print(f"Task {task_name} Finished. Best Val Acc: {best_acc:.3f}")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[Error] CSV not found: {CSV_PATH}")
        return

    all_data_raw = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_data_raw.append(row)
            
    print(f"Loaded {len(all_data_raw)} raw rows from CSV.")
    
    # Train Slope (0, 1)
    train_task("slope", "slope_label", 2, all_data_raw)
    
    # Train Layer (0, 1, 2, 3)
    train_task("layer", "layer_label", 4, all_data_raw)

if __name__ == "__main__":
    main()