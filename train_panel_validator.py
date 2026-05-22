"""
train_panel_validator.py
------------------------
Trains a binary solar panel validator:
  Class 0 = NOT a solar panel (random images from CIFAR-10 + STL-10)
  Class 1 = IS a solar panel  (our dataset)

Uses MobileNetV3-Small (1.5MB) for fast inference.
Output: panel_validator.pth
"""

import os, sys, shutil, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
BATCH      = 64
EPOCHS     = 50
LR         = 2e-4
SAVE_PATH  = "panel_validator.pth"
DATA_DIR   = Path("validator_data")

print(f"Device: {DEVICE}")

# ── Transforms ─────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Solar panel dataset ─────────────────────────────────────────
class SolarPanelDataset(Dataset):
    def __init__(self, split_dirs, transform):
        self.paths = []
        for d in split_dirs:
            for cls in ['Bird-drop','Clean','Dusty','Electrical-damage','Physical-Damage','Snow-Covered']:
                cls_dir = Path(d) / cls
                if cls_dir.exists():
                    for ext in ['*.jpg','*.JPG','*.jpeg','*.png']:
                        self.paths.extend(cls_dir.glob(ext))
        random.shuffle(self.paths)
        self.transform = transform
        print(f"  Solar panels: {len(self.paths)} images from {split_dirs}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), 1   # label 1 = solar panel
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (50,50,50))
            return self.transform(img), 1

# ── Non-panel dataset (CIFAR-10 resized) ───────────────────────
class NonPanelDataset(Dataset):
    """Wraps a torchvision dataset, returns label=0 (not a panel)."""
    def __init__(self, base_dataset, transform):
        self.base      = base_dataset
        self.transform = transform

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        # img is already a tensor from base dataset — convert back to PIL
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        return self.transform(img), 0   # label 0 = not a panel

# ── Download non-panel data ─────────────────────────────────────
print("\nDownloading non-panel images (CIFAR-10)...")
cifar_raw_tf = transforms.ToTensor()
cifar_train = datasets.CIFAR10(root=str(DATA_DIR/"cifar"), train=True,
                                download=True, transform=cifar_raw_tf)
cifar_val   = datasets.CIFAR10(root=str(DATA_DIR/"cifar"), train=False,
                                download=True, transform=cifar_raw_tf)
print(f"  CIFAR-10: {len(cifar_train)} train + {len(cifar_val)} val")

# ── Build datasets ──────────────────────────────────────────────
print("\nBuilding datasets...")
solar_train = SolarPanelDataset(["train","val"], train_tf)
solar_val   = SolarPanelDataset(["test"],        val_tf)

non_train = NonPanelDataset(cifar_train, train_tf)
non_val   = NonPanelDataset(cifar_val,   val_tf)

# Balance: use same number of non-panel as panel
n_solar_train = len(solar_train)
n_solar_val   = len(solar_val)

# Subsample non-panel to match solar count
from torch.utils.data import Subset
non_train_sub = Subset(non_train, list(range(min(n_solar_train, len(non_train)))))
non_val_sub   = Subset(non_val,   list(range(min(n_solar_val*3, len(non_val)))))

train_ds = ConcatDataset([solar_train, non_train_sub])
val_ds   = ConcatDataset([solar_val,   non_val_sub])

print(f"  Train: {len(train_ds)} ({n_solar_train} panel + {len(non_train_sub)} non-panel)")
print(f"  Val:   {len(val_ds)} ({n_solar_val} panel + {len(non_val_sub)} non-panel)")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=True)

# ── Model: MobileNetV3-Small ────────────────────────────────────
print("\nBuilding MobileNetV3-Small binary classifier...")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# Replace classifier head for binary classification
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params/1e6:.2f}M")

# ── Training ────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.amp.GradScaler(enabled=DEVICE.type=="cuda")

best_val_acc = 0.0
print(f"\nTraining for {EPOCHS} epochs...\n")

for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    tr_correct = tr_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type=="cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        preds = out.argmax(1)
        tr_correct += (preds==labels).sum().item()
        tr_total   += labels.size(0)
    scheduler.step()

    # Validate
    model.eval()
    vl_correct = vl_total = 0
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out   = model(imgs)
            preds = out.argmax(1)
            vl_correct += (preds==labels).sum().item()
            vl_total   += labels.size(0)
            tp += ((preds==1)&(labels==1)).sum().item()
            tn += ((preds==0)&(labels==0)).sum().item()
            fp += ((preds==1)&(labels==0)).sum().item()
            fn += ((preds==0)&(labels==1)).sum().item()

    tr_acc = tr_correct/tr_total
    vl_acc = vl_correct/vl_total
    precision = tp/max(tp+fp,1)
    recall    = tp/max(tp+fn,1)
    f1        = 2*precision*recall/max(precision+recall,1e-6)

    print(f"Epoch {epoch:3d}/{EPOCHS}  "
          f"tr={tr_acc:.4f}  val={vl_acc:.4f}  "
          f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

print(f"\nDone! Best val accuracy: {best_val_acc:.4f}")
print(f"Model saved: {SAVE_PATH}")
