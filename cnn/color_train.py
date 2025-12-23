import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import json

# =====================
# AYAR
# =====================
IMG_SIZE = 96
BATCH = 64
EPOCHS = 18
LR = 1e-3

DATA_DIR = Path("../color_dataset")
OUT_DIR = Path("cnn/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "color_cnn.pt"
META_PATH  = OUT_DIR / "color_meta.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# =====================
# TRANSFORM
# =====================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.06),
    transforms.RandomRotation(6),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

classes = dataset.classes
num_classes = len(classes)
print("CLASSES:", classes)

# =====================
# MODEL
# =====================
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# =====================
# TRAIN
# =====================
best_acc = 0.0
for ep in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / max(total, 1)
    print(f"[{ep+1}/{EPOCHS}] acc={acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)
        META_PATH.write_text(json.dumps({
            "classes": classes,
            "img_size": IMG_SIZE,
            "best_train_acc": float(best_acc)
        }, indent=2), encoding="utf-8")
        print("✅ BEST saved:", MODEL_PATH)

print("✅ COLOR CNN READY:", MODEL_PATH)
print("✅ META:", META_PATH)
