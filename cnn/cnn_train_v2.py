import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# AYARLAR
IMG_SIZE = 80
BATCH_SIZE = 16
EPOCHS = 60
LR = 0.001
DATA_ROOT = Path("../dataset/train")
OUT_DIR = Path("cnn/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "number_cnn_best.pt"
META_PATH = OUT_DIR / "number_cnn_meta.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- AKILLI KARELEŞTİRME (SÜNDÜRMEDEN) ---
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        padding = (p_left, p_top, p_right, p_bottom)
        # padding_mode='edge' kenar rengini (beyaz/gri) devam ettirir. Siyah bant oluşmaz.
        return transforms.functional.pad(image, padding, padding_mode='edge')

train_tf = transforms.Compose([
    SquarePad(), # Önce kareye tamamla
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Sonra 64x64 yap
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
])

# BASİT VE HIZLI MODEL
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        # Boyut hesabı: 64 -> 32 -> 16 -> 8 (yaklaşık)
        # Tam hesap yerine AdaptivePool kullanıyoruz, hata riskini sıfırlar.
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    if not DATA_ROOT.exists():
        print("❌ Dataset yok!")
        return

    dataset = datasets.ImageFolder(str(DATA_ROOT), transform=train_tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    classes = dataset.classes
    print(f"🚀 Sınıflar: {classes}")
    
    model = SimpleCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(loader, desc=f"Ep {epoch+1}", ncols=80)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(acc=f"{correct/total:.1%}")

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            META_PATH.write_text(json.dumps({"classes": classes, "img_size": IMG_SIZE}, indent=2))
    
    print(f"✅ EĞİTİM BİTTİ. En iyi Acc: %{best_acc*100:.1f}")

if __name__ == "__main__":
    train()