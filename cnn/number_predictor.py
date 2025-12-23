import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Dosya Yolları
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "number_cnn_best.pt"
META_PATH = BASE_DIR / "models" / "number_cnn_meta.json"

# --- MODEL YAPISI ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
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

# --- AYARLAR ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
CLASSES = []

if MODEL_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        CLASSES = meta["classes"]
        model = SimpleCNN(len(CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device).eval()
    except Exception as e:
        print(f"❌ Model hatası: {e}")

# Transform
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return transforms.functional.pad(image, (p_left, p_top, p_right, p_bottom), padding_mode='edge')

base_tf = transforms.Compose([
    SquarePad(),
    transforms.Resize((80, 80), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

def preprocess_image(roi_bgr):
    """
    Sarı Kanalı + CLAHE
    """
    b, g, r = cv2.split(roi_bgr)
    darkest = np.minimum(np.minimum(b, g), r)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(darkest)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

@torch.no_grad()
def predict_number(roi_bgr):
    if model is None or roi_bgr is None: return None, 0.0

    processed_img = preprocess_image(roi_bgr)
    img_pil = Image.fromarray(processed_img)

    # --- HASSAS MULTI-VIEW (12 Sorunu Çözümü) ---
    # 1. Normal (%80 Ağırlık) - PATRON BU
    t1 = base_tf(img_pil).to(device)
    
    # 2. Sola Eğik (%10 Ağırlık) - Sadece 3 derece (5 değil!)
    t2 = base_tf(img_pil.rotate(3)).to(device)
    
    # 3. Sağa Eğik (%10 Ağırlık) - Sadece 3 derece
    t3 = base_tf(img_pil.rotate(-3)).to(device)

    # Batch
    batch = torch.stack([t1, t2, t3])
    
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    
    # AĞIRLIKLI ORTALAMA
    # Düz görüntüye (probs[0]) çok daha fazla güveniyoruz (%80).
    # Yanlar sadece "acaba mı?" diye kontrol ediyor.
    # Bu sayede 12'nin düz çizgisi yamulup 13'e benzemiyor.
    w_probs = (probs[0] * 0.8) + (probs[1] * 0.1) + (probs[2] * 0.1)
    
    idx = int(w_probs.argmax().item())
    conf = float(w_probs[idx].item())
    
    if conf < 0.35:
        return None, 0.0
    
    try:
        return int(CLASSES[idx]), conf
    except:
        return None, 0.0