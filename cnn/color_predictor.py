import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "color_cnn.pt"
META_PATH  = BASE_DIR / "models" / "color_meta.json"

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError(f"Color model veya meta yok: {MODEL_PATH} | {META_PATH}")

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
IMG_SIZE = int(meta["img_size"])
classes = meta["classes"]

device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

@torch.no_grad()
def predict_color(img):
    """
    img: OpenCV BGR ndarray OR path string
    return: (class_name, conf)
    """
    if isinstance(img, str):
        pil = Image.open(img).convert("RGB")
    else:
        if img is None:
            return "Unknown", 0.0
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # OpenCV BGR -> RGB
            pil = Image.fromarray(img[..., ::-1]).convert("RGB")
        else:
            pil = Image.fromarray(img).convert("RGB")

    x = tf(pil).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    idx = int(prob.argmax().item())
    return classes[idx], float(prob[idx].item())
