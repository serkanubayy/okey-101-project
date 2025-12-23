import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = Path("cnn/models/number_cnn_best.pt")
META_PATH  = Path("cnn/models/number_cnn_meta.json")

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError("Model veya meta yok. Önce train_cnn.py ile eğitim yap.")

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
model.classifier[-1] = nn.Linear(in_features, 13)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

@torch.no_grad()
def predict(img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    idx = int(prob.argmax().item())
    return classes[idx], float(prob[idx].item())

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("KULLANIM: python cnn/predict.py path/to/image.png")
        raise SystemExit

    label, conf = predict(sys.argv[1])
    print("PRED:", label, "CONF:", conf)
