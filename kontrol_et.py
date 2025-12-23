import os
import cv2
import numpy as np
import math
from pathlib import Path

# Dataset yolu
DATA_DIR = Path("dataset/train")
OUTPUT_IMG = "dataset_kontrol.png"

if not DATA_DIR.exists():
    print("❌ Dataset klasörü yok!")
    exit()

classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
images_to_show = []

print("📂 Klasörler taranıyor...")

for cls in classes:
    folder = DATA_DIR / cls
    files = list(folder.glob("*.jpg"))[:5] # Her sayıdan 5 örnek al
    
    if not files: continue
    
    # O sayı için yan yana 5 resim
    row_imgs = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None: continue
        # Hepsini 64x64 yap ki düzgün dursun
        img = cv2.resize(img, (64, 64))
        # Üstüne hangi klasörden geldiğini yaz
        cv2.putText(img, cls, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        row_imgs.append(img)
    
    if row_imgs:
        # Yan yana birleştir
        row_concat = np.hstack(row_imgs)
        images_to_show.append(row_concat)

if images_to_show:
    # Alt alta birleştir
    max_w = max(img.shape[1] for img in images_to_show)
    final_rows = []
    for img in images_to_show:
        # Genişlik eşitle (padding)
        if img.shape[1] < max_w:
            pad = np.zeros((64, max_w - img.shape[1], 3), dtype=np.uint8)
            img = np.hstack([img, pad])
        final_rows.append(img)
    
    final_grid = np.vstack(final_rows)
    cv2.imwrite(OUTPUT_IMG, final_grid)
    print(f"✅ KONTROL RESMİ OLUŞTURULDU: {OUTPUT_IMG}")
    print("Lütfen bu resmi aç ve sayıların doğru klasörde olup olmadığına bak!")
else:
    print("⚠️ Gösterilecek resim bulunamadı.")