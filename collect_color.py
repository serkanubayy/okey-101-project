import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# =============================
# AYAR
# =============================
SAVE_ROOT = "color_dataset"
CLASSES = ["Red", "Blue", "Orange", "Black"]

# hangi tuş hangi renk
KEY_MAP = {
    ord('r'): "Red",
    ord('b'): "Blue",
    ord('o'): "Orange",
    ord('k'): "Black",   # k = black
}

# model
model = YOLO("best.pt")
CONF_TH = 0.45
IOU_TH = 0.45
IMG_SIZE = 640

# ROI (aynı mantık)
def box_center(box):
    x1,y1,x2,y2 = box
    return np.array([(x1+x2)//2,(y1+y2)//2])

def inside_roi(box, rx1, ry1, rx2, ry2):
    cx,cy = box_center(box)
    return rx1 < cx < rx2 and ry1 < cy < ry2

def crop_tile(img, box, pad=6):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = map(int,box)
    x1,y1 = max(0,x1-pad), max(0,y1-pad)
    x2,y2 = min(w,x2+pad), min(h,y2+pad)
    return img[y1:y2, x1:x2] if x2>x1 and y2>y1 else None

# klasörleri oluştur
for c in CLASSES:
    os.makedirs(os.path.join(SAVE_ROOT, c), exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

print("\n--- RENK DATASET TOPLAMA ---")
print("Renk seç:")
print("  r = Red, b = Blue, o = Orange, k = Black")
print("Kaydetmek için: SPACE")
print("Çıkış: q")
print("----------------------------\n")

selected = "Red"
last_tiles = []

while True:
    ret, img = cap.read()
    if not ret:
        break

    h,w,_ = img.shape
    rx1,rx2 = 100, w-100
    ry1,ry2 = h//2-160, h//2+160

    # ROI çiz
    cv2.rectangle(img,(rx1,ry1),(rx2,ry2),(0,255,255),3)
    cv2.putText(img, f"SELECTED: {selected}   (r/b/o/k degistir)   SPACE: kaydet",
                (rx1, ry1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # YOLO ile kutuları al (preview)
    results = model(img, conf=CONF_TH, iou=IOU_TH, imgsz=IMG_SIZE)[0]
    tiles = []
    for b in results.boxes:
        xyxy = b.xyxy[0].tolist()
        if inside_roi(xyxy, rx1, ry1, rx2, ry2):
            tile = crop_tile(img, xyxy)
            if tile is not None:
                tiles.append(tile)

    last_tiles = tiles  # SPACE basınca bunu kaydedeceğiz

    cv2.imshow("COLLECT COLOR", img)
    key = cv2.waitKey(1) & 0xFF

    if key in KEY_MAP:
        selected = KEY_MAP[key]

    if key == ord('q'):
        break

    if key == ord(' '):
        # o frame'deki tüm taş crop'larını kaydet
        if len(last_tiles) == 0:
            print("⚠️ Bu frame'de tas bulunamadı, biraz daha net tut.")
            continue

        ts = str(time.time()).replace(".", "")
        saved = 0
        for i, tile in enumerate(last_tiles):
            out_path = os.path.join(SAVE_ROOT, selected, f"{ts}_{i}.jpg")
            cv2.imwrite(out_path, tile)
            saved += 1
        print(f"✅ {selected} klasörüne {saved} foto kaydedildi.")

cap.release()
cv2.destroyAllWindows()
