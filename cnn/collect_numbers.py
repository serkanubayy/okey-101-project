import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
from number_crop import crop_number_region
from ultralytics import YOLO


model = YOLO("../best.pt")

SAVE_ROOT = "../cnn/dataset/train"

os.makedirs(SAVE_ROOT, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

print("NUMARA DATASET TOPLAMA MODU")
print("SPACE → dondur / sonraki sayıya geç")
print("1-9 → sayı | 0→10 | q→11 | w→12 | e→13")
print("ESC → canlıya dön / çıkış")

while True:
    ret, img = cap.read()
    if not ret:
        break

    cv2.imshow("CAM", img)
    key = cv2.waitKey(1) & 0xFF

    # =============================
    # SPACE → FREEZE + YOLO
    # =============================
    if key == ord(' '):
        frame = img.copy()
        print("[+] Frame donduruldu")

        results = model(frame, conf=0.45, iou=0.45)[0]

        tiles = []
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            tile = frame[y1:y2, x1:x2]
            num = crop_number_region(tile)
            if num is not None:
                tiles.append(num)

        if not tiles:
            print("⚠️ Sayı bulunamadı")
            continue

        index = 0

        while True:
            cv2.imshow("NUMBER", tiles[index])
            k = cv2.waitKey(0) & 0xFF

            label = None
            if k in range(ord('1'), ord('9') + 1):
                label = chr(k)
            elif k == ord('0'):
                label = '10'
            elif k == ord('q'):
                label = '11'
            elif k == ord('w'):
                label = '12'
            elif k == ord('e'):
                label = '13'
            elif k == 27:
                break

            if label:
                save_dir = os.path.join(SAVE_ROOT, label)
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{time.time()}.png"
                cv2.imwrite(os.path.join(save_dir, filename), tiles[index])
                print(f"[✓] Kaydedildi → {label}")

                index += 1
                if index >= len(tiles):
                    print("[✓] Bu frame bitti")
                    break

        cv2.destroyWindow("NUMBER")
        print("[+] Canlı görüntüye dönüldü")

    # =============================
    # ESC → ÇIKIŞ
    # =============================
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
