import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from number_crop import crop_number_region

# =============================
# AYARLAR
# =============================
SAVE_ROOT = "dataset/train"  # Kayıt yeri
MODEL_PATH = "best.pt"       # YOLO modelin

if not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)

# Model yükle
model = YOLO(MODEL_PATH)
model.overrides["verbose"] = False

# Kamera
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

print("\n========================================")
print("   SEKO BABA - DİJİTAL VERİ TOPLAYICI   ")
print("========================================")
print("1. Kamerayı ekrana tut (Sabit olsun).")
print("2. 'SPACE' tuşuna basıp dondur.")
print("3. Ekrana gelen sayı için klavyeden bas:")
print("   [1-9] -> Sayı")
print("   [0]   -> 10")
print("   [q]   -> 11")
print("   [w]   -> 12")
print("   [e]   -> 13")
print("   [x]   -> Pas Geç (Hatalıysa)")
print("   [ESC] -> Çıkış")
print("========================================\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    # ROI Çiz (Odaklanma alanı)
    h, w, _ = frame.shape
    rx1, rx2 = 100, w - 100
    ry1, ry2 = h // 2 - 150, h // 2 + 150
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
    cv2.putText(frame, "SPACE: DONDUR", (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Kamera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27: # ESC
        break

    # SPACE BASINCA DONDUR VE ANALİZ ET
    if key == ord(' '):
        frozen = frame.copy()
        print("❄️  Görüntü donduruldu, taşlar aranıyor...")
        
        # YOLO ile taşları bul
        results = model(frozen, conf=0.45)[0]
        tiles = []
        
        # Sadece ROI içindekileri al
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            if rx1 < cx < rx2 and ry1 < cy < ry2:
                tile_img = frozen[y1:y2, x1:x2]
                if tile_img.size > 0:
                    tiles.append(tile_img)

        # Soldan sağa sırala (kullanım kolaylığı için)
        # (Burada basitçe X koordinatına göre sıralayamadık çünkü tile_img listesi koordinatsız
        # ama sorun değil, rastgele sorsun.)
        
        print(f"👉 {len(tiles)} taş bulundu. Sırayla soruyorum...")

        for i, tile in enumerate(tiles):
            # Sayıyı Kırp
            roi = crop_number_region(tile)
            if roi is None: continue

            # Göster
            disp = cv2.resize(roi, (200, 200))
            cv2.imshow("BU SAYI KAC?", disp)
            
            # Kullanıcıdan tuş bekle
            print(f"[{i+1}/{len(tiles)}] Bu sayı kaç? (x:atla)")
            
            saved = False
            while True:
                k = cv2.waitKey(0) & 0xFF
                label = None
                
                if k == ord('1'): label = "1"
                elif k == ord('2'): label = "2"
                elif k == ord('3'): label = "3"
                elif k == ord('4'): label = "4"
                elif k == ord('5'): label = "5"
                elif k == ord('6'): label = "6"
                elif k == ord('7'): label = "7"
                elif k == ord('8'): label = "8"
                elif k == ord('9'): label = "9"
                elif k == ord('0'): label = "10"
                elif k == ord('q'): label = "11"
                elif k == ord('w'): label = "12"
                elif k == ord('e'): label = "13"
                elif k == ord('x'): 
                    print("   -> Atlandı.")
                    break # While'dan çık, sonraki taşa geç
                elif k == 27: # ESC
                    print("Programdan çıkılıyor...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                if label:
                    # Klasörü oluştur
                    target_dir = os.path.join(SAVE_ROOT, label)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Kaydet
                    fname = f"{int(time.time()*1000)}.jpg"
                    cv2.imwrite(os.path.join(target_dir, fname), roi)
                    print(f"   ✅ Kaydedildi: {label} -> {fname}")
                    saved = True
                    break # Sonraki taşa geç

            if not saved and k == ord('x'):
                continue

        print("--- Bu ekran bitti, SPACE ile devam et ---\n")
        cv2.destroyWindow("BU SAYI KAC?")

cap.release()
cv2.destroyAllWindows()