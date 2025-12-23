import cv2
import os
import glob
from ultralytics import YOLO
from number_crop import crop_number_region

# =============================
# AYARLAR
# =============================
INPUT_FOLDER = "ham_fotolar"   # Ekran görüntülerini buraya at
OUTPUT_ROOT = "dataset/train"  # Kesilenler buraya gidecek
MODEL_PATH = "best.pt"         # Taşları bulan YOLO modeli

if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

model = YOLO(MODEL_PATH)

# Klasördeki tüm resimleri bul
images = glob.glob(os.path.join(INPUT_FOLDER, "*.*"))
print(f"\n✂️  {len(images)} adet fotoğraf işlenecek...")

for img_path in images:
    frame = cv2.imread(img_path)
    if frame is None: continue
    
    print(f"\n📂 İşleniyor: {img_path}")
    
    # YOLO ile taşları bul
    results = model(frame, conf=0.45)[0]
    
    # Koordinatlarına göre soldan sağa sırala
    boxes = []
    for b in results.boxes:
        boxes.append(b.xyxy[0].tolist())
    
    # X koordinatına göre sırala (Soldan sağa sorsun diye)
    boxes.sort(key=lambda x: x[0])

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Taşı kes
        tile_img = frame[y1:y2, x1:x2]
        
        # Sayıyı kes (Senin number_crop fonksiyonunu kullanır)
        roi = crop_number_region(tile_img)
        
        if roi is None: continue

        # --- GÖSTER VE SOR ---
        # Resmi biraz büyüt ki rahat gör
        disp = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("BU SAYI KAC?", disp)
        
        print(f"   👉 Taş {i+1}: Hangi sayı? (0=10, q=11, w=12, e=13, x=atla, ESC=çık)")
        
        while True:
            k = cv2.waitKey(0) & 0xFF
            label = None
            
            if k == 27: exit() # ESC
            elif k == ord('x'): break # Atla
            
            if k in [ord(str(n)) for n in range(1, 10)]: label = chr(k)
            elif k == ord('0'): label = "10"
            elif k == ord('q'): label = "11"
            elif k == ord('w'): label = "12"
            elif k == ord('e'): label = "13"
            
            if label:
                save_dir = os.path.join(OUTPUT_ROOT, label)
                os.makedirs(save_dir, exist_ok=True)
                
                # Benzersiz isimle kaydet
                import uuid
                fname = f"{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(os.path.join(save_dir, fname), roi)
                print(f"      ✅ {label} olarak kaydedildi.")
                break
        
    print("--- Bu resim bitti ---")

cv2.destroyAllWindows()
print("\n🎉 TÜM RESİMLER KESİLDİ! dataset/train klasörünü kontrol et.")