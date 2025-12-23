import cv2
import numpy as np
from ultralytics import YOLO
from okey_zeka import OkeyZeka
import re
import time

from number_crop import crop_number_region
from temporal_tracker import TemporalTracker
from tile_id_manager import TileIDManager
from cnn.number_predictor import predict_number
from cnn.color_predictor import predict_color
from tile_validator import validate_tiles
from color_hsv import detect_color_hsv

# =============================
# KONFİGÜRASYON
# =============================
PRINT_COOLDOWN = 0.6
LAST_PRINT_TIME = 0.0

# Model ve Zeka
model = YOLO("best.pt")
model.overrides["verbose"] = False
zeka = OkeyZeka()

# Parametreler
CONF_TH = 0.45
IOU_TH = 0.45
IMG_SIZE = 640

# İstikrar Ayarları
STABLE_FRAMES_NEEDED = 8  # Biraz daha hızlandırdım
MIN_READY_TILES = 12
SAMPLE_TIMEOUT_SEC = 10.0

# Trackers
tracker = TemporalTracker(window=9, min_votes=3, conf_floor=0.30, lock_after=5, lock_min_conf=0.60)
tile_id_manager = TileIDManager(iou_th=0.40, dist_th=95.0, ttl=10)

# =============================
# YARDIMCI FONKSİYONLAR
# =============================

# DİKKAT: ultra_netlestir fonksiyonunu kaldırdık!
# Dijital ekranda keskinleştirme gürültü yaratır.

def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

def inside_roi(box, rx1, ry1, rx2, ry2):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return (rx1 < cx < rx2) and (ry1 < cy < ry2)

def crop_tile(img, box, pad=5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2].copy()

# Çakışma Önleyici
def force_spatial_cleanup(tiles, iou_thresh=0.3):
    if not tiles: return []
    boxes = np.array([t["box"] for t in tiles])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    keep = []
    indices = np.arange(len(tiles))
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(tiles[current])
        if len(indices) == 1: break
        
        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[current] + areas[indices[1:]] - inter)
        indices = indices[1:][iou <= iou_thresh]
        
    return keep

# =============================
# ANA DÖNGÜ
# =============================
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Mümkünse otomaik odağı kapat, elle ayarla

print("\n==============================")
print(" SEKO BABA OKEY V4.5 (RÖNTGEN)")
print("==============================")
print(" SPACE: Başlat | q: Çıkış")
print("==============================\n")

collecting = False
collect_start_time = 0.0
collect_frames = 0

while True:
    ret, img = cap.read()
    if not ret: break

    h, w, _ = img.shape
    rx1, rx2 = 100, w - 100
    ry1, ry2 = h // 2 - 150, h // 2 + 150

    color_ui = (0, 255, 0) if collecting else (0, 255, 255)
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color_ui, 3)
    
    status_text = f"ANALIZ: {collect_frames}/{STABLE_FRAMES_NEEDED}" if collecting else "BASLAMAK ICIN SPACE BAS"
    cv2.putText(img, status_text, (rx1, ry1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ui, 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if (not collecting) and key == ord(' '):
        collecting = True
        collect_start_time = time.time()
        collect_frames = 0
        tracker.reset()
        if hasattr(tile_id_manager, "reset"): tile_id_manager.reset()
        else: tile_id_manager = TileIDManager()

    if collecting:
        if time.time() - collect_start_time > SAMPLE_TIMEOUT_SEC:
            collecting = False
            print("❌ Zaman aşımı!")
            continue

        # DİKKAT: ultra_netlestir yok! Doğal resim.
        clean = img.copy() 
        results = model(clean, conf=CONF_TH, iou=IOU_TH, imgsz=IMG_SIZE, verbose=False)[0]

        boxes_input = []
        for b in results.boxes:
            xyxy = b.xyxy[0].tolist()
            if inside_roi(xyxy, rx1, ry1, rx2, ry2):
                area = box_area_xyxy(xyxy)
                if 2000 < area < 40000:
                    conf = float(b.conf[0])
                    boxes_input.append({"box": xyxy, "conf": conf})

        assigned = tile_id_manager.assign_ids(boxes_input)
        
        # DEBUG GÖRÜNTÜSÜ İÇİN LİSTE
        debug_crops = []

        for item in assigned:
            if item is None: continue
            tile_id, xyxy, _ = item if len(item) == 3 else (*item, 1.0)
            tile_img = crop_tile(clean, xyxy)
            if tile_img is None: continue

            # --- RENK ---
            cnn_color, cnn_conf = predict_color(tile_img)
            hsv_color = detect_color_hsv(tile_img)

            # HSV patron
            if hsv_color != "Unknown":
                color = hsv_color
                color_conf = 1.0 
            else:
                color = cnn_color
                color_conf = cnn_conf

            if color == "Green": continue 

            # --- SAYI ---
            num_roi = crop_number_region(tile_img)
            number_val, number_conf = (None, 0.0)

            if num_roi is not None:
                # DEBUG: Sayı ROI'sini gri yapıp listeye ekle
                gray_debug = cv2.cvtColor(num_roi, cv2.COLOR_BGR2GRAY)
                gray_debug = cv2.resize(gray_debug, (64, 64))
                gray_debug_bgr = cv2.cvtColor(gray_debug, cv2.COLOR_GRAY2BGR)
                
                # Resmin üstüne tahmin edilen sayıyı yaz (Kırmızı renk)
                pred_temp, _ = predict_number(num_roi)
                cv2.putText(gray_debug_bgr, str(pred_temp), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                debug_crops.append(gray_debug_bgr)

                number_val, number_conf = predict_number(num_roi)

            tracker.update(tile_id, number_val, number_conf, color, color_conf)
        
        # --- DEBUG PENCERESİ: CNN NE GÖRÜYOR? ---
        if debug_crops:
            # En fazla 10 tanesini yan yana dizip göster
            limit = min(len(debug_crops), 14)
            stack = np.hstack(debug_crops[:limit])
            cv2.imshow("CNN NE GORUYOR? (Rontgen)", stack)

        collect_frames += 1

        if collect_frames >= STABLE_FRAMES_NEEDED:
            raw_stabilized = []
            
            for p in tile_id_manager.prev:
                tid = p["id"]
                tr = tracker.get(tid)
                if tr.number is not None and tr.color != "Unknown":
                    raw_stabilized.append({
                        "id": tid,
                        "box": p["box"],
                        "color": tr.color,
                        "val": int(tr.number),
                        "is_joker": False
                    })
            
            stabilized = force_spatial_cleanup(raw_stabilized, iou_thresh=0.20)
            stabilized.sort(key=lambda t: t["box"][1])

            rows = []
            if stabilized:
                current_row = [stabilized[0]]
                box_h = stabilized[0]["box"][3] - stabilized[0]["box"][1]
                row_threshold = box_h * 0.6 
                
                for i in range(1, len(stabilized)):
                    prev_y = current_row[-1]["box"][1]
                    curr_y = stabilized[i]["box"][1]
                    
                    if abs(curr_y - prev_y) < row_threshold:
                        current_row.append(stabilized[i])
                    else:
                        rows.append(current_row)
                        current_row = [stabilized[i]]
                rows.append(current_row)

            final_sorted_list = []
            for row in rows:
                row.sort(key=lambda t: t["box"][0])
                final_sorted_list.extend(row)
            
            stabilized = final_sorted_list

            print("\n" + "="*40)
            print(f"✅ TESPİT EDİLEN TAŞLAR ({len(stabilized)} adet):")
            line = " | ".join([f"{t['color']} {t['val']}" for t in stabilized])
            print(line)

            if len(stabilized) >= MIN_READY_TILES:
                perler, puan = zeka.find_best_hand(stabilized)
                print(f"\n--- HESAPLANAN EN İYİ EL (Puan: {puan}) ---")
                if not perler:
                    print("Per bulunamadı.")
                else:
                    for j, per in enumerate(perler):
                        isimler = [f"{t['color']}{t['val']}" for t in per]
                        print(f"Per {j+1}: {' - '.join(isimler)}")
            else:
                print("⚠️ Yeterli taş okunamadı.")
            
            print("="*40 + "\n")
            collecting = False

    cv2.imshow("SEKO BABA OKEY V4.5 (RÖNTGEN)", img)

cap.release()
cv2.destroyAllWindows()