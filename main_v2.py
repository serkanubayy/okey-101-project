import cv2
import numpy as np
from ultralytics import YOLO
from okey_zeka import OkeyZeka
import time

from number_crop import crop_number_region
from temporal_tracker import TemporalTracker
from tile_id_manager import TileIDManager
from cnn.number_predictor import predict_number, preprocess_image
from cnn.color_predictor import predict_color
from color_hsv import detect_color_hsv

# =============================
# KONFİGÜRASYON (GOLDEN STANDARD)
# =============================
model = YOLO("best.pt")
model.overrides["verbose"] = False
zeka = OkeyZeka()

CONF_TH = 0.45
IOU_TH = 0.45
IMG_SIZE = 640
STABLE_FRAMES_NEEDED = 8 
MIN_READY_TILES = 12
SAMPLE_TIMEOUT_SEC = 25.0 

tracker = TemporalTracker(window=7, min_votes=3, conf_floor=0.30, lock_after=4, lock_min_conf=0.60)
tile_id_manager = TileIDManager(iou_th=0.40, dist_th=95.0, ttl=10)

def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

def inside_roi(box, rx1, ry1, rx2, ry2):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return (rx1 < cx < rx2) and (ry1 < cy < ry2)

def crop_tile(img, box, pad=0): 
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2].copy()

def force_spatial_cleanup(tiles, iou_thresh=0.2):
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

# --- YARDIMCI: Kullanıcıdan Gösterge İste ---
def ask_indicator():
    print("\n" + "!"*40)
    print("Seko Baba, Gösterge Taşı Nedir?")
    print("Formatlar: r13 (Kırmızı 13), b5 (Mavi 5), k1 (Siyah 1), o10 (Turuncu 10)")
    raw = input("GİRİŞ YAP >> ").strip().lower()
    
    color_map = {'r': 'Red', 'b': 'Blue', 'k': 'Black', 'o': 'Orange'}
    
    # Varsayılan (Hata olursa)
    selected_color = "Red"
    selected_val = 13
    
    if len(raw) >= 2:
        c_char = raw[0]
        v_str = raw[1:]
        if c_char in color_map:
            selected_color = color_map[c_char]
            try:
                val = int(v_str)
                if 1 <= val <= 13:
                    selected_val = val
            except:
                pass
    
    print(f"✅ GÖSTERGE ALINDI: {selected_color} {selected_val}")
    print("!"*40 + "\n")
    return {"color": selected_color, "val": selected_val}

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

print("\n==============================")
print(" SEKO BABA FINAL V9 (SMART)")
print("==============================")

collecting = False
collect_frames = 0
collect_start = 0

while True:
    ret, img = cap.read()
    if not ret: break

    h, w, _ = img.shape
    rx1, rx2 = 100, w - 100
    ry1, ry2 = h // 2 - 150, h // 2 + 150

    color_ui = (0, 255, 0) if collecting else (0, 255, 255)
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color_ui, 3)
    cv2.putText(img, "SPACE: Baslat", (rx1, ry1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ui, 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if (not collecting) and key == ord(' '):
        collecting = True
        collect_frames = 0
        collect_start = time.time()
        tracker.reset()
        tile_id_manager = TileIDManager()

    if collecting:
        if time.time() - collect_start > SAMPLE_TIMEOUT_SEC:
            collecting = False
            print("❌ Zaman aşımı.")
            continue

        clean = img.copy() 
        results = model(clean, conf=CONF_TH, iou=IOU_TH, imgsz=IMG_SIZE, verbose=False)[0]

        boxes_input = []
        for b in results.boxes:
            xyxy = b.xyxy[0].tolist()
            if inside_roi(xyxy, rx1, ry1, rx2, ry2):
                area = box_area_xyxy(xyxy)
                if 2000 < area < 40000:
                    boxes_input.append({"box": xyxy, "conf": float(b.conf[0])})

        assigned = tile_id_manager.assign_ids(boxes_input)
        
        debug_crops = []

        for item in assigned:
            if item is None: continue
            tile_id, xyxy, _ = item if len(item) == 3 else (*item, 1.0)
            tile_img = crop_tile(clean, xyxy)
            if tile_img is None: continue

            hsv_pred = detect_color_hsv(tile_img)
            cnn_pred, cnn_conf = predict_color(tile_img)

            final_color = "Unknown"
            final_conf = 0.0

            if hsv_pred == cnn_pred and hsv_pred != "Unknown":
                final_color = hsv_pred
                final_conf = 1.0
            elif hsv_pred == "Unknown":
                final_color = cnn_pred
                final_conf = cnn_conf
            elif (hsv_pred in ["Red", "Orange"] and cnn_pred == "Black"):
                if cnn_conf > 0.85:
                    final_color = "Black"
                    final_conf = cnn_conf
                else:
                    final_color = hsv_pred
                    final_conf = 0.70
            elif (hsv_pred == "Black" and cnn_pred == "Red"):
                if cnn_conf > 0.90:
                    final_color = "Red"
                    final_conf = cnn_conf
                else:
                    final_color = "Black"
                    final_conf = 0.70
            else:
                if cnn_conf > 0.95:
                    final_color = cnn_pred
                    final_conf = cnn_conf
                else:
                    final_color = hsv_pred
                    final_conf = 0.60

            color = final_color
            color_conf = final_conf

            if color == "Green": continue

            num_roi = crop_number_region(tile_img)
            number_val, number_conf = (None, 0.0)

            if num_roi is not None:
                processed_debug = preprocess_image(num_roi)
                processed_debug = cv2.resize(processed_debug, (64, 64))
                number_val, number_conf = predict_number(num_roi)
                text = str(number_val) if number_val else "?"
                txt_color = (0, 0, 255) if number_conf > 0.60 else (0, 255, 255)
                cv2.putText(processed_debug, text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, txt_color, 3)
                debug_crops.append(processed_debug)

            tracker.update(tile_id, number_val, number_conf, color, color_conf)

        if debug_crops:
            limit = min(len(debug_crops), 16)
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
                        "id": tid, "box": p["box"], "color": tr.color, "val": int(tr.number), "is_joker": False
                    })
            
            stabilized = force_spatial_cleanup(raw_stabilized)
            stabilized.sort(key=lambda t: t["box"][1])
            
            rows = []
            if stabilized:
                current_row = [stabilized[0]]
                box_h = stabilized[0]["box"][3] - stabilized[0]["box"][1]
                for i in range(1, len(stabilized)):
                    if abs(stabilized[i]["box"][1] - current_row[-1]["box"][1]) < box_h * 0.6:
                        current_row.append(stabilized[i])
                    else:
                        rows.append(current_row)
                        current_row = [stabilized[i]]
                rows.append(current_row)
            
            final_sorted = []
            for r in rows:
                r.sort(key=lambda t: t["box"][0])
                final_sorted.extend(r)
            
            print("\n" + "="*40)
            print(f"✅ BULUNAN TAŞLAR ({len(final_sorted)}):")
            print(" | ".join([f"{t['color']} {t['val']}" for t in final_sorted]))
            
            if len(final_sorted) >= MIN_READY_TILES:
                gosterge_tasi = ask_indicator()

                # Zekadan sonuç sözlüğü (dict) alıyoruz artık
                sonuc = zeka.find_best_hand(final_sorted, gosterge_tasi)
                
                print(f"🃏 JOKER: {sonuc['joker_info']}")
                print(f"💰 SERİ PUANI: {sonuc['score']} (Baraj: 101)")
                print(f"👯 ÇİFT SAYISI: {sonuc['pair_count']} (Hedef: 5)")
                
                print("\n--- SERİ/PER DİZİLİMİ ---")
                if sonuc['best_hand']:
                    for j, p in enumerate(sonuc['best_hand']):
                        per_puan = sum(t.get('virtual_val', t['val']) for t in p)
                        per_str = ' - '.join([f"{t['color']}{t['val']}{t.get('virtual_str','')}" for t in p])
                        print(f"   Per {j+1}: {per_str}  [Puan: {per_puan}]")
                else:
                    print("   (Per bulunamadı)")

                print("\n--- ÇİFTLER ---")
                if sonuc['pairs']:
                    for p in sonuc['pairs']:
                        print(f"   [{p[0]['color']}{p[0]['val']}] - [{p[1]['color']}{p[1]['val']}{p[1].get('virtual_str','')}]")
                else:
                    print("   (Çift bulunamadı)")

            print("="*40 + "\n")
            collecting = False

    cv2.imshow("SEKO BABA V8 (GOLDEN)", img)

cap.release()
cv2.destroyAllWindows()