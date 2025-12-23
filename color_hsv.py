import cv2
import numpy as np

# ==========================================
# ROOKIES OKEY - RENK AYARI (SEKO BABA V7)
# ==========================================

COLOR_RANGES = {
    "Red": [
        # Kırmızı (Alt ton - Biraz daha daralttık ki Turuncuya kaymasın)
        ((0, 120, 100), (10, 255, 255)),
        # Kırmızı (Üst ton - Mora kaymasın)
        ((170, 120, 100), (180, 255, 255))
    ],
    "Orange": [
        # SARI / TURUNCU (Altın Rengi)
        # Hue 11-25 arası tam okey sarısıdır.
        ((11, 100, 100), (25, 255, 255))
    ],
    "Blue": [
        # MAVİ (Cyan/Mavi karışımı)
        ((85, 100, 80), (130, 255, 255))
    ],
    "Black": [
        # SİYAH (KRİTİK GÜNCELLEME)
        # Siyahın "Kırmızı" sanılmasını engellemek için Value (Parlaklık) sınırını artırdık.
        # Artık koyu gri taşları da Siyah kabul edecek.
        # Saturation (Renk Doygunluğu) 100'ün altındaysa renksizdir (siyahtır).
        ((0, 0, 0), (180, 100, 120))
    ],
    "Green": [
        # Yeşil (Varsa elensin diye)
        ((40, 50, 50), (80, 255, 255))
    ]
}

def _mask_for_ranges(hsv, ranges):
    mask_total = None
    for (lo, hi) in ranges:
        lo = np.array(lo, dtype=np.uint8)
        hi = np.array(hi, dtype=np.uint8)
        m = cv2.inRange(hsv, lo, hi)
        mask_total = m if mask_total is None else cv2.bitwise_or(mask_total, m)
    return mask_total

def detect_color_hsv(tile_bgr):
    """
    Siyahı Kırmızıdan, Sarıyı Turuncudan ayırır.
    """
    if tile_bgr is None: return "Unknown"

    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    
    # Sadece taşın göbeğine (sayının olduğu yere) bak
    # Kenarlardaki beyazlıklar yanıltmasın
    cx1, cy1 = int(w * 0.35), int(h * 0.25)
    cx2, cy2 = int(w * 0.65), int(h * 0.65)
    
    hsv_center = hsv[cy1:cy2, cx1:cx2]
    if hsv_center.size == 0: return "Unknown"

    scores = {}
    total_pixels = hsv_center.shape[0] * hsv_center.shape[1]

    for color, ranges in COLOR_RANGES.items():
        mask = _mask_for_ranges(hsv_center, ranges)
        scores[color] = np.sum(mask > 0)

    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]
    
    # --- ÇAKIŞMA KONTROLÜ (Siyah vs Kırmızı) ---
    # Eğer model kararsız kaldıysa ve Siyah puanı biraz bile varsa, Siyahı seç.
    # Çünkü Siyah genelde ışıktan dolayı Kırmızı gibi görünür ama tersi olmaz.
    if best_color == "Red" and scores["Black"] > (best_score * 0.4):
        return "Black"

    # Eğer hiçbiri uymuyorsa (çok az piksel varsa)
    if best_score < (total_pixels * 0.1): # %10 eşik
        return "Unknown"

    return best_color