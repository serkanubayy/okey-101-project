import cv2
import numpy as np

def safe_crop(img, box, pad=6):
    """
    box: [x1,y1,x2,y2]
    pad: kenarlardan biraz pay bırak (taş tam çıksın diye)
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2].copy()

def normalize_tile(tile, size=160):
    """
    taşı CNN/HSV için sabit boyuta getirir
    """
    return cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
