def crop_number_region(tile_img):
    """
    GENİŞ AÇI KIRPMA (12 ve 13 Sığsın Diye - GÜNCELLENDİ)
    """
    if tile_img is None: return None
    h, w = tile_img.shape[:2]
    if h < 20 or w < 20: return None

    # --- SEKO BABA AYARI ---
    # 2'nin ayağını kesmemek için altı (y2) biraz daha aşağı indirdik (0.68 -> 0.74).
    # 12 ve 13 geniş olduğu için yanları (x1, x2) biraz daha açtık (0.07 -> 0.04).
    
    y1 = int(h * 0.10)       # Üst boşluk (biraz daha temizledik)
    y2 = int(h * 0.74)       # ALT SINIR: Kritik nokta! 2'nin tabanı görünsün.
    
    x1 = int(w * 0.04)       # Sol boşluk
    x2 = int(w * 0.96)       # Sağ boşluk

    roi = tile_img[y1:y2, x1:x2]
    if roi.size == 0: return None

    return roi