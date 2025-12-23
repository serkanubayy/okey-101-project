import cv2
import os
import time

# Kayıt klasörü oluştur
if not os.path.exists("egitim_fotolari"):
    os.makedirs("egitim_fotolari")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

count = 0
print("Seko Baba Veri Toplama Aracı")
print("'s' tuşuna basarak fotoğraf çek.")
print("'q' tuşuna basarak çık.")

while True:
    success, img = cap.read()
    if not success: break

    cv2.imshow("Veri Toplayici", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Fotoğrafı kaydet
        filename = f"egitim_fotolari/seko_okey_{int(time.time())}.jpg"
        cv2.imwrite(filename, img)
        print(f"Fotoğraf Kaydedildi: {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()