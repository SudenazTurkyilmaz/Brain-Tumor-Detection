import os
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 256

# Kaynak veri yolu
BASE_PATH = r"C:\brain_tumor_tez\dataset"

# Çıktı veri yolu
CLEANED_PATH = r"C:\brain_tumor_tez\cleaned_dataset"


def crop_img_safe(img_bgr, add_pixels=0):
    """Kontura göre kırpar. Kontur yoksa orijinali döndürür.
    Bu fonksiyonun amacı görüntüdeki boş siyah alanları azaltıp beyin bölgesini kırpmaktır."""
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # Görüntü gri tona çevrilir
    gray = cv2.GaussianBlur(gray, (3, 3), 0) #Hafif bulanıklaştırma uygulanır,Görüntüdeki küçük gürültüler azaltılır.

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1] #Eşikleme yapılır(Piksel değeri 45’ten büyük olan bölgeler beyaz, küçük olan bölgeler siyah yapılır.), yani siyah arka plan ile beyin bölgesi ayrılır.
    thresh = cv2.erode(thresh, None, iterations=2) #Beyaz bölgelerdeki küçük gürültüler temizlenir(erezyon).
    thresh = cv2.dilate(thresh, None, iterations=2)#Erozyondan sonra kalan ana beyin bölgesi tekrar genişletilir.

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Beyaz bölgelerin dış sınırları bulunur.
    if not cnts:
        return img_bgr

    c = max(cnts, key=cv2.contourArea) #En büyük kontur bulunur.
    #Konturun uç noktaları bulunur
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    h, w = img_bgr.shape[:2] # Görüntünün yüksekliği ve genişliği alınır.(boyutu)
    # Kırpma koordinatları hesaplanır
    x1 = max(extLeft[0] - add_pixels, 0)
    x2 = min(extRight[0] + add_pixels, w)
    y1 = max(extTop[1] - add_pixels, 0)
    y2 = min(extBot[1] + add_pixels, h)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return img_bgr

    return img_bgr[y1:y2, x1:x2].copy()


def advanced_preprocess_gray(img_bgr, out_size=IMG_SIZE):
    """Gri ton + median blur + CLAHE + resize uygular."""
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # Renkli görüntü tek kanallı gri görüntüye çevrilir.
    gray = cv2.medianBlur(gray, 3) # Görüntüdeki küçük gürültüler azaltılır.

    """MRI görüntülerinde bazı bölgeler çok koyu veya çok düşük kontrastlı olabilir. 
    CLAHE sayesinde tümör, doku ve beyin sınırları daha belirgin hale gelir."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #CLAHE, görüntünün kontrastını artırır.
    gray = clahe.apply(gray)

    #Tüm görüntüler aynı boyuta getirilir:
    gray = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return gray


def preprocess_folder(src_root, dst_root):
    """
    src_root içindeki sınıf klasörlerini okuyup
    dst_root içine aynı klasör yapısıyla kaydeder.
    """
    if not os.path.exists(src_root):
        print(f"Hata: Kaynak klasör bulunamadı -> {src_root}")
        return

    classes = sorted(
        [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    )

    for cls in classes:
        src_cls = os.path.join(src_root, cls)
        dst_cls = os.path.join(dst_root, cls)
        os.makedirs(dst_cls, exist_ok=True)

        img_names = sorted(os.listdir(src_cls))

        for img_name in tqdm(img_names, desc=f"Preprocess {os.path.basename(src_root)}:{cls}"):
            src_path = os.path.join(src_cls, img_name)

            img = cv2.imread(src_path)
            if img is None:
                print(f"Okunamadı: {src_path}")
                continue

            cropped = crop_img_safe(img, add_pixels=0)
            processed = advanced_preprocess_gray(cropped, out_size=IMG_SIZE)

            if processed is None:
                print(f"İşlenemedi: {src_path}")
                continue

            dst_path = os.path.join(dst_cls, img_name)
            cv2.imwrite(dst_path, processed)


def main():
    train_src = os.path.join(BASE_PATH, "Training")
    test_src = os.path.join(BASE_PATH, "Testing")

    train_dst = os.path.join(CLEANED_PATH, "Training")
    test_dst = os.path.join(CLEANED_PATH, "Testing")

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    print("✅ Training preprocess başlıyor...")
    preprocess_folder(train_src, train_dst)

    print("✅ Testing preprocess başlıyor...")
    preprocess_folder(test_src, test_dst)

    print(f"✅ Preprocess bitti: {CLEANED_PATH}")


if __name__ == "__main__":
    main()