# Brain Tumor Detection

## Proje Hakkında

Bu tez projesinde, MRI beyin görüntülerini dört sınıfta (glioma, meningioma, pituitary ve tümör yok) sınıflandırmak için VGG19 tabanlı bir derin öğrenme modeli geliştirilmiştir. Transfer öğrenmesi yaklaşımıyla oluşturulan modelin yanı sıra, görüntüleri ön işleme ve sınıflandırma sonuçlarını kullanıcıya sunma adımları bir web uygulamasında bir araya getirilmiştir. Uygulama, doktorların değerlendirme sürecini desteklemeyi amaçlar; tek başına tıbbi tanı aracı değildir.

## Özellikler

- MRI görüntülerini web arayüzünden yükleyip sınıflandırma
- Dört sınıf için tahmin: glioma, meningioma, pituitary ve tümör yok
- Tahmin edilen sınıfı, güven değerini ve sınıf olasılıklarını görüntüleme
- Eğitim verileri için beyin bölgesini kırpma, gürültü azaltma ve CLAHE ile kontrast iyileştirme adımlarını içeren ön işleme betiği
- VGG19 tabanlı transfer öğrenmesi ve fine-tuning ile model eğitimi
- Ek olarak ResNet50 ile eğitim ve test çalışmaları; testte yaklaşık %90 doğruluk oranı
- React frontend ve FastAPI backend

## Tez posteri
[sude_poster 1.pdf](https://github.com/user-attachments/files/33460083/sude_poster.1.pdf)


## Gereksinimler

- Python 3.10
- Node.js ve npm
- Model ağırlık dosyası: `backend/vgg19_best.weights.h5`

### Veri seti ve model ağırlıkları

Projede kullanılan [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) Kaggle üzerinden indirilebilir. Veri setini indirip çıkardıktan sonra ön işleme betiğinin beklediği `Training` ve `Testing` klasör yapısını sağlayın.

Model ağırlıkları (`vgg19_best.weights.h5`) Git deposunda tutulmaz. Ağırlıkları yeniden oluşturmak için `vgg19_model/preprocessin1.py` ve ardından `vgg19_model/train3_vgg19.py` betiklerini çalıştırın. Betiklerdeki veri seti, işlenmiş veri ve model çıktı yolları bilgisayara özel olarak tanımlıdır; çalıştırmadan önce bu yolları kendi klasör konumlarınıza göre düzenleyin. Eğitim tamamlandığında oluşan `vgg19_best.weights.h5` dosyasını `backend` klasörüne kopyalayın.

Model eğitimi için backend bağımlılıklarına ek olarak, backend sanal ortamı etkin durumdayken şu paketleri yükleyin:

```powershell
python -m pip install tqdm scikit-learn
```

## Kurulum ve çalıştırma

Backend ve frontend'i ayrı terminallerde çalıştırın.

### 1. Backend

Proje kök klasöründen PowerShell'de:

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Backend başladığında API dokümantasyonu `http://127.0.0.1:8000/docs` adresinde açılır. Backend'i başlatmadan önce model ağırlık dosyasının `backend/vgg19_best.weights.h5` konumunda bulunduğundan emin olun.

### 2. Frontend

Yeni bir terminal açıp proje kök klasöründen:

```powershell
cd frontend
npm install
npm run dev
```

Vite'ın terminalde gösterdiği yerel adresi (genellikle `http://localhost:5173`) tarayıcıda açın.

## Kullanım

1. Backend ve frontend'i başlatıp frontend'i tarayıcıda açın.
2. Bir MRI görüntüsü seçin.
3. **Tahmin** düğmesine basın.
4. Tahmin edilen sınıfı ve sınıf olasılıklarını görüntüleyin.

Frontend, backend API'sine `http://127.0.0.1:8000` adresinden bağlanır. Tahmin isteği göndermeden önce backend'in çalıştığını kontrol edin.
