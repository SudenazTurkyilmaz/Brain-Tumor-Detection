import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 32 # her adımda modele verilen görüntü sayısı
EPOCHS_STAGE1 = 10   # ilk aşama: base_model frozen
EPOCHS_STAGE2 = 10   # ikinci aşama: fine-tuning
SEED = 42 # aynı sonuçlara yakın sonuçlar almak için sabit değer

CLEANED_PATH = r"C:\brain_tumor_tez\cleaned_dataset"
TRAIN_PATH = os.path.join(CLEANED_PATH, "Training")

MODEL_DIR = r"C:\brain_tumor_tez\model"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "vgg19_best.keras")
BEST_WEIGHTS_PATH = os.path.join(MODEL_DIR, "vgg19_best.weights.h5")

np.random.seed(SEED)
tf.random.set_seed(SEED)

# VERİ YÜKLEME

def load_data(folder_path):
    X, y = [], []

    for idx, category in enumerate(CATEGORIES): #Her sınıf için bir indeks oluşturulur.
        category_path = os.path.join(folder_path, category)

        if not os.path.exists(category_path):
            print(f"Uyarı: Klasör bulunamadı -> {category_path}")
            continue

        for img_name in tqdm(sorted(os.listdir(category_path)), desc=f"Yükleniyor: {category}", leave=False):
            img_path = os.path.join(category_path, img_name) # Her görüntünün tam dosya yolu elde edilir.

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Görüntü gri tonlu olarak okunur.
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)#224x224 boyutuna getirilir.
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)# Gri görüntü RGB formatına çevrilir
            """Gri görüntünün RGB’ye çevrilme sebebi, VGG19 modelinin 3 kanallı giriş beklemesidir."""
            X.append(img) # Görüntü X listesine eklenir.
            y.append(idx)# Sınıf etiketi y listesine eklenir.

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)

    return X, y

""" Etiketler sayısal hale getirilir (indeks)
glioma     → 0
meningioma → 1
notumor    → 2
pituitary  → 3
"""
# MODEL OLUŞTURMA

def build_model():
    # Eğitim sırasında görüntülere rastgele dönüşümler uygulanır.Amaç modelin ezber yapmasını azaltmaktır.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"), #görüntüyü yatay çevirir
        layers.RandomRotation(0.05), #küçük açılarla döndürür
        layers.RandomZoom(0.10), #yakınlaştırma/uzaklaştırma yapar
        layers.RandomContrast(0.10), # kontrastı değiştirir
    ], name="data_augmentation")

    #ImageNet üzerinde önceden eğitilmiş VGG19 modeli alınır.
    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False #İlk aşamada VGG19’un kendi katmanları eğitilmez.

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)) # 224*224*3
    x = data_augmentation(inputs) #rastgele artırma
    x = preprocess_input(x) # vgg19 beklediği formata getirilir
    x = base_model(x, training=False) #görüntü özelliklerini çıkarır.
    x = layers.GlobalAveragePooling2D()(x) # çıkan özellik haritaları tek boyutlu vektöre dönüştürülür.
    x = layers.Dense(256, activation="relu")(x) # 256 nöronlu tam bağlantılı katman eklenir. sınıflandırmaya yardımcı olur
    x = layers.BatchNormalization()(x) #Eğitimi daha dengeli hale getirir. Model daha kararlı öğrenir.
    x = layers.Dropout(0.5)(x) # Nöronların %50’si eğitim sırasında rastgele devre dışı bırakılır.Amaç overfitting yani ezberlemeyi azaltmaktır
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x) #4 sınıflı çıkış katmanı oluşturulur.

    model = models.Model(inputs, outputs) #Girişten çıkışa kadar tüm model birleştirilir.
    return model, base_model

# ANA PROGRAM

print("Training verisi yükleniyor...")
X_train, y_train_raw = load_data(TRAIN_PATH)

print(f"Toplam görüntü sayısı: {len(X_train)}")
print("Sınıf dağılımı:")
for i, cat in enumerate(CATEGORIES):
    print(f"{cat}: {(y_train_raw == i).sum()}")
# Etiketler one-hot forma çevrilir
y_train = tf.keras.utils.to_categorical(y_train_raw, NUM_CLASSES)

# Sınıflar arasında veri sayısı farkı varsa dengelemek için ağırlık hesaplanır.
# Az verisi olan sınıfa daha fazla önem verilir.
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_raw),
    y=y_train_raw
)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass weights:")
print(class_weight_dict)

model, base_model = build_model() #VGG19 tabanlı model oluşturulur.


# 1. AŞAMA: FROZEN TRAINING

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Özeti:")
model.summary()

#Validation loss 3 epoch boyunca iyileşmezse eğitim durur.
early_stop_1 = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

#Validation loss en iyi olduğunda model kaydedilir.
checkpoint_1 = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

#Validation loss 2 epoch boyunca iyileşmezse learning rate yarıya düşürülür.
reduce_lr_1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("\n1. aşama eğitim başlıyor (base model frozen)...")
history1 = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS_STAGE1,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_weight=class_weight_dict,
    callbacks=[early_stop_1, checkpoint_1, reduce_lr_1],
    verbose=1
)

# 2. AŞAMA: FINE-TUNING

print("\nFine-tuning başlıyor...")

#Artık VGG19’un bazı katmanları da eğitilebilir.
base_model.trainable = True

# Sadece son 8 katmanı eğitime açılır, çoğunu dondur
for layer in base_model.layers[:-8]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop_2 = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

checkpoint_2 = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr_2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)
#Fine-tuning eğitimi yapılır
#VGG19’un son 8 katmanı eğitilir.
#Learning rate düşüktür.
#Model MRI görüntülerine daha iyi uyum sağlar.
history2 = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS_STAGE2,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_weight=class_weight_dict,
    callbacks=[early_stop_2, checkpoint_2, reduce_lr_2],
    verbose=1
)


best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
best_model.save_weights(BEST_WEIGHTS_PATH)

print("\nKaydedildi:")
print(BEST_MODEL_PATH)
print(BEST_WEIGHTS_PATH)