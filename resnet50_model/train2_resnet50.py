import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 5
SEED = 42

TRAIN_PATH = r"C:\brain_tumor_tez\cleaned_dataset\Training"
MODEL_DIR = r"C:\brain_tumor_tez\model"

os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_best.keras")
BEST_WEIGHTS_PATH = os.path.join(MODEL_DIR, "resnet50_best.weights.h5")

np.random.seed(SEED)
tf.random.set_seed(SEED)

CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CATEGORIES)


def load_cleaned_data_rgb(folder_path, categories):
    X, y = [], []

    for idx, category in enumerate(categories):
        category_path = os.path.join(folder_path, category)

        if not os.path.exists(category_path):
            print("Klasör bulunamadı:", category_path)
            continue

        for img_name in tqdm(sorted(os.listdir(category_path)), desc=f"Yükleniyor: {category}", leave=False):
            img_path = os.path.join(category_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            X.append(img)
            y.append(idx)

    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def build_model(num_classes):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


print("Training verisi yükleniyor...")
X, y_raw = load_cleaned_data_rgb(TRAIN_PATH, CATEGORIES)

print("Toplam görüntü:", len(X))
print("Sınıf dağılımı:", np.bincount(y_raw))

X = preprocess_input(X.astype(np.float32))
y = tf.keras.utils.to_categorical(y_raw, NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y_raw
)

model, base_model = build_model(NUM_CLASSES)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    mode="max",
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

print("\n1. Aşama: Frozen ResNet50 eğitimi başlıyor...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_HEAD,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print("\n2. Aşama: Kontrollü fine-tuning başlıyor...")

base_model.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FINE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
best_model.save_weights(BEST_WEIGHTS_PATH)

val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=1)

print("\nKaydedildi:")
print(BEST_MODEL_PATH)
print(BEST_WEIGHTS_PATH)
print("Validation Accuracy:", val_acc)
print("Validation Loss:", val_loss)