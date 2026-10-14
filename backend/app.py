import os
import io
import webbrowser
import warnings
import numpy as np
from PIL import Image

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# =========================
# AYARLAR
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "vgg19_best.weights.h5")

IMG_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# =========================
# FASTAPI
# =========================
app = FastAPI(
    title="Brain Tumor API",
    description="Sınıflandırma",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL
# =========================
def build_models():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
        name="vgg19_base"
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation="softmax")(x)

    classifier = Model(inputs=inputs, outputs=outputs)
    return classifier


print("Weights var mi?:", os.path.exists(WEIGHTS_PATH))

model = build_models()

dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
model(dummy)

with warnings.catch_warnings(record=True) as load_warnings:
    warnings.simplefilter("always")
    model.load_weights(WEIGHTS_PATH, skip_mismatch=True)

for w in load_warnings:
    msg = str(w.message)
    if "could not be loaded" in msg and "objects" in msg:
        print("Uyari: bazı katmanlar agirlik dosyasinda yok, mevcut model agirliklariyla devam ediliyor.")
        break

print("Model hazir")

# =========================
# GÖRÜNTÜ HAZIRLAMA
# =========================
def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    return arr, np.array(img)

# =========================
# PREDICT
# =========================
def predict_image(image_bytes):
    img, _ = prepare_image(image_bytes)
    preds = model.predict(img, verbose=0)[0]

    idx = int(np.argmax(preds))

    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": float(preds[idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
        }
    }

# =========================
# ENDPOINTLER
# =========================
@app.get("/")
def root():
    return {
        "status": "API çalışıyor",
        "docs": "/docs"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Görüntü yükle")

    data = await file.read()
    return JSONResponse(predict_image(data))

# =========================
# RUN
# =========================
if __name__ == "__main__":
    url = "http://127.0.0.1:8000/docs"

    print("\nSunucu calisiyor!")
    print("Swagger:", url)

    webbrowser.open(url)

    uvicorn.run(app, host="0.0.0.0", port=8000)