import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report

MODEL_PATH = r"C:\brain_tumor_tez\model\resnet50_best.keras"
TEST_DIR = r"C:\brain_tumor_tez\cleaned_dataset\Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate_model():
    model = load_model(MODEL_PATH)
    print("Model yüklendi.\n")
    model.summary()

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = test_ds.class_names
    print("Sınıflar:", class_names)

    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == "__main__":
    evaluate_model()