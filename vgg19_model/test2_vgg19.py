import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = r"C:\brain_tumor_tez\model\vgg19_best.keras"
TEST_DIR = r"C:\brain_tumor_tez\cleaned_dataset\Testing"

RESULTS_DIR = r"C:\brain_tumor_tez\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate_model():
    model = load_model(MODEL_PATH)
    print("Model yüklendi.\n")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = test_ds.class_names

    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    print("\nClassification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:\n")
    print(cm)

    # Karmaşıklık matrisi görseli
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Karmaşıklık Matrisi")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.tight_layout()

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nKarmaşıklık matrisi kaydedildi: {cm_path}")


if __name__ == "__main__":
    evaluate_model()

