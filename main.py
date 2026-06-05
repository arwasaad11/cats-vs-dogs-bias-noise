"""
Biased & Noisy Dataset Analysis for Object Recognition (Cats vs Dogs).

Pipeline:
    1. Clean corrupted images (data noise).
    2. Build augmented training / validation generators.
    3. Train a baseline CNN and evaluate it (shows the bias).
    4. Train a ResNet50 transfer-learning model with class weights
       (reduces bias and noise impact) and evaluate it.
    5. Save confusion matrices for both models.

Designed to run on Google Colab or locally once the dataset is in place.
Set DATA_ROOT to wherever the dataset lives.

Run:
    python main.py
"""

from __future__ import annotations

import os

from src.data import (
    build_generators,
    compute_class_weights,
    remove_corrupted_images,
)
from src.evaluate import get_predictions, metrics, save_confusion
from src.model import build_cnn, build_resnet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# On Colab this is usually "/content/dataset".
DATA_ROOT = os.environ.get("DATA_ROOT", "data/dataset")
TRAIN_DIR = os.path.join(DATA_ROOT, "training_set")
TEST_DIR = os.path.join(DATA_ROOT, "test_set")
RESULTS_DIR = "results"

EPOCHS_CNN = 20
EPOCHS_RESNET = 10


def _callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5),
    ]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Clean noisy data
    removed = remove_corrupted_images(TRAIN_DIR) + remove_corrupted_images(TEST_DIR)
    print(f"Removed {removed} corrupted image(s).\n")

    # 2. Baseline CNN -----------------------------------------------------
    print("=== Baseline CNN ===")
    train_gen, val_gen = build_generators(TRAIN_DIR, TEST_DIR, backbone="cnn")
    class_weights = compute_class_weights(train_gen)

    cnn = build_cnn()
    cnn.fit(train_gen, epochs=EPOCHS_CNN, validation_data=val_gen,
            callbacks=_callbacks())

    y_true, y_pred = get_predictions(cnn, val_gen)
    print("CNN metrics:", {k: round(v, 4) for k, v in metrics(y_true, y_pred).items()})
    save_confusion(y_true, y_pred, "Confusion Matrix — Baseline CNN",
                   os.path.join(RESULTS_DIR, "01_cnn_confusion.png"))

    # 3. ResNet50 transfer learning + class weights -----------------------
    print("\n=== ResNet50 (transfer learning + class weights) ===")
    train_gen_r, val_gen_r = build_generators(TRAIN_DIR, TEST_DIR, backbone="resnet")

    resnet = build_resnet50(weights="imagenet")
    resnet.fit(train_gen_r, epochs=EPOCHS_RESNET, validation_data=val_gen_r,
               class_weight=class_weights, callbacks=_callbacks())

    y_true_r, y_pred_r = get_predictions(resnet, val_gen_r)
    print("ResNet50 metrics:", {k: round(v, 4) for k, v in metrics(y_true_r, y_pred_r).items()})
    save_confusion(y_true_r, y_pred_r, "Confusion Matrix — ResNet50",
                   os.path.join(RESULTS_DIR, "02_resnet_confusion.png"))

    print(f"\nDone. Confusion matrices saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
