"""
Model architectures.

    build_cnn       - a small CNN trained from scratch (baseline)
    build_resnet50  - ResNet50 transfer learning (improved model)
"""

from __future__ import annotations

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128


def _compile(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn():
    """Baseline convolutional network trained from scratch."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation="relu"),
        Dense(1, activation="sigmoid"),  # binary: cat vs dog
    ])
    return _compile(model)


def build_resnet50(weights: str | None = "imagenet"):
    """ResNet50 transfer-learning model.

    GlobalAveragePooling2D is used instead of Flatten to keep the head
    small and reduce overfitting. The convolutional base is frozen so we
    only train the new classification head.
    """
    base_model = ResNet50(
        weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    return _compile(model)
