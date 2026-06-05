"""
Data utilities for the cats-vs-dogs image dataset.

Handles the two data-quality problems described in the study:
    * corrupted / unreadable images  -> removed before training
    * class imbalance                -> balanced via class weights

Expected folder layout:

    dataset/
        training_set/{cats, dogs}/
        test_set/{cats, dogs}/
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.utils import class_weight
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32


def remove_corrupted_images(directory: str) -> int:
    """Delete unreadable / corrupted image files under ``directory``.

    Returns the number of files removed. This is the real data-cleaning
    step: every file is opened and verified, and anything that fails is
    deleted so the model only ever sees valid images.
    """
    removed = 0
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                with Image.open(path) as img:
                    img.verify()  # raises if the file is broken
            except Exception:
                os.remove(path)
                removed += 1
    return removed


def build_generators(train_dir: str, test_dir: str, backbone: str = "cnn"):
    """Create training (augmented) and validation generators.

    Parameters
    ----------
    backbone : {"cnn", "resnet"}
        Selects the correct preprocessing. The custom CNN expects pixels
        scaled to [0, 1]; ResNet50 expects its own ``preprocess_input``.

    The validation generator uses ``shuffle=False`` so that
    ``generator.classes`` lines up with ``model.predict`` — without this,
    the confusion matrix compares mismatched orders and is meaningless.
    """
    if backbone == "resnet":
        train_aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.1, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest",
        )
        val_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        train_aug = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.1, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest",
        )
        val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_aug.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary",
    )
    val_generator = val_aug.flow_from_directory(
        test_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary", shuffle=False,
    )
    return train_generator, val_generator


def compute_class_weights(train_generator) -> dict:
    """Balanced class weights from the training class distribution."""
    classes = train_generator.classes
    print("Class distribution (train):", dict(Counter(classes)))
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes,
    )
    return dict(enumerate(weights))
