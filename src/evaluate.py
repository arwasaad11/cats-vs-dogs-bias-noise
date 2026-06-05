"""
Evaluation: aligned predictions, metrics and confusion-matrix plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_predictions(model, val_generator):
    """Return aligned (y_true, y_pred).

    Relies on the validation generator being created with shuffle=False,
    so generator.classes matches the prediction order.
    """
    y_proba = model.predict(val_generator).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    y_true = val_generator.classes
    return y_true, y_pred


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def save_confusion(y_true, y_pred, title: str, path: str,
                   labels=("Cat", "Dog")) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
