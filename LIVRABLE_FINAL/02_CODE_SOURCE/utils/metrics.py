from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calcule les metriques de classification demandees dans le cahier des charges."""

    # `weighted` permet de tenir compte d'un desequilibre possible entre classes.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def confusion_matrix_array(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Retourne la matrice de confusion pour analyse et visualisation."""

    return confusion_matrix(y_true, y_pred)
