from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

from .io import ensure_dir


def plot_history(history: Dict[str, List[float]], title: str, output_path: str | Path) -> None:
    """Trace les courbes d'apprentissage pour l'annexe experimentale."""

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(10, 4))
    # Selon les experiences, certaines courbes peuvent ne pas exister.
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation loss")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Validation accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, labels, title: str, output_path: str | Path) -> None:
    """Enregistre une matrice de confusion lisible pour le rapport."""

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 5))
    # La heatmap rend la matrice de confusion plus facile a lire dans le rapport.
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
