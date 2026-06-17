from __future__ import annotations

"""Script de recuperation locale des datasets du projet.

Ce fichier ne lance pas l'entrainement.
Il sert uniquement a transformer des sources distantes en fichiers CSV locaux
faciles a relire et a reutiliser ensuite par `main.py`.
"""

from pathlib import Path

import pandas as pd
import sys


def ensure_dir(path: Path) -> Path:
    """Cree un dossier s'il n'existe pas encore."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def download_tabular_dataset(output_dir: Path) -> Path:
    """Recupere le dataset UCI Online Shoppers et l'exporte en CSV local."""

    # Chargement direct depuis le depot UCI via la bibliotheque `ucimlrepo`.
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=468)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if hasattr(targets, "columns"):
        target_values = targets.iloc[:, 0]
    else:
        target_values = targets

    # On reconstruit un DataFrame complet avec variables explicatives + cible.
    dataframe = features.copy()
    dataframe["Revenue"] = target_values.astype(int)

    output_path = ensure_dir(output_dir) / "online_shoppers_intention.csv"
    dataframe.to_csv(output_path, index=False)
    return output_path


def download_text_dataset(output_dir: Path, sample_size: int = 20000) -> Path:
    """Recupere un sous-ensemble de `NiyatiC/amazon_food_reviews` et l'exporte en CSV local."""

    from datasets import load_dataset

    # On telecharge la split d'entrainement puis on en garde un sous-ensemble raisonnable.
    dataset = load_dataset("NiyatiC/amazon_food_reviews", split="train")
    dataset = dataset.select(range(min(sample_size, len(dataset))))
    dataframe = dataset.to_pandas()[["Text", "Summary"]].dropna()
    dataframe = dataframe.rename(columns={"Text": "text", "Summary": "summary"})
    dataframe = dataframe[dataframe["summary"].str.len() > 0].copy()

    output_path = ensure_dir(output_dir) / "amazon_food_reviews.csv"
    dataframe.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Procedure complete de telechargement et d'export des datasets."""

    project_root = Path(__file__).resolve().parent.parent
    tabular_path = download_tabular_dataset(project_root / "data" / "tabular")
    text_path = download_text_dataset(project_root / "data" / "text", sample_size=20000)

    # Fashion-MNIST est deja telecharge automatiquement par torchvision pendant l'entrainement.
    out = sys.stdout.buffer
    out.write(f"Dataset tabulaire exporte vers : {tabular_path}\n".encode("utf-8", errors="replace"))
    out.write(f"Dataset textuel exporte vers : {text_path}\n".encode("utf-8", errors="replace"))
    out.write("Fashion-MNIST sera telecharge automatiquement lors de la premiere execution de la partie image.\n".encode("utf-8"))


if __name__ == "__main__":
    main()
