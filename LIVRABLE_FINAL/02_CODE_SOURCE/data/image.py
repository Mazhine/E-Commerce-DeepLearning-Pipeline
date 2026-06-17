from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class ImageDataBundle:
    """Conteneur simple pour la partie image."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: List[str]


def build_fashion_mnist_loaders(
    root: str | Path,
    batch_size: int = 128,
    val_size: int = 5000,
) -> ImageDataBundle:
    """Prepare Fashion-MNIST pour la partie CNN.

    Fashion-MNIST est retenu car il reste simple pedagogiquement tout en gardant
    un lien naturel avec le scenario e-commerce du projet.
    """

    # `ToTensor()` convertit une image en tenseur PyTorch.
    # `Normalize()` recentre grossierement les valeurs des pixels.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    root = Path(root)
    # `download=True` evite d'avoir a recuperer manuellement le dataset.
    train_dataset = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    # On reserve une partie du jeu d'entrainement pour la validation.
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Les DataLoader prepareront les mini-lots pour le CNN.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = list(train_dataset.classes)
    return ImageDataBundle(train_loader, val_loader, test_loader, class_names)
