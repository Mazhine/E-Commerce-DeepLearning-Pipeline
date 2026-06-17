from __future__ import annotations

import torch
from torch import nn


def manual_cross_correlation2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Implementation manuelle de la correlation croisee 2D.

    Cette fonction sert a illustrer le mecanisme mathematique de base derriere
    la convolution telle qu'elle est enseignee dans le module.
    """

    # Le noyau glisse sur l'image. Sa taille controle la fenetre locale observee.
    h, w = kernel.shape
    output_h = x.shape[0] - h + 1
    output_w = x.shape[1] - w + 1
    output = torch.zeros((output_h, output_w), dtype=x.dtype)
    for i in range(output_h):
        for j in range(output_w):
            output[i, j] = (x[i : i + h, j : j + w] * kernel).sum()
    return output


def manual_max_pool2d(x: torch.Tensor, kernel_size: int = 2, stride: int = 2) -> torch.Tensor:
    """Implementation manuelle du max-pooling 2D."""

    output_h = 1 + (x.shape[0] - kernel_size) // stride
    output_w = 1 + (x.shape[1] - kernel_size) // stride
    output = torch.zeros((output_h, output_w), dtype=x.dtype)
    for i in range(output_h):
        for j in range(output_w):
            region = x[i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size]
            # Max-pooling : on conserve la plus grande valeur de la region.
            output[i, j] = region.max()
    return output


def manual_avg_pool2d(x: torch.Tensor, kernel_size: int = 2, stride: int = 2) -> torch.Tensor:
    """Implementation manuelle de l'average-pooling 2D."""

    output_h = 1 + (x.shape[0] - kernel_size) // stride
    output_w = 1 + (x.shape[1] - kernel_size) // stride
    output = torch.zeros((output_h, output_w), dtype=x.dtype)
    for i in range(output_h):
        for j in range(output_w):
            region = x[i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size]
            # Average-pooling : on remplace la region par sa moyenne.
            output[i, j] = region.mean()
    return output


class LeNetLikeCNN(nn.Module):
    """CNN inspire de LeNet, avec options pour l'etude comparative."""

    def __init__(
        self,
        num_classes: int = 10,
        num_filters: tuple[int, int] = (16, 32),
        padding: int = 0,
        stride: int = 1,
        pooling: str = "max",
        use_conv1x1: bool = False,
    ) -> None:
        super().__init__()

        # Le type de pooling est un parametre d'experience.
        pool_layer = nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2)
        # Une convolution 1x1 permet de recombiner les canaux sans changer la taille spatiale.
        conv1x1 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=1) if use_conv1x1 else nn.Identity()

        # Bloc d'extraction de caracteristiques.
        self.features = nn.Sequential(
            nn.Conv2d(1, num_filters[0], kernel_size=5, stride=stride, padding=padding),
            nn.ReLU(),
            pool_layer,
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=5, stride=1, padding=padding),
            nn.ReLU(),
            conv1x1,
            pool_layer,
        )

        # Bloc final de classification.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # `LazyLinear` deduit automatiquement la dimension d'entree au premier passage.
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ImageMLP(nn.Module):
    """MLP simple utilise comme baseline pour montrer ses limites sur les images."""

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        # Ce modele aplatit l'image et oublie sa structure spatiale.
        # C'est precisement sa limite face a un CNN.
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
