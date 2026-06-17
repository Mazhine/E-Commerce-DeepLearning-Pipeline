from __future__ import annotations

import torch
from torch import nn


class SequentialMLP(nn.Module):
    """Version MLP fondee sur nn.Sequential.

    Cette version est volontairement concise. Elle permet d'illustrer la construction
    rapide d'un reseau lorsque la logique de propagation avant reste lineaire.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Chaque bloc cache suit le schema classique :
            # lineaire -> ReLU -> dropout
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CustomMLP(nn.Module):
    """Version MLP fondee sur une classe personnalisee.

    Cette variante est pedagogiquement utile car elle montre comment controler
    explicitement les couches, la propagation avant et d'eventuelles extensions.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.2):
        super().__init__()
        # `ModuleList` est utile quand on veut gerer explicitement une liste de couches.
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On applique chaque couche cachee dans l'ordre.
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
        return self.output_layer(x)


def initialize_weights(model: nn.Module, strategy: str = "xavier") -> None:
    """Applique une strategie d'initialisation sur les couches lineaires.

    Le cahier des charges exige de comparer plusieurs methodes. Cette fonction
    centralise les variantes pour que l'experimentation reste propre.
    """

    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Le choix d'initialisation influence souvent la stabilite et la vitesse
            # de convergence au debut de l'entrainement.
            if strategy == "gaussian":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif strategy == "constant":
                nn.init.constant_(module.weight, 0.05)
            elif strategy == "xavier":
                nn.init.xavier_uniform_(module.weight)
            else:
                raise ValueError(f"Strategie d'initialisation inconnue : {strategy}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
