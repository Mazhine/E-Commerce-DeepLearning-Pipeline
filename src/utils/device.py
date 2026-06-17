from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Retourne le meilleur device disponible pour l'entrainement.

    Cette fonction centralise la logique CPU/GPU afin d'eviter de la dupliquer
    dans plusieurs scripts. Le rapport peut ainsi montrer clairement que le
    projet verifie la coherence entre donnees et modele sur le meme device.
    """

    # Si CUDA est disponible, on privilegie le GPU pour accelerer l'entrainement.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch, device: torch.device):
    """Deplace un lot de donnees vers le device cible.

    Le lot peut etre un tuple `(x, y)` ou une structure plus riche selon la
    partie du projet. On conserve une implementation simple et pedagogique.
    """

    if isinstance(batch, (list, tuple)):
        # Cas le plus courant : le lot est une structure du type `(inputs, targets)`.
        return tuple(item.to(device) if hasattr(item, "to") else item for item in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch
