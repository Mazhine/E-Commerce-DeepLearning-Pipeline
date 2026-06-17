from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def ensure_dir(path: str | Path) -> Path:
    """Cree le dossier s'il n'existe pas et retourne son chemin."""

    path = Path(path)
    # `parents=True` cree aussi les dossiers parents si besoin.
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Sauvegarde des metadonnees d'experience au format JSON."""

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Sauvegarde le state_dict d'un modele PyTorch."""

    path = Path(path)
    ensure_dir(path.parent)
    # La pratique PyTorch classique consiste a sauvegarder le `state_dict`.
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, map_location: str = "cpu") -> torch.nn.Module:
    """Recharge un modele a partir d'un checkpoint pour evaluer ou reprendre l'entrainement."""

    # `map_location` permet de charger sur CPU meme si le modele venait d'un GPU.
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model
