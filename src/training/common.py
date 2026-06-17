from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.utils.device import move_batch_to_device
from src.utils.metrics import classification_metrics


@dataclass
class TrainingResult:
    """Resultat standard d'un entrainement.

    `history` contient les courbes.
    `best_state_dict` contient les meilleurs poids.
    """

    history: Dict[str, List[float]]
    best_state_dict: Dict


def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 1e-3,
) -> TrainingResult:
    """Boucle d'entrainement generique pour MLP et CNN."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_acc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # On deplace le mini-lot sur le bon device.
            inputs, targets = move_batch_to_device(batch, device)
            # Pipeline d'optimisation standard.
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation sur validation a la fin de chaque epoque.
        val_loss, val_metrics = evaluate_classifier(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss / max(len(train_loader), 1))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])

        # On garde les meilleurs poids selon l'accuracy de validation.
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return TrainingResult(history=history, best_state_dict=best_state or {})


def evaluate_classifier(model: nn.Module, data_loader, device: torch.device, criterion=None):
    """Evaluation standard d'un classifieur."""

    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = move_batch_to_device(batch, device)
            logits = model(inputs)
            if criterion is not None:
                total_loss += criterion(logits, targets).item()
            # La classe predite est celle dont le score est maximal.
            predictions = logits.argmax(dim=1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    metrics = classification_metrics(y_true, y_pred)
    avg_loss = total_loss / max(len(data_loader), 1)
    return avg_loss, metrics


def train_seq2seq(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    pad_idx: int,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    gradient_clip: float = 1.0,
) -> TrainingResult:
    """Boucle d'entrainement pour le mini systeme Seq2Seq."""

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            outputs = model(src, tgt)
            # On ignore le premier token qui sert de signal de depart.
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            # Le gradient clipping stabilise l'entrainement recurrent.
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_loss += loss.item()

        val_loss = evaluate_seq2seq(model, val_loader, device, pad_idx)
        train_loss = train_loss / max(len(train_loader), 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return TrainingResult(history=history, best_state_dict=best_state or {})


def evaluate_seq2seq(model: nn.Module, data_loader, device: torch.device, pad_idx: int) -> float:
    """Calcule une perte moyenne sur l'ensemble de validation/test."""

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            # En evaluation, on coupe le teacher forcing.
            outputs = model(src, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)
