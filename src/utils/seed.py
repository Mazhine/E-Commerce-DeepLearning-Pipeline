from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixe les graines aleatoires pour des experiences plus reproductibles."""

    # Sans seed fixe, deux executions du meme code peuvent diverger legerement.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
