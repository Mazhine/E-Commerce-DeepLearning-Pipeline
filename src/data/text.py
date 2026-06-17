from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# Role des tokens speciaux :
# - PAD : complete les sequences plus courtes
# - UNK : remplace les mots inconnus
# - BOS : indique le debut de la sequence cible
# - EOS : indique la fin de la sequence


def simple_tokenize(text: str) -> List[str]:
    """Tokenisation simple, volontairement transparente pedagogiquement."""

    return str(text).lower().strip().split()


class Vocabulary:
    """Vocabulaire minimaliste pour garder le pipeline explicable."""

    def __init__(self, tokens: Sequence[Sequence[str]], min_freq: int = 2) -> None:
        # On compte la frequence de chaque token pour eliminer les mots trop rares.
        counter = Counter(token for sentence in tokens for token in sentence)
        self.itos = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        for token, freq in counter.items():
            if freq >= min_freq:
                self.itos.append(token)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def encode(self, tokens: Sequence[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        # Conversion d'une liste de mots vers une liste d'indices entiers.
        indices = []
        if add_bos:
            indices.append(self.stoi[BOS_TOKEN])
        indices.extend(self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens)
        if add_eos:
            indices.append(self.stoi[EOS_TOKEN])
        return indices

    def __len__(self) -> int:
        return len(self.itos)


class Seq2SeqReviewDataset(Dataset):
    """Dataset texte -> resume pour la partie sequence."""

    def __init__(self, texts: List[List[int]], summaries: List[List[int]]) -> None:
        self.texts = texts
        self.summaries = summaries

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        # Chaque exemple renvoye est une paire (source, cible).
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.summaries[idx], dtype=torch.long)


@dataclass
class TextDataBundle:
    """Conteneur complet pour la partie texte."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocab: Vocabulary
    pad_idx: int
    bos_idx: int
    eos_idx: int


def collate_seq2seq(batch, pad_idx: int):
    """Assemble un mini-lot avec padding.

    Le padding est necessaire pour entrainer efficacement des modeles recursifs
    sur des sequences de longueurs variables.
    """

    # On separe les sources et les cibles du mini-lot.
    src_batch, tgt_batch = zip(*batch)
    # `pad_sequence` aligne les longueurs pour pouvoir empiler les tenseurs.
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def load_text_dataset(csv_path: str | Path, text_col: str = "text", summary_col: str = "summary") -> pd.DataFrame:
    """Charge un corpus d'avis clients."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Corpus textuel introuvable : {csv_path}")
    df = pd.read_csv(csv_path)
    return df[[text_col, summary_col]].dropna()


def load_hf_amazon_food_reviews(
    split: str = "train",
    sample_size: int | None = 20000,
) -> pd.DataFrame:
    """Charge le corpus `NiyatiC/amazon_food_reviews` depuis Hugging Face.

    Le mapping retenu pour le projet est :
    - `Text` comme sequence source
    - `Summary` comme sequence cible

    Ce choix est pertinent pour une tache de resume court en e-commerce.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Le package `datasets` est necessaire pour charger le corpus Hugging Face.") from exc

    dataset = load_dataset("NiyatiC/amazon_food_reviews", split=split)
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    dataframe = dataset.to_pandas()
    dataframe = dataframe[["Text", "Summary"]].dropna()
    dataframe = dataframe.rename(columns={"Text": "text", "Summary": "summary"})
    dataframe = dataframe[dataframe["summary"].str.len() > 0].copy()
    return dataframe


def build_text_loaders(
    dataframe: pd.DataFrame,
    text_col: str = "text",
    summary_col: str = "summary",
    batch_size: int = 32,
    max_text_len: int = 80,
    max_summary_len: int = 20,
) -> TextDataBundle:
    """Prepare les donnees textuelles pour les modeles recursifs et Seq2Seq."""

    # On tronque les textes pour garder un cout d'entrainement raisonnable.
    text_tokens = [simple_tokenize(text)[:max_text_len] for text in dataframe[text_col].astype(str).tolist()]
    summary_tokens = [simple_tokenize(text)[:max_summary_len] for text in dataframe[summary_col].astype(str).tolist()]

    # Le vocabulaire est bati sur les tokens observes dans les textes et les resumes.
    vocab = Vocabulary(text_tokens + summary_tokens)

    # La source recoit seulement EOS.
    # La cible recoit BOS + EOS pour l'apprentissage du decodeur.
    text_encoded = [vocab.encode(tokens, add_eos=True) for tokens in text_tokens]
    summary_encoded = [vocab.encode(tokens, add_bos=True, add_eos=True) for tokens in summary_tokens]

    combined = list(zip(text_encoded, summary_encoded))
    # Split train / validation / test.
    train_data, test_data = train_test_split(combined, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = Seq2SeqReviewDataset([x for x, _ in train_data], [y for _, y in train_data])
    val_dataset = Seq2SeqReviewDataset([x for x, _ in val_data], [y for _, y in val_data])
    test_dataset = Seq2SeqReviewDataset([x for x, _ in test_data], [y for _, y in test_data])

    pad_idx = vocab.stoi[PAD_TOKEN]
    # Le collate function definit comment former un mini-lot a partir d'exemples individuels.
    collate_fn = lambda batch: collate_seq2seq(batch, pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return TextDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
        pad_idx=pad_idx,
        bos_idx=vocab.stoi[BOS_TOKEN],
        eos_idx=vocab.stoi[EOS_TOKEN],
    )
