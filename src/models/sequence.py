from __future__ import annotations

import random

import torch
from torch import nn


class RecurrentLanguageEncoder(nn.Module):
    """Bloc recurrent configurable en RNN, LSTM ou GRU."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        rnn_type: str = "gru",
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        # L'embedding transforme chaque token en vecteur dense.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # On choisit dynamiquement la cellule recurrente voulue.
        rnn_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[self.rnn_type]

        self.rnn = rnn_cls(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor):
        # Etapes standards :
        # tokens -> embeddings -> bloc recurrent
        embedded = self.embedding(x)
        return self.rnn(embedded)


class Seq2SeqEncoder(nn.Module):
    """Encodeur recurrent pour le systeme Seq2Seq."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        # `outputs` contient un etat par pas de temps.
        # `hidden` resume l'information finale de la sequence.
        return outputs, hidden


class Seq2SeqDecoder(nn.Module):
    """Decodeur recurrent avec projection vers le vocabulaire."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens: torch.Tensor, hidden: torch.Tensor):
        embedded = self.embedding(input_tokens)
        output, hidden = self.gru(embedded, hidden)
        # Projection des etats caches vers l'espace du vocabulaire.
        logits = self.output_layer(output)
        return logits, hidden


class Seq2SeqModel(nn.Module):
    """Mini systeme encodeur-decodeur pour generation de resumes."""

    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder, bos_idx: int, eos_idx: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        # Pendant l'entrainement, le teacher forcing remplace parfois la prediction
        # precedente du modele par le vrai token cible precedent.
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.output_layer.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

        _, hidden = self.encoder(src)
        decoder_input = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t : t + 1, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=-1)
            # Selon le teacher forcing, le prochain token d'entree du decodeur
            # sera le vrai token ou la prediction du modele.
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def greedy_decode(self, src: torch.Tensor, max_len: int = 20) -> torch.Tensor:
        """Decodage glouton pour la comparaison experimentale."""

        _, hidden = self.encoder(src)
        decoder_input = torch.full((src.size(0), 1), self.bos_idx, dtype=torch.long, device=src.device)
        generated = [decoder_input]

        for _ in range(max_len):
            logits, hidden = self.decoder(decoder_input, hidden)
            # En greedy decoding, on prend toujours le token le plus probable localement.
            decoder_input = logits.argmax(dim=-1)
            generated.append(decoder_input)

        return torch.cat(generated, dim=1)

    def beam_search_decode(self, src: torch.Tensor, max_len: int = 20, beam_width: int = 3) -> torch.Tensor:
        """Implementation simple et pedagogique d'un beam search.

        Pour garder le code lisible, cette version traite la premiere sequence du lot.
        Pour une production industrielle, on vectoriserait davantage cette logique.
        """

        self.eval()
        _, hidden = self.encoder(src[:1])
        sequences = [([self.bos_idx], 0.0, hidden)]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, current_hidden in sequences:
                if seq[-1] == self.eos_idx:
                    all_candidates.append((seq, score, current_hidden))
                    continue
                decoder_input = torch.tensor([[seq[-1]]], device=src.device)
                logits, next_hidden = self.decoder(decoder_input, current_hidden)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                top_scores, top_indices = torch.topk(log_probs, beam_width, dim=-1)
                # On genere plusieurs continuations candidates pour chaque sequence.
                for token_score, token_idx in zip(top_scores[0], top_indices[0]):
                    candidate_seq = seq + [int(token_idx.item())]
                    candidate_score = score + float(token_score.item())
                    all_candidates.append((candidate_seq, candidate_score, next_hidden))
            # On ne conserve que les hypotheses les plus prometteuses.
            ordered = sorted(all_candidates, key=lambda item: item[1], reverse=True)
            sequences = ordered[:beam_width]

        best_sequence = sequences[0][0]
        return torch.tensor(best_sequence, device=src.device).unsqueeze(0)
