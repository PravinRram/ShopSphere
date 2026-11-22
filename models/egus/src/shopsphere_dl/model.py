"""TwoBranchNet: a GRU over each customer's event sequence + an MLP over customer-level facts.

    events (right-aligned, length L) -> per-event embedding -> [start token] + events
        -> GRU (packed, so padding is never read) -> pooling -> sequence summary (hidden)
    static features -> embeddings + numeric -> MLP -> static summary (128)
    concat -> MLP head -> one logit (sigmoid gives the purchase probability)

The learned start token is always at position 0, so a customer with no events still has a
sequence of length 1: the 'no history' case is handled by the model, not by a special rule.

Switches for the ablations in the training plan:
    use_sequence=False -> static-only MLP (E2, E3)
    use_static=False   -> sequence-only (E6)
    pooling in {"attention", "mean", "last"} (E7)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def emb_dim(cardinality: int) -> int:
    return int(min(16, max(2, (cardinality + 1) // 2)))


class StaticBranch(nn.Module):
    def __init__(self, n_numeric: int, cat_cards: list[int], hidden=(256, 128), dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c)) for c in cat_cards])
        d = n_numeric + sum(emb_dim(c) for c in cat_cards)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, num, cat):
        x = torch.cat([num] + [e(cat[:, j]) for j, e in enumerate(self.embs)], dim=1)
        return self.mlp(x)


class SequenceBranch(nn.Module):
    def __init__(self, n_numeric: int, n_time: int, cat_cards: list[int], hidden=64, layers=2,
                 dropout=0.2, pooling="attention"):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c), padding_idx=0) for c in cat_cards])
        d_in = n_numeric + n_time + sum(emb_dim(c) for c in cat_cards)
        self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.start = nn.Parameter(torch.zeros(1, 1, hidden))        # learned start / no-history token
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.pooling = pooling
        if pooling == "attention":
            self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.out_dim = hidden

    def forward(self, num, cat, time, length):
        b, n, _ = num.shape
        x = torch.cat([num, time] + [e(cat[..., j]) for j, e in enumerate(self.embs)], dim=-1)
        x = torch.cat([self.start.expand(b, 1, -1), self.proj(x)], dim=1)      # [b, n + 1, hidden]
        lengths = (length + 1).cpu()                                          # + start token
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=n + 1)
        mask = torch.arange(n + 1, device=num.device)[None, :] < (length + 1)[:, None]
        if self.pooling == "last":
            return h[-1]
        if self.pooling == "mean":
            return (out * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        score = self.att(out).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1)
        return (out * w[..., None]).sum(1)


class TwoBranchNet(nn.Module):
    def __init__(self, static_numeric: int, static_cards: list[int], seq_numeric: int, seq_time: int,
                 seq_cards: list[int], hidden=64, gru_layers=2, dropout=0.2, pooling="attention",
                 use_static=True, use_sequence=True):
        super().__init__()
        assert use_static or use_sequence
        self.use_static, self.use_sequence = use_static, use_sequence
        d = 0
        if use_static:
            self.static = StaticBranch(static_numeric, static_cards, dropout=dropout)
            d += self.static.out_dim
        if use_sequence:
            self.sequence = SequenceBranch(seq_numeric, seq_time, seq_cards, hidden, gru_layers, dropout, pooling)
            d += self.sequence.out_dim
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))     # no sigmoid: BCEWithLogitsLoss

    def forward(self, batch):
        parts = []
        if self.use_static:
            parts.append(self.static(batch["static_num"], batch["static_cat"]))
        if self.use_sequence:
            parts.append(self.sequence(batch["seq_num"], batch["seq_cat"], batch["seq_time"], batch["seq_len"]))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)
