from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn


class LRURecInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LRURecArtifacts:
    item2idx: Dict[int, int]
    idx2item: Dict[int, int]
    pad_token_id: int
    mask_token_id: int
    max_len: int


class LRUCell(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.nu_log = nn.Parameter(torch.zeros(hidden_size))
        self.theta = nn.Parameter(torch.zeros(hidden_size))
        self.D = nn.Parameter(torch.zeros(hidden_size))
        self.W_in = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.W_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, candidate = self.W_in(hidden).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        candidate = torch.tanh(candidate)

        # A compact recurrent approximation matching the checkpoint parameterization.
        state = torch.zeros_like(candidate[:, 0, :])
        outputs = []
        decay = torch.sigmoid(-torch.exp(self.nu_log)).view(1, -1)
        for t in range(candidate.size(1)):
            state = decay * state + gate[:, t, :] * candidate[:, t, :]
            outputs.append(state)
        recurrent = torch.stack(outputs, dim=1)
        mixed = self.W_out(recurrent) + hidden * self.D.view(1, 1, -1)
        return self.norm(mixed + hidden)


class LRUBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lru = LRUCell(hidden_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.lru(hidden)
        return hidden + self.ffn(hidden)


class LRURecModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([LRUBlock(hidden_size, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_size)
        self.register_parameter("out_bias", nn.Parameter(torch.zeros(vocab_size)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.emb_dropout(self.item_embedding(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.out_norm(hidden)
        return torch.matmul(hidden, self.item_embedding.weight.transpose(0, 1)) + self.out_bias


class LRURecRecommender:
    def __init__(self, checkpoint_path: str, mapping_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.artifacts = self._load_mapping(mapping_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if "item_embedding.weight" not in state_dict:
            raise LRURecInferenceError("LRURec checkpoint does not contain item_embedding.weight.")

        hidden_size = int(state_dict["item_embedding.weight"].shape[1])
        vocab_size = int(state_dict["item_embedding.weight"].shape[0])
        num_layers = 1 + max(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("blocks.") and key.endswith(".lru.nu_log")
        )
        config = checkpoint.get("config", {})

        model = LRURecModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(config.get("dropout", 0.0)),
            pad_token_id=int(config.get("pad_token_id", self.artifacts.pad_token_id)),
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model.to(self.device)
        self.max_len = int(config.get("max_len", self.artifacts.max_len))
        self.mask_token_id = int(checkpoint.get("mask_token_id", self.artifacts.mask_token_id))

    def _load_mapping(self, mapping_path: str) -> LRURecArtifacts:
        with open(mapping_path, mode="r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        item2idx = {int(k): int(v) for k, v in payload.get("item2idx", {}).items()}
        idx2item = {int(k): int(v) for k, v in payload.get("idx2item", {}).items()}
        return LRURecArtifacts(
            item2idx=item2idx,
            idx2item=idx2item,
            pad_token_id=0,
            mask_token_id=int(payload.get("mask_token_id", 0)),
            max_len=200,
        )

    def _build_input_sequence(self, movie_ids: Sequence[int]) -> tuple[torch.Tensor, List[int]]:
        seq_indices = [
            self.artifacts.item2idx[int(movie_id)]
            for movie_id in movie_ids
            if int(movie_id) in self.artifacts.item2idx
        ]
        if not seq_indices:
            raise LRURecInferenceError("No selected movies are present in LRURec item mapping.")

        seq_indices = seq_indices[-(self.max_len - 1) :]
        model_seq = seq_indices + [self.mask_token_id]
        if len(model_seq) < self.max_len:
            model_seq = [self.artifacts.pad_token_id] * (self.max_len - len(model_seq)) + model_seq

        input_tensor = torch.tensor(model_seq, dtype=torch.long, device=self.device).unsqueeze(0)
        return input_tensor, seq_indices

    @torch.inference_mode()
    def score_all(self, movie_ids: Sequence[int]) -> Dict[int, float]:
        input_tensor, seen_indices = self._build_input_sequence(movie_ids)
        logits = self.model(input_tensor)
        next_item_logits = logits[0, -1, :].clone()

        blocked_indices = set(seen_indices)
        blocked_indices.add(self.artifacts.pad_token_id)
        blocked_indices.add(self.mask_token_id)
        for idx in blocked_indices:
            if 0 <= idx < next_item_logits.size(0):
                next_item_logits[idx] = float("-inf")

        score_map: Dict[int, float] = {}
        for idx, score in enumerate(next_item_logits.tolist()):
            if score == float("-inf"):
                continue
            movie_id = self.artifacts.idx2item.get(int(idx))
            if movie_id is not None:
                score_map[int(movie_id)] = float(score)
        return score_map

    @torch.inference_mode()
    def recommend(self, movie_ids: Sequence[int], top_k: int = 10) -> List[int]:
        score_map = self.score_all(movie_ids)
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in ranked[:top_k]]
