from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn


class BERT4RecInferenceError(RuntimeError):
    pass


class BERT4RecModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_len: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_len = max_len

        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(hidden_size)
        # Checkpoint uses a hidden projection, then ties logits to item embeddings.
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.register_parameter("out_bias", nn.Parameter(torch.zeros(vocab_size)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_len:
            raise BERT4RecInferenceError("Input sequence is longer than model max_len.")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.item_embedding(input_ids) + self.position_embedding(positions)

        key_padding_mask = input_ids.eq(self.pad_token_id)
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.layernorm(hidden)

        hidden = self.out_proj(hidden)
        logits = torch.matmul(hidden, self.item_embedding.weight.transpose(0, 1)) + self.out_bias
        return logits


@dataclass(frozen=True)
class BERT4RecArtifacts:
    item2idx: Dict[int, int]
    idx2item: Dict[int, int]
    pad_token_id: int
    mask_token_id: int
    max_len: int


class BERT4RecRecommender:
    def __init__(self, checkpoint_path: str, mapping_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.artifacts = self._load_mapping(mapping_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})

        vocab_size = int(checkpoint.get("vocab_size", 0))
        if vocab_size <= 0:
            raise BERT4RecInferenceError("Invalid vocab_size in checkpoint.")

        model = BERT4RecModel(
            vocab_size=vocab_size,
            hidden_size=int(config.get("hidden_size", 256)),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 2)),
            dropout=float(config.get("dropout", 0.2)),
            max_len=int(config.get("max_len", 200)),
            pad_token_id=int(config.get("pad_token_id", 0)),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        self.model = model.to(self.device)
        self.max_len = int(config.get("max_len", 200))
        self.mask_token_id = int(checkpoint.get("mask_token_id", self.artifacts.mask_token_id))

    def _load_mapping(self, mapping_path: str) -> BERT4RecArtifacts:
        with open(mapping_path, mode="r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        item2idx_raw = payload.get("item2idx", {})
        idx2item_raw = payload.get("idx2item", {})

        item2idx: Dict[int, int] = {}
        idx2item: Dict[int, int] = {}

        for key, value in item2idx_raw.items():
            item2idx[int(key)] = int(value)
        for key, value in idx2item_raw.items():
            idx2item[int(key)] = int(value)

        return BERT4RecArtifacts(
            item2idx=item2idx,
            idx2item=idx2item,
            pad_token_id=0,
            mask_token_id=int(payload.get("mask_token_id", 0)),
            max_len=200,
        )

    def _build_input_sequence(self, movie_ids: Sequence[int]) -> tuple[torch.Tensor, List[int]]:
        seq_indices: List[int] = []
        for movie_id in movie_ids:
            mapped = self.artifacts.item2idx.get(int(movie_id))
            if mapped is not None:
                seq_indices.append(mapped)

        if not seq_indices:
            raise BERT4RecInferenceError("No selected movies are present in BERT4Rec item mapping.")

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
            if movie_id is None:
                continue
            score_map[int(movie_id)] = float(score)
        return score_map

    @torch.inference_mode()
    def recommend(self, movie_ids: Sequence[int], top_k: int = 10) -> List[int]:
        score_map = self.score_all(movie_ids)
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in ranked[:top_k]]
