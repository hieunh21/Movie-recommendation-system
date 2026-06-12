from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


class LightGCNInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LightGCNArtifacts:
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    candidates: List[int]
    popular_movies: List[int]


class LightGCNRecommender:
    def __init__(
        self,
        model_path: str,
        user_mapping_path: str,
        item_mapping_path: str,
        candidates_path: str,
        popular_movies_path: str,
    ) -> None:
        self.artifacts = self._load_artifacts(
            user_mapping_path=user_mapping_path,
            item_mapping_path=item_mapping_path,
            candidates_path=candidates_path,
            popular_movies_path=popular_movies_path,
        )
        self.embeddings = self._load_embeddings(model_path)

        expected_rows = len(self.artifacts.user2idx) + len(self.artifacts.item2idx)
        if self.embeddings.shape[0] < expected_rows:
            raise LightGCNInferenceError(
                f"LightGCN embedding has {self.embeddings.shape[0]} rows, expected at least {expected_rows}."
            )

    @staticmethod
    def _load_embeddings(model_path: str) -> np.ndarray:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"LightGCN model not found: {path}")

        with h5py.File(path, "r") as h5:
            if "vars" not in h5 or "0" not in h5["vars"]:
                raise LightGCNInferenceError("LightGCN.h5 does not contain vars/0 embeddings.")
            embeddings = np.asarray(h5["vars"]["0"], dtype=np.float32)

        if embeddings.ndim != 2:
            raise LightGCNInferenceError(f"Expected 2D embedding matrix, got shape {embeddings.shape}.")
        return embeddings

    @staticmethod
    def _load_json_mapping(path: str, key: str) -> Dict[int, int]:
        with open(path, mode="r", encoding="utf-8") as f:
            payload = json.load(f)
        return {int(k): int(v) for k, v in payload.get(key, {}).items()}

    @staticmethod
    def _load_movie_ids_from_csv(csv_path: str) -> List[int]:
        movie_ids: List[int] = []
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("movieId") or row.get("movie_id") or row.get("itemId") or row.get("item_id")
                try:
                    movie_id = int(raw or 0)
                except (TypeError, ValueError):
                    continue
                if movie_id > 0:
                    movie_ids.append(movie_id)
        return movie_ids

    def _load_artifacts(
        self,
        user_mapping_path: str,
        item_mapping_path: str,
        candidates_path: str,
        popular_movies_path: str,
    ) -> LightGCNArtifacts:
        return LightGCNArtifacts(
            user2idx=self._load_json_mapping(user_mapping_path, "user2idx"),
            item2idx=self._load_json_mapping(item_mapping_path, "item2idx"),
            candidates=self._load_movie_ids_from_csv(candidates_path),
            popular_movies=self._load_movie_ids_from_csv(popular_movies_path),
        )

    def score_all(self, user_id: int | str) -> Dict[int, float]:
        user_id = int(user_id)
        if user_id not in self.artifacts.user2idx:
            total = max(len(self.artifacts.popular_movies), 1)
            return {int(movie_id): float(total - idx) for idx, movie_id in enumerate(self.artifacts.popular_movies)}

        user_idx = self.artifacts.user2idx[user_id]
        user_vec = self.embeddings[user_idx]
        item_offset = len(self.artifacts.user2idx)
        min_item_idx = min(self.artifacts.item2idx.values(), default=0)

        scored: Dict[int, float] = {}
        for movie_id in self.artifacts.candidates:
            item_idx = self.artifacts.item2idx.get(int(movie_id))
            if item_idx is None:
                continue
            row_idx = item_offset + item_idx - min_item_idx
            if row_idx >= self.embeddings.shape[0]:
                continue
            scored[int(movie_id)] = float(np.dot(user_vec, self.embeddings[row_idx]))
        return scored

    def recommend(self, user_id: int | str, top_k: int = 10) -> List[int]:
        score_map = self.score_all(user_id)
        if not score_map:
            return self.artifacts.popular_movies[:top_k]
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in ranked[:top_k]]
