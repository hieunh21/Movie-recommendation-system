from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf


class NeuMFInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class NeuMFArtifacts:
    user2idx: Dict[int, int]
    idx2user: Dict[int, int]
    item2idx: Dict[int, int]
    idx2item: Dict[int, int]
    candidates: List[int]  # All items to score
    popular_movies: List[int]  # Fallback for cold-start


class NeuMFRecommender:
    def __init__(
        self,
        model_path: str,
        user_mapping_path: str,
        item_mapping_path: str,
        candidates_path: str,
        popular_movies_path: str,
    ) -> None:
        """
        Initialize NeuMF recommender.

        Args:
            model_path: Path to NeuMF.keras model file
            user_mapping_path: Path to user_mapping_neumf.json
            item_mapping_path: Path to item_mapping_neumf.json
            candidates_path: Path to neumf_candidates.csv (list of items to score)
            popular_movies_path: Path to popular_movies.csv (for cold-start fallback)
        """
        self.model = tf.keras.models.load_model(model_path)
        self.artifacts = self._load_artifacts(user_mapping_path, item_mapping_path, candidates_path, popular_movies_path)

    def _load_artifacts(
        self,
        user_mapping_path: str,
        item_mapping_path: str,
        candidates_path: str,
        popular_movies_path: str,
    ) -> NeuMFArtifacts:
        """Load all mapping and candidate files."""
        # Load user mapping
        with open(user_mapping_path, mode="r", encoding="utf-8") as f:
            user_payload = json.load(f)
        user2idx_raw = user_payload.get("user2idx", {})
        idx2user_raw = user_payload.get("idx2user", {})

        user2idx: Dict[int, int] = {int(k): int(v) for k, v in user2idx_raw.items()}
        idx2user: Dict[int, int] = {int(k): int(v) for k, v in idx2user_raw.items()}

        # Load item mapping (keys are strings in JSON, convert to int for consistency)
        with open(item_mapping_path, mode="r", encoding="utf-8") as f:
            item_payload = json.load(f)
        item2idx_raw = item_payload.get("item2idx", {})
        idx2item_raw = item_payload.get("idx2item", {})

        # Normalize: all keys should be integers for consistency
        item2idx: Dict[int, int] = {}
        for k, v in item2idx_raw.items():
            try:
                item2idx[int(k)] = int(v)
            except (ValueError, TypeError):
                continue
        
        idx2item: Dict[int, int] = {}
        for k, v in idx2item_raw.items():
            try:
                idx2item[int(k)] = int(v)
            except (ValueError, TypeError):
                continue

        # Load candidates and popular movies. Files may include UTF-8 BOM in headers.
        candidates = self._load_movie_ids_from_csv(candidates_path)
        popular_movies = self._load_movie_ids_from_csv(popular_movies_path)

        return NeuMFArtifacts(
            user2idx=user2idx,
            idx2user=idx2user,
            item2idx=item2idx,
            idx2item=idx2item,
            candidates=candidates,
            popular_movies=popular_movies,
        )

    @staticmethod
    def _load_movie_ids_from_csv(csv_path: str) -> List[int]:
        movie_ids: List[int] = []
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id_raw = (
                    row.get("movieId")
                    or row.get("movie_id")
                    or row.get("itemId")
                    or row.get("item_id")
                    or "0"
                )
                try:
                    movie_id = int(movie_id_raw)
                except (TypeError, ValueError):
                    continue
                if movie_id > 0:
                    movie_ids.append(movie_id)
        return movie_ids

    @staticmethod
    def _resolve_item_index(item2idx: Dict[int, int], movie_id: int) -> int | None:
        """Handle both int and string-like keys safely (useful with stale cached artifacts)."""
        if movie_id in item2idx:
            return int(item2idx[movie_id])
        as_str = str(movie_id)
        if as_str in item2idx:  # type: ignore[operator]
            return int(item2idx[as_str])  # type: ignore[index]
        return None

    def score_all(self, user_id: int | str) -> Dict[int, float]:
        user_id = int(user_id)

        if user_id not in self.artifacts.user2idx:
            # Cold-start: assign descending scores from popular list
            scores: Dict[int, float] = {}
            total = max(len(self.artifacts.popular_movies), 1)
            for idx, movie_id in enumerate(self.artifacts.popular_movies):
                scores[int(movie_id)] = float(total - idx)
            return scores

        user_idx = self.artifacts.user2idx[user_id]

        filtered_candidates: List[int] = []
        resolved_item_indices: List[int] = []
        for movie_id in self.artifacts.candidates:
            item_idx = self._resolve_item_index(self.artifacts.item2idx, int(movie_id))
            if item_idx is None:
                continue
            filtered_candidates.append(int(movie_id))
            resolved_item_indices.append(item_idx)

        if not filtered_candidates:
            return {}

        user_indices = np.array([user_idx] * len(filtered_candidates), dtype=np.int32)
        item_indices = np.array(resolved_item_indices, dtype=np.int32)

        try:
            scores = self.model.predict([user_indices, item_indices], verbose=0).flatten()
        except Exception as e:
            raise NeuMFInferenceError(f"Model inference failed: {e}")

        score_map: Dict[int, float] = {}
        for idx, movie_id in enumerate(filtered_candidates):
            score_map[int(movie_id)] = float(scores[idx])
        return score_map

    def recommend(self, user_id: int | str, top_k: int = 10) -> List[int]:
        """
        Generate Top-K recommendations for a user.

        Args:
            user_id: User ID (will be looked up in user2idx)
            top_k: Number of recommendations to return

        Returns:
            List of movieIds (up to top_k items)

        Raises:
            NeuMFInferenceError: If inference fails or user not in training set
        """
        score_map = self.score_all(user_id)
        if not score_map:
            return self.artifacts.popular_movies[:top_k]
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in ranked[:top_k]]
