from __future__ import annotations

from typing import Dict, List, Sequence

from src.recommenders.bert4rec import BERT4RecInferenceError, BERT4RecRecommender
from src.recommenders.neumf import NeuMFInferenceError, NeuMFRecommender


class HybridInferenceError(RuntimeError):
    pass


class HybridRecommender:
    def __init__(self, session_recommender: BERT4RecRecommender, history_recommender: NeuMFRecommender) -> None:
        self.session_recommender = session_recommender
        self.history_recommender = history_recommender

    @staticmethod
    def _alpha_from_sequence_length(seq_len: int) -> float:
        if seq_len == 0:
            return 0.0
        if seq_len <= 2:
            return 0.3
        if seq_len <= 4:
            return 0.5
        return 0.7

    @staticmethod
    def _normalize(score_map: Dict[int, float]) -> Dict[int, float]:
        if not score_map:
            return {}
        values = list(score_map.values())
        min_v = min(values)
        max_v = max(values)
        if max_v <= min_v:
            return {k: 0.0 for k in score_map}
        denom = max_v - min_v
        return {k: (v - min_v) / denom for k, v in score_map.items()}

    def recommend(
        self,
        user_id: int,
        click_sequence: Sequence[int],
        top_k: int = 10,
    ) -> List[int]:
        seen = {int(mid) for mid in click_sequence}

        try:
            history_raw = self.history_recommender.score_all(user_id)
        except (NeuMFInferenceError, RuntimeError) as exc:
            raise HybridInferenceError(str(exc))

        session_raw: Dict[int, float] = {}
        if click_sequence:
            try:
                session_raw = self.session_recommender.score_all(click_sequence)
            except (BERT4RecInferenceError, RuntimeError):
                session_raw = {}

        alpha = self._alpha_from_sequence_length(len(click_sequence))
        session_norm = self._normalize(session_raw)
        history_norm = self._normalize(history_raw)

        candidates = set(session_norm.keys()) | set(history_norm.keys())
        if not candidates:
            return []

        final_scores: Dict[int, float] = {}
        for movie_id in candidates:
            if movie_id in seen:
                continue
            session_score = session_norm.get(movie_id, 0.0)
            history_score = history_norm.get(movie_id, 0.0)
            final_scores[movie_id] = alpha * session_score + (1.0 - alpha) * history_score

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in ranked[:top_k]]
