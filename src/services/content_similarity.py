from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


class ContentSimilarityError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimilarMovie:
    movie_id: int
    score: float


@dataclass(frozen=True)
class ContentSimilarityArtifacts:
    topk_similar: Dict[int, List[SimilarMovie]]


class ContentSimilarityService:
    def __init__(self, artifacts_path: str | Path) -> None:
        path = Path(artifacts_path)
        if not path.exists():
            raise FileNotFoundError(f"Content similarity artifact not found: {path}")

        with path.open("rb") as f:
            payload = pickle.load(f)

        topk_raw = payload.get("topk_similar", {})
        topk_similar: Dict[int, List[SimilarMovie]] = {}
        for key, values in topk_raw.items():
            movie_id = int(key)
            parsed_values: List[SimilarMovie] = []
            for item in values:
                similar_id = int(item.get("movie_id"))
                score = float(item.get("score", 0.0))
                parsed_values.append(SimilarMovie(movie_id=similar_id, score=score))
            topk_similar[movie_id] = parsed_values

        self.artifacts = ContentSimilarityArtifacts(topk_similar=topk_similar)

    def get_similar(self, movie_id: int, top_k: int = 10) -> List[SimilarMovie]:
        if top_k <= 0:
            return []
        return self.artifacts.topk_similar.get(int(movie_id), [])[:top_k]
