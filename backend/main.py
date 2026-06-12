from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow importing from project root (src/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from src.recommenders.bert4rec import BERT4RecInferenceError, BERT4RecRecommender
from src.recommenders.hybrid import HybridInferenceError, HybridRecommender
from src.recommenders.lightgcn import LightGCNInferenceError, LightGCNRecommender
from src.recommenders.lrurec import LRURecInferenceError, LRURecRecommender
from src.recommenders.neumf import NeuMFInferenceError, NeuMFRecommender
from src.services.content_similarity import ContentSimilarityError, ContentSimilarityService
from src.services.id_mapper import MovieIdMapper
from src.services.movies_catalog import MovieCatalog
from src.services.tmdb import TMDBClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = Path("model")
NEW_MODEL_DIR = Path("new_model")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TOP_K = int(os.getenv("TOP_K", "10"))
MIN_CLICKS = int(os.getenv("MIN_CLICKS_FOR_COLD_START", "3"))
TMDB_TIMEOUT = int(os.getenv("TMDB_TIMEOUT_SECONDS", "10"))

SESSION_MODELS = {
    "bert4rec": "BERT4Rec",
    "bert4rec_improved": "BERT4Rec Improved",
    "lrurec": "LRURec",
}
HISTORY_MODELS = {
    "neumf": "NeuMF",
    "lightgcn": "LightGCN",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Movie Recommendation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons (loaded once on first call)
# ---------------------------------------------------------------------------
def _find(candidates: list[str], base_dir: Path = MODEL_DIR) -> str:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Missing model file. Checked in {base_dir}: {candidates}")


def _validate_option(value: str, allowed: dict[str, str], default: str) -> str:
    value = (value or default).strip()
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown model option: {value}")
    return value


@lru_cache(maxsize=1)
def _tmdb() -> TMDBClient | None:
    return TMDBClient(api_key=TMDB_API_KEY, timeout_seconds=TMDB_TIMEOUT) if TMDB_API_KEY else None


@lru_cache(maxsize=1)
def _mapper() -> MovieIdMapper:
    return MovieIdMapper.from_link_csv(_find(["link.csv"]))


@lru_cache(maxsize=1)
def _catalog() -> MovieCatalog:
    return MovieCatalog.from_movies_csv(_find(["movies.csv"]))


@lru_cache(maxsize=1)
def _bert4rec() -> BERT4RecRecommender:
    return BERT4RecRecommender(
        checkpoint_path=_find(["best_bert4rec_ml1m (1).pt", "best_bert4rec_ml1m.pt"]),
        mapping_path=_find(["item_mapping_bert4rec_ml1m (1).json", "item_mapping_bert4rec_ml1m.json"]),
        device="cpu",
    )


@lru_cache(maxsize=1)
def _bert4rec_improved() -> BERT4RecRecommender:
    return BERT4RecRecommender(
        checkpoint_path=_find(["best_bert4rec_ml1m_improved.pt"], NEW_MODEL_DIR),
        mapping_path=_find(["item_mapping_bert4rec_ml1m_improved.json"], NEW_MODEL_DIR),
        device="cpu",
    )


@lru_cache(maxsize=1)
def _lrurec() -> LRURecRecommender:
    return LRURecRecommender(
        checkpoint_path=_find(["best_lrurec_ml1m.pt"], NEW_MODEL_DIR),
        mapping_path=_find(["item_mapping_lrurec_ml1m.json"], NEW_MODEL_DIR),
        device="cpu",
    )


@lru_cache(maxsize=1)
def _neumf() -> NeuMFRecommender:
    return NeuMFRecommender(
        model_path=_find(["NeuMF.keras"]),
        user_mapping_path=_find(["user_mapping_neumf.json"]),
        item_mapping_path=_find(["item_mapping_neumf.json"]),
        candidates_path=_find(["neumf_candidates.csv"]),
        popular_movies_path=_find(["popular_movies.csv"]),
    )


@lru_cache(maxsize=1)
def _lightgcn() -> LightGCNRecommender:
    return LightGCNRecommender(
        model_path=_find(["LightGCN.h5"], NEW_MODEL_DIR),
        user_mapping_path=_find(["user_mapping_neumf.json"]),
        item_mapping_path=_find(["item_mapping_lrurec_ml1m.json"], NEW_MODEL_DIR),
        candidates_path=_find(["neumf_candidates.csv"]),
        popular_movies_path=_find(["popular_movies.csv"]),
    )


@lru_cache(maxsize=1)
def _hybrid() -> HybridRecommender:
    return HybridRecommender(session_recommender=_bert4rec(), history_recommender=_neumf())


@lru_cache(maxsize=None)
def _hybrid_for(session_model: str, history_model: str) -> HybridRecommender:
    return HybridRecommender(
        session_recommender=_session_recommender(session_model),
        history_recommender=_history_recommender(history_model),
    )


def _session_recommender(model_name: str):
    model_name = _validate_option(model_name, SESSION_MODELS, "bert4rec")
    if model_name == "bert4rec":
        return _bert4rec()
    if model_name == "bert4rec_improved":
        return _bert4rec_improved()
    if model_name == "lrurec":
        return _lrurec()
    raise HTTPException(status_code=400, detail=f"Unknown session model: {model_name}")


def _history_recommender(model_name: str):
    model_name = _validate_option(model_name, HISTORY_MODELS, "neumf")
    if model_name == "neumf":
        return _neumf()
    if model_name == "lightgcn":
        return _lightgcn()
    raise HTTPException(status_code=400, detail=f"Unknown history model: {model_name}")


@lru_cache(maxsize=1)
def _similarity() -> ContentSimilarityService:
    return ContentSimilarityService(_find(["topk_similar.pkl"]))


# ---------------------------------------------------------------------------
# Helper: MovieLens IDs → serializable list of dicts
# ---------------------------------------------------------------------------
def _fetch_tmdb(tmdb: TMDBClient, tmdb_id: int) -> dict | None:
    """Fetch one movie's TMDB details. Returns a partial dict to merge, or None on failure."""
    try:
        s = tmdb.get_movie_details(tmdb_id)
        return {
            "title": s.title,
            "year": s.year,
            "genres": list(s.genres),
            "poster_url": s.poster_url,
            "vote_average": s.vote_average,
            "overview": s.overview,
        }
    except Exception:
        return None


def _to_summaries(movie_ids: list[int]) -> list[dict]:
    catalog = _catalog()
    mapper = _mapper()
    tmdb = _tmdb()

    # Build base data from local CSV (instant)
    results: list[dict] = []
    tmdb_jobs: list[tuple[int, int]] = []  # (tmdb_id, result_index)

    for mid in movie_ids:
        item = catalog.get(mid)
        results.append({
            "movie_id": mid,
            "title": item.title if item else f"Movie #{mid}",
            "year": str(item.year) if (item and item.year) else "N/A",
            "genres": item.genres if item else [],
            "poster_url": "",
            "vote_average": 0.0,
            "overview": "",
        })
        if tmdb:
            tmdb_id = mapper.to_tmdb(mid)
            if tmdb_id:
                tmdb_jobs.append((tmdb_id, len(results) - 1))

    # Fetch all TMDB details in parallel (was sequential → N × latency, now ~1 × latency)
    if tmdb_jobs:
        with ThreadPoolExecutor(max_workers=min(len(tmdb_jobs), 12)) as pool:
            future_to_idx = {
                pool.submit(_fetch_tmdb, tmdb, tmdb_id): idx
                for tmdb_id, idx in tmdb_jobs
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                detail = future.result()
                if detail:
                    results[idx].update(detail)

    return results



# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class NewUserRequest(BaseModel):
    movie_ids: List[int]
    model: str = "bert4rec"


class ExistingUserRequest(BaseModel):
    user_id: int
    click_sequence: List[int] = []
    session_model: str = "bert4rec"
    history_model: str = "neumf"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/config")
def get_config():
    return {"top_k": TOP_K, "min_clicks_for_cold_start": MIN_CLICKS}


@app.get("/recommend/options")
def recommend_options():
    return {
        "session_models": [{"value": value, "label": label} for value, label in SESSION_MODELS.items()],
        "history_models": [{"value": value, "label": label} for value, label in HISTORY_MODELS.items()],
        "defaults": {"session_model": "bert4rec", "history_model": "neumf"},
    }


@app.get("/movies/search")
def search_movies(q: str = Query(default=""), limit: int = Query(default=20)):
    if not q.strip():
        return []
    catalog = _catalog()
    eligible = set(_bert4rec().artifacts.item2idx.keys())
    items = catalog.search(q, allowed_ids=eligible, limit=limit)
    return _to_summaries([i.movie_id for i in items])


@app.get("/movies/trending")
def trending_movies(limit: int = Query(default=12)):
    catalog = _catalog()
    eligible = set(_bert4rec().artifacts.item2idx.keys())
    items = catalog.trending_local(allowed_ids=eligible, limit=limit)
    return _to_summaries([i.movie_id for i in items])


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    summaries = _to_summaries([movie_id])
    if not summaries:
        raise HTTPException(status_code=404, detail="Not found")
    return summaries[0]


@app.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: int, top_k: int = Query(default=10)):
    try:
        similar = _similarity().get_similar(movie_id, top_k=top_k)
    except (FileNotFoundError, ContentSimilarityError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    scores = {s.movie_id: s.score for s in similar}
    summaries = _to_summaries([s.movie_id for s in similar])
    for s in summaries:
        s["similarity_score"] = round(scores.get(s["movie_id"], 0.0), 4)
    return summaries


@app.post("/recommend/new-user")
def recommend_new_user(req: NewUserRequest):
    if len(req.movie_ids) < MIN_CLICKS:
        return []
    try:
        model_name = _validate_option(req.model, SESSION_MODELS, "bert4rec")
        ids = _session_recommender(model_name).recommend(req.movie_ids, top_k=TOP_K)
        return _to_summaries(ids)
    except (BERT4RecInferenceError, LRURecInferenceError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/recommend/existing-user")
def recommend_existing_user(req: ExistingUserRequest):
    try:
        session_model = _validate_option(req.session_model, SESSION_MODELS, "bert4rec")
        history_model = _validate_option(req.history_model, HISTORY_MODELS, "neumf")
        if req.click_sequence:
            ids = _hybrid_for(session_model, history_model).recommend(req.user_id, req.click_sequence, top_k=TOP_K)
        else:
            ids = _history_recommender(history_model).recommend(req.user_id, top_k=TOP_K)
        if not ids:
            ids = _neumf().artifacts.popular_movies[:TOP_K]
        return _to_summaries(ids)
    except (HybridInferenceError, NeuMFInferenceError, LightGCNInferenceError, LRURecInferenceError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/users/sample")
def sample_users(limit: int = Query(default=200)):
    all_ids = sorted(_neumf().artifacts.user2idx.keys())
    return all_ids[:limit]
