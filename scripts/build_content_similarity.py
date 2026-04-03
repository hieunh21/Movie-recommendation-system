from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_title(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\(\d{4}\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    movies_path = base_path / "movies.csv"
    ratings_path = base_path / "ratings.csv"
    users_path = base_path / "users.csv"

    if not movies_path.exists():
        raise FileNotFoundError(f"movies.csv not found at {movies_path}")

    movies = pd.read_csv(movies_path, sep=";", encoding="latin-1")
    ratings = pd.read_csv(ratings_path, sep=";", encoding="latin-1") if ratings_path.exists() else None
    users = pd.read_csv(users_path, sep=";", encoding="latin-1") if users_path.exists() else None

    movies = movies.loc[:, ~movies.columns.str.contains(r"^Unnamed")]
    if ratings is not None:
        ratings = ratings.loc[:, ~ratings.columns.str.contains(r"^Unnamed")]
    if users is not None:
        users = users.loc[:, ~users.columns.str.contains(r"^Unnamed")]

    print(f"Movies shape : {movies.shape}")
    if ratings is not None:
        print(f"Ratings shape: {ratings.shape}")
    else:
        print("Ratings shape: not loaded")
    if users is not None:
        print(f"Users shape  : {users.shape}")
    else:
        print("Users shape  : not loaded")

    print(f"\nMovies columns : {movies.columns.tolist()}")
    if ratings is not None:
        print(f"Ratings columns: {ratings.columns.tolist()}")
    if users is not None:
        print(f"Users columns  : {users.columns.tolist()}")

    return movies, ratings, users


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    required = {"movieId", "title", "genres"}
    missing = required - set(movies.columns)
    if missing:
        raise RuntimeError(f"movies.csv missing required columns: {sorted(missing)}")

    movies = movies[["movieId", "title", "genres"]].copy()
    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    movies["title_clean"] = movies["title"].apply(clean_title)
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False).str.lower()
    movies["content"] = (movies["title_clean"] + " " + movies["genres_clean"]).str.strip()
    movies = movies[movies["content"] != ""].reset_index(drop=True)

    print(f"So phim sau tien xu ly: {len(movies)}")
    return movies


def build_similarity(movies: pd.DataFrame) -> tuple[TfidfVectorizer, object, np.ndarray]:
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = tfidf.fit_transform(movies["content"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
    return tfidf, tfidf_matrix, cosine_sim


def build_topk_from_cosine(movies: pd.DataFrame, cosine_sim: np.ndarray, top_k: int) -> Dict[int, list[dict]]:
    movie_ids = movies["movieId"].astype(int).tolist()
    topk_similar: Dict[int, list[dict]] = {}
    k = max(1, int(top_k))

    for idx, movie_id in enumerate(movie_ids):
        scores = cosine_sim[idx].copy()
        scores[idx] = -1.0

        candidate_count = min(k, len(scores) - 1) if len(scores) > 1 else 0
        if candidate_count <= 0:
            topk_similar[movie_id] = []
            continue

        top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        items: list[dict] = []
        for sim_idx in top_indices:
            score = float(scores[sim_idx])
            if score <= 0:
                continue
            items.append({"movie_id": int(movie_ids[sim_idx]), "score": round(score, 6)})
        topk_similar[movie_id] = items

    return topk_similar


def write_artifacts(
    movies: pd.DataFrame,
    tfidf: TfidfVectorizer,
    tfidf_matrix: object,
    topk_similar: Dict[int, list[dict]],
    movies_clean_out: Path,
    topk_out: Path,
    vectorizer_out: Path,
    tfidf_matrix_out: Path,
) -> None:
    movies.to_csv(movies_clean_out, index=False, encoding="utf-8")

    with topk_out.open("wb") as f:
        pickle.dump(
            {
                "vectorizer_config": {
                    "stop_words": "english",
                    "ngram_range": (1, 2),
                    "min_df": 1,
                },
                "topk_similar": topk_similar,
            },
            f,
        )

    with vectorizer_out.open("wb") as f:
        pickle.dump(tfidf, f)

    save_npz(tfidf_matrix_out, tfidf_matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build content-similarity artifacts for movie recommendations.")
    parser.add_argument("--base-path", default="model")
    parser.add_argument("--movies-clean-out", default="model/movies_clean.csv")
    parser.add_argument("--topk-out", default="model/topk_similar.pkl")
    parser.add_argument("--vectorizer-out", default="model/tfidf_vectorizer.pkl")
    parser.add_argument("--tfidf-matrix-out", default="model/tfidf_matrix.npz")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    base_path = Path(args.base_path)
    movies_clean_out = Path(args.movies_clean_out)
    topk_out = Path(args.topk_out)
    vectorizer_out = Path(args.vectorizer_out)
    tfidf_matrix_out = Path(args.tfidf_matrix_out)

    movies_raw, _, _ = load_dataset(base_path)
    movies = preprocess_movies(movies_raw)
    if movies.empty:
        raise RuntimeError("No movies loaded after preprocessing.")

    tfidf, tfidf_matrix, cosine_sim = build_similarity(movies)
    topk_similar = build_topk_from_cosine(movies, cosine_sim, top_k=args.top_k)
    write_artifacts(
        movies=movies,
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        topk_similar=topk_similar,
        movies_clean_out=movies_clean_out,
        topk_out=topk_out,
        vectorizer_out=vectorizer_out,
        tfidf_matrix_out=tfidf_matrix_out,
    )

    print(f"Built movies_clean: {movies_clean_out}")
    print(f"Built topk_similar artifact: {topk_out}")
    print(f"Built tfidf vectorizer: {vectorizer_out}")
    print(f"Built tfidf matrix: {tfidf_matrix_out}")
    print(f"Movies processed: {len(movies)}")


if __name__ == "__main__":
    main()
