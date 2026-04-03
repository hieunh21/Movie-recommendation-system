from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, List, Sequence

import requests


class TMDBClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class MovieSummary:
    movie_id: int
    title: str
    year: str
    poster_url: str
    vote_average: float
    genres: Sequence[str]
    overview: str


class TMDBClient:
    def __init__(self, api_key: str, timeout_seconds: int = 10) -> None:
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://api.themoviedb.org/3"
        self.poster_base_url = "https://image.tmdb.org/t/p/w342"
        self._detail_cache: Dict[int, MovieSummary] = {}

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise TMDBClientError("TMDB_API_KEY is missing.")

        endpoint = f"{self.base_url}{path}"
        merged_params = {
            "api_key": self.api_key,
            "language": "en-US",
            **params,
        }

        last_exception: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.get(endpoint, params=merged_params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exception = exc
                if attempt < 2:
                    sleep(0.25 * (attempt + 1))

        raise TMDBClientError(f"TMDB request failed for {path}: {last_exception}")

    def _build_poster_url(self, poster_path: str | None) -> str:
        if not poster_path:
            return ""
        return f"{self.poster_base_url}{poster_path}"

    def _build_movie_summary(self, payload: Dict[str, Any]) -> MovieSummary:
        release_date = str(payload.get("release_date") or "")
        year = release_date[:4] if len(release_date) >= 4 else "N/A"

        genre_names: List[str] = []
        for genre in payload.get("genres", []):
            name = genre.get("name")
            if name:
                genre_names.append(str(name))

        return MovieSummary(
            movie_id=int(payload.get("id", 0)),
            title=str(payload.get("title") or payload.get("name") or "Unknown"),
            year=year,
            poster_url=self._build_poster_url(payload.get("poster_path")),
            vote_average=float(payload.get("vote_average") or 0.0),
            genres=genre_names,
            overview=str(payload.get("overview") or "").strip(),
        )

    def search_movies(self, query: str, limit: int = 8) -> List[MovieSummary]:
        query = query.strip()
        if not query:
            return []

        payload = self._request("/search/movie", {"query": query, "page": 1, "include_adult": False})
        results = payload.get("results", [])

        movies: List[MovieSummary] = []
        for result in results[:limit]:
            movie = MovieSummary(
                movie_id=int(result.get("id", 0)),
                title=str(result.get("title") or result.get("name") or "Unknown"),
                year=(str(result.get("release_date") or "")[:4] or "N/A"),
                poster_url=self._build_poster_url(result.get("poster_path")),
                vote_average=float(result.get("vote_average") or 0.0),
                genres=[],
                overview=str(result.get("overview") or "").strip(),
            )
            movies.append(movie)

        return movies

    def get_trending_movies(self, limit: int = 12) -> List[MovieSummary]:
        payload = self._request("/trending/movie/week", {"page": 1})
        results = payload.get("results", [])

        movies: List[MovieSummary] = []
        for result in results[:limit]:
            movie = MovieSummary(
                movie_id=int(result.get("id", 0)),
                title=str(result.get("title") or result.get("name") or "Unknown"),
                year=(str(result.get("release_date") or "")[:4] or "N/A"),
                poster_url=self._build_poster_url(result.get("poster_path")),
                vote_average=float(result.get("vote_average") or 0.0),
                genres=[],
                overview=str(result.get("overview") or "").strip(),
            )
            movies.append(movie)

        return movies

    def get_movie_details(self, movie_id: int) -> MovieSummary:
        if movie_id in self._detail_cache:
            return self._detail_cache[movie_id]

        payload = self._request(f"/movie/{movie_id}", {})
        movie = self._build_movie_summary(payload)
        self._detail_cache[movie_id] = movie
        return movie

    def hydrate_movies(self, movie_ids: Sequence[int]) -> List[MovieSummary]:
        hydrated: List[MovieSummary] = []
        for movie_id in movie_ids:
            try:
                hydrated.append(self.get_movie_details(int(movie_id)))
            except Exception:
                hydrated.append(
                    MovieSummary(
                        movie_id=int(movie_id),
                        title=f"Movie #{movie_id}",
                        year="N/A",
                        poster_url="",
                        vote_average=0.0,
                        genres=[],
                        overview="",
                    )
                )
        return hydrated
