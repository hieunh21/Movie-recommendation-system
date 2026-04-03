from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class MovieIdMapper:
    movie_to_tmdb: Dict[int, int]
    tmdb_to_movie: Dict[int, int]

    @classmethod
    def from_link_csv(cls, csv_path: str | Path) -> "MovieIdMapper":
        movie_to_tmdb: Dict[int, int] = {}
        tmdb_to_movie: Dict[int, int] = {}

        with open(csv_path, mode="r", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                movie_id_raw = row.get("movieId")
                tmdb_id_raw = row.get("tmdbId")
                if not movie_id_raw or not tmdb_id_raw:
                    continue
                try:
                    movie_id = int(movie_id_raw)
                    tmdb_id = int(tmdb_id_raw)
                except ValueError:
                    continue

                movie_to_tmdb[movie_id] = tmdb_id
                tmdb_to_movie[tmdb_id] = movie_id

        return cls(movie_to_tmdb=movie_to_tmdb, tmdb_to_movie=tmdb_to_movie)

    def to_tmdb(self, movie_id: int) -> Optional[int]:
        return self.movie_to_tmdb.get(movie_id)

    def to_movie(self, tmdb_id: int) -> Optional[int]:
        return self.tmdb_to_movie.get(tmdb_id)
