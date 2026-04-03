from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List


@dataclass(frozen=True)
class MovieCatalogItem:
    movie_id: int
    title: str
    genres: List[str]
    year: int | None


class MovieCatalog:
    def __init__(self, items_by_id: Dict[int, MovieCatalogItem]) -> None:
        self._items_by_id = items_by_id

    @classmethod
    def from_movies_csv(cls, csv_path: str | Path) -> "MovieCatalog":
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"movies.csv not found: {csv_path}")

        items_by_id: Dict[int, MovieCatalogItem] = {}

        with path.open("r", encoding="utf-8", errors="replace") as file_obj:
            next(file_obj, None)
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]

                parts = line.split(";")
                while parts and parts[-1] == "":
                    parts.pop()

                if len(parts) < 2:
                    continue

                try:
                    movie_id = int(parts[0])
                except ValueError:
                    continue

                if len(parts) == 2:
                    title = parts[1].strip()
                    genres_raw = ""
                else:
                    title = ";".join(parts[1:-1]).strip()
                    genres_raw = parts[-1].strip()

                genres = [genre for genre in genres_raw.split("|") if genre]
                items_by_id[movie_id] = MovieCatalogItem(
                    movie_id=movie_id,
                    title=title,
                    genres=genres,
                    year=_extract_year(title),
                )

        return cls(items_by_id)

    def get(self, movie_id: int) -> MovieCatalogItem | None:
        return self._items_by_id.get(movie_id)

    def search(self, query: str, allowed_ids: set[int] | None = None, limit: int = 20) -> List[MovieCatalogItem]:
        normalized = query.strip().lower()
        if not normalized:
            return []

        matched: List[MovieCatalogItem] = []
        for movie_id, item in self._items_by_id.items():
            if allowed_ids is not None and movie_id not in allowed_ids:
                continue
            if normalized in item.title.lower():
                matched.append(item)

        matched.sort(key=lambda item: item.title)
        return matched[:limit]

    def count(self, allowed_ids: set[int] | None = None) -> int:
        if allowed_ids is None:
            return len(self._items_by_id)
        return sum(1 for movie_id in self._items_by_id if movie_id in allowed_ids)

    def trending_local(self, allowed_ids: set[int] | None = None, limit: int = 12) -> List[MovieCatalogItem]:
        candidates: List[MovieCatalogItem] = []
        for movie_id, item in self._items_by_id.items():
            if allowed_ids is not None and movie_id not in allowed_ids:
                continue
            candidates.append(item)

        # Local trending heuristic: newer movies first, then title.
        candidates.sort(key=lambda item: ((item.year or 0), item.title.lower()), reverse=True)
        return candidates[:limit]


def _extract_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", title)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
