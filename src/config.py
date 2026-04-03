from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class AppConfig:
    top_k: int
    min_clicks_for_cold_start: int
    tmdb_timeout_seconds: int
    tmdb_api_key: str


def _get_secret(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
        return str(value)
    except Exception:
        return default


def _get_secret_int(key: str, default: int) -> int:
    raw_value = _get_secret(key, str(default))
    try:
        return int(raw_value)
    except ValueError:
        return default


def load_config() -> AppConfig:
    return AppConfig(
        top_k=_get_secret_int("TOP_K", 10),
        min_clicks_for_cold_start=_get_secret_int("MIN_CLICKS_FOR_COLD_START", 3),
        tmdb_timeout_seconds=_get_secret_int("TMDB_TIMEOUT_SECONDS", 10),
        tmdb_api_key=_get_secret("TMDB_API_KEY", ""),
    )
