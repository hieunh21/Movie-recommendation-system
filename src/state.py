from typing import Dict, List

import streamlit as st


DEFAULT_NEW_USER_STATE: Dict[str, object] = {
    "selected_movie_ids": [],
    "last_recommendations": [],
    "last_recommendation_error": "",
    "search_query": "",
    "search_results": [],
}

DEFAULT_EXISTING_USER_STATE: Dict[str, object] = {
    "user_id": "",
    "last_recommendations": [],
    "last_recommendation_error": "",
    "session_click_movie_ids": [],
}

DEFAULT_MOVIE_INFO_STATE: Dict[str, object] = {
    "selected_movie_id": 0,
    "similar_movie_ids": [],
    "similar_scores": {},
    "error": "",
}


def _ensure_list(value: object) -> List[int]:
    if isinstance(value, list):
        return value
    return []


def _ensure_str(value: object, default: str = "") -> str:
    if isinstance(value, str):
        return value
    return default


def _ensure_dict(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return value
    return {}


def init_session_state() -> None:
    if "active_user_mode" not in st.session_state:
        st.session_state.active_user_mode = "new"

    if "active_page" not in st.session_state:
        st.session_state.active_page = "home"

    if "new_user" not in st.session_state:
        st.session_state.new_user = DEFAULT_NEW_USER_STATE.copy()

    if "existing_user" not in st.session_state:
        st.session_state.existing_user = DEFAULT_EXISTING_USER_STATE.copy()

    if "movie_info" not in st.session_state:
        st.session_state.movie_info = DEFAULT_MOVIE_INFO_STATE.copy()

    st.session_state.new_user["selected_movie_ids"] = _ensure_list(
        st.session_state.new_user.get("selected_movie_ids")
    )
    st.session_state.new_user["last_recommendations"] = _ensure_list(
        st.session_state.new_user.get("last_recommendations")
    )
    st.session_state.new_user["search_query"] = _ensure_str(
        st.session_state.new_user.get("search_query")
    )
    st.session_state.new_user["last_recommendation_error"] = _ensure_str(
        st.session_state.new_user.get("last_recommendation_error")
    )
    if not isinstance(st.session_state.new_user.get("search_results"), list):
        st.session_state.new_user["search_results"] = []
    st.session_state.existing_user["last_recommendations"] = _ensure_list(
        st.session_state.existing_user.get("last_recommendations")
    )
    st.session_state.existing_user["session_click_movie_ids"] = _ensure_list(
        st.session_state.existing_user.get("session_click_movie_ids")
    )
    st.session_state.existing_user["last_recommendation_error"] = _ensure_str(
        st.session_state.existing_user.get("last_recommendation_error")
    )

    if not isinstance(st.session_state.movie_info.get("selected_movie_id"), int):
        st.session_state.movie_info["selected_movie_id"] = 0
    st.session_state.movie_info["similar_movie_ids"] = _ensure_list(
        st.session_state.movie_info.get("similar_movie_ids")
    )
    st.session_state.movie_info["similar_scores"] = _ensure_dict(
        st.session_state.movie_info.get("similar_scores")
    )
    st.session_state.movie_info["error"] = _ensure_str(
        st.session_state.movie_info.get("error")
    )

    if not isinstance(st.session_state.get("active_page"), str):
        st.session_state.active_page = "home"
