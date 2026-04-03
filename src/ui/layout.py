from pathlib import Path
from typing import Dict, List, Literal

import streamlit as st

from src.config import AppConfig
from src.recommenders.bert4rec import BERT4RecInferenceError, BERT4RecRecommender
from src.recommenders.hybrid import HybridInferenceError, HybridRecommender
from src.recommenders.neumf import NeuMFInferenceError, NeuMFRecommender
from src.services.content_similarity import ContentSimilarityError, ContentSimilarityService
from src.services.id_mapper import MovieIdMapper
from src.services.movies_catalog import MovieCatalog, MovieCatalogItem
from src.services.tmdb import MovieSummary, TMDBClient, TMDBClientError

UserMode = Literal["new", "existing"]
MODEL_DIR = Path("model")


def render_sidebar(config: AppConfig) -> UserMode:
    st.sidebar.header("User Mode")
    mode_label = st.sidebar.radio(
        "Select profile",
        options=["New user", "Existing user"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.caption("App settings")
    st.sidebar.write(f"Top-K recommendations: {config.top_k}")
    st.sidebar.write(f"Min clicks (cold-start): {config.min_clicks_for_cold_start}")

    return "new" if mode_label == "New user" else "existing"


@st.cache_resource(show_spinner=False)
def _get_tmdb_client(api_key: str, timeout_seconds: int) -> TMDBClient:
    return TMDBClient(api_key=api_key, timeout_seconds=timeout_seconds)


def _get_tmdb_client_optional(config: AppConfig) -> TMDBClient | None:
    if not config.tmdb_api_key:
        return None
    try:
        return _get_tmdb_client(config.tmdb_api_key, config.tmdb_timeout_seconds)
    except Exception:
        return None


def _find_first_path(candidates: List[str]) -> str:
    for candidate in candidates:
        path = MODEL_DIR / candidate
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Missing expected file. Checked: {candidates}")


@st.cache_resource(show_spinner=False)
def _get_id_mapper() -> MovieIdMapper:
    link_path = _find_first_path(["link.csv"])
    return MovieIdMapper.from_link_csv(link_path)


@st.cache_resource(show_spinner=False)
def _get_movies_catalog() -> MovieCatalog:
    movies_path = _find_first_path(["movies.csv"])
    return MovieCatalog.from_movies_csv(movies_path)


@st.cache_resource(show_spinner=False)
def _get_bert4rec_recommender() -> BERT4RecRecommender:
    checkpoint_path = _find_first_path(["best_bert4rec_ml1m (1).pt", "best_bert4rec_ml1m.pt"])
    mapping_path = _find_first_path([
        "item_mapping_bert4rec_ml1m (1).json",
        "item_mapping_bert4rec_ml1m.json",
    ])
    return BERT4RecRecommender(checkpoint_path=checkpoint_path, mapping_path=mapping_path, device="cpu")


@st.cache_resource(show_spinner=False)
def _get_neumf_recommender() -> NeuMFRecommender:
    model_path = _find_first_path(["NeuMF.keras"])
    user_mapping_path = _find_first_path(["user_mapping_neumf.json"])
    item_mapping_path = _find_first_path(["item_mapping_neumf.json"])
    candidates_path = _find_first_path(["neumf_candidates.csv"])
    popular_movies_path = _find_first_path(["popular_movies.csv"])
    return NeuMFRecommender(
        model_path=model_path,
        user_mapping_path=user_mapping_path,
        item_mapping_path=item_mapping_path,
        candidates_path=candidates_path,
        popular_movies_path=popular_movies_path,
    )


@st.cache_resource(show_spinner=False)
def _get_hybrid_recommender() -> HybridRecommender:
    return HybridRecommender(bert=_get_bert4rec_recommender(), neumf=_get_neumf_recommender())


@st.cache_resource(show_spinner=False)
def _get_content_similarity_service() -> ContentSimilarityService:
    artifact_path = _find_first_path(["topk_similar.pkl"])
    return ContentSimilarityService(artifact_path)


@st.cache_data(show_spinner=False)
def _get_quick_user_ids(limit: int = 200) -> List[int]:
    recommender = _get_neumf_recommender()
    all_ids = sorted(recommender.artifacts.user2idx.keys())
    return all_ids[:limit]


def _render_movie_card(
    movie: MovieSummary,
    button_key: str,
    button_label: str,
    extra_caption: str = "",
    button_disabled: bool = False,
    info_button_key: str | None = None,
    image_width: int | None = None,
) -> tuple[bool, bool]:
    overview_text = str(getattr(movie, "overview", "") or "").strip()
    if len(overview_text) > 180:
        overview_text = overview_text[:177].rstrip() + "..."

    genres_value = getattr(movie, "genres", [])
    genres_text = ", ".join(genres_value) if genres_value else ""

    with st.container(border=True):
        if movie.poster_url:
            if image_width is not None:
                left_col, mid_col, right_col = st.columns([1, 2, 1])
                with mid_col:
                    st.image(movie.poster_url, width=image_width)
            else:
                st.image(movie.poster_url, use_container_width=True)
        st.markdown(f"**{movie.title}**")
        st.caption(f"{movie.year} • TMDB {movie.vote_average:.1f}")

        # Fixed-height overview block for consistent card layout.
        st.markdown(
            f"<div style='min-height:72px; max-height:72px; overflow:hidden; color: rgba(49,51,63,0.7); font-size:0.9rem;'>{overview_text}</div>",
            unsafe_allow_html=True,
        )

        if extra_caption:
            st.caption(extra_caption)

        # Fixed-height genre block for consistent card layout.
        st.markdown(
            f"<div style='min-height:40px; max-height:40px; overflow:hidden; color: rgba(49,51,63,0.7); font-size:0.9rem;'>{genres_text}</div>",
            unsafe_allow_html=True,
        )

        action_clicked = False
        if button_label:
            action_clicked = st.button(
                button_label,
                key=button_key,
                use_container_width=True,
                disabled=button_disabled,
            )
        info_clicked = False
        if info_button_key is not None:
            info_clicked = st.button(
                "View",
                key=info_button_key,
                use_container_width=True,
            )
        return action_clicked, info_clicked


def _track_view_for_new_user(movie_id: int) -> None:
    mode = st.session_state.get("active_user_mode")
    if mode == "new":
        key = "selected_movie_ids"
        container = st.session_state.new_user
    elif mode == "existing":
        key = "session_click_movie_ids"
        container = st.session_state.existing_user
    else:
        return

    ids = container.get(key, [])
    if movie_id not in ids:
        ids.append(movie_id)
        container[key] = ids


def _safe_hydrate(
    movie_ids: List[int],
    mapper: MovieIdMapper,
    client: TMDBClient | None,
) -> Dict[int, MovieSummary]:
    if client is None or not movie_ids:
        return {}
    try:
        return _hydrate_movielens_ids(movie_ids, mapper, client)
    except TMDBClientError:
        return {}


def _show_movie_info(movie_id: int) -> None:
    _track_view_for_new_user(int(movie_id))
    try:
        service = _get_content_similarity_service()
        similar = service.get_similar(movie_id, top_k=10)
    except (FileNotFoundError, ContentSimilarityError) as exc:
        st.session_state.movie_info["error"] = str(exc)
        st.session_state.movie_info["selected_movie_id"] = int(movie_id)
        st.session_state.movie_info["similar_movie_ids"] = []
        st.session_state.movie_info["similar_scores"] = {}
        return

    st.session_state.movie_info["selected_movie_id"] = int(movie_id)
    st.session_state.movie_info["similar_movie_ids"] = [item.movie_id for item in similar]
    st.session_state.movie_info["similar_scores"] = {
        int(item.movie_id): float(item.score) for item in similar
    }
    st.session_state.movie_info["error"] = ""
    st.session_state.active_page = "movie_info"


def _hydrate_movielens_ids(
    movie_ids: List[int],
    mapper: MovieIdMapper,
    client: TMDBClient,
) -> Dict[int, MovieSummary]:
    tmdb_to_movie: Dict[int, int] = {}
    tmdb_ids: List[int] = []

    for movie_id in movie_ids:
        tmdb_id = mapper.to_tmdb(movie_id)
        if tmdb_id is None:
            continue
        tmdb_to_movie[tmdb_id] = movie_id
        tmdb_ids.append(tmdb_id)

    summaries_by_movie_id: Dict[int, MovieSummary] = {}
    if not tmdb_ids:
        return summaries_by_movie_id

    hydrated = client.hydrate_movies(tmdb_ids)
    for summary in hydrated:
        original_movie_id = tmdb_to_movie.get(summary.movie_id)
        if original_movie_id is not None:
            summaries_by_movie_id[original_movie_id] = summary

    return summaries_by_movie_id


def _summary_from_catalog(movie_id: int, catalog: MovieCatalog) -> MovieSummary:
    item = catalog.get(movie_id)
    if item is None:
        return MovieSummary(
            movie_id=movie_id,
            title=f"MovieLens #{movie_id}",
            year="N/A",
            poster_url="",
            vote_average=0.0,
            genres=[],
            overview="",
        )

    return MovieSummary(
        movie_id=movie_id,
        title=item.title,
        year=str(item.year) if item.year is not None else "N/A",
        poster_url="",
        vote_average=0.0,
        genres=item.genres,
        overview="",
    )


def _render_movie_info_section(
    catalog: MovieCatalog,
    mapper: MovieIdMapper,
    client: TMDBClient | None,
) -> None:
    st.markdown("### Movie Info & Similar Content")

    info_error = st.session_state.movie_info.get("error", "")
    if info_error:
        st.error(info_error)
        st.caption("Run offline build first: python scripts/build_content_similarity.py")

    selected_movie_id = int(st.session_state.movie_info.get("selected_movie_id", 0) or 0)
    similar_ids: List[int] = st.session_state.movie_info.get("similar_movie_ids", [])
    similar_scores: Dict[int, float] = st.session_state.movie_info.get("similar_scores", {})

    if selected_movie_id <= 0:
        st.info("Click 'View' on any movie card to see details and top-10 similar movies.")
        return

    selected_map = _safe_hydrate([selected_movie_id], mapper, client)

    selected_movie = selected_map.get(selected_movie_id, _summary_from_catalog(selected_movie_id, catalog))
    st.markdown("#### Selected Movie")
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        _render_movie_card(
            selected_movie,
            button_key=f"selected_info_{selected_movie_id}",
            button_label="Selected",
            extra_caption=f"MovieLens ID: {selected_movie_id}",
            button_disabled=True,
            info_button_key=None,
            image_width=320,
        )

    if not similar_ids:
        st.info("No similar movies available for this title.")
        return

    similar_map = _safe_hydrate(similar_ids, mapper, client)

    st.markdown("#### Top 10 Similar Movies (Content-based)")
    cols = st.columns(5)
    for idx, movie_id in enumerate(similar_ids[:10]):
        movie = similar_map.get(movie_id, _summary_from_catalog(movie_id, catalog))
        score = float(similar_scores.get(movie_id, 0.0))
        with cols[idx % 5]:
            _, open_info_clicked = _render_movie_card(
                movie,
                button_key=f"similar_info_{selected_movie_id}_{movie_id}",
                button_label="",
                extra_caption=f"Score: {score:.4f}",
                info_button_key=f"similar_open_info_{selected_movie_id}_{movie_id}",
            )
            if open_info_clicked:
                _show_movie_info(movie_id)
                st.rerun()


def _render_movie_info_page(config: AppConfig) -> None:
    st.subheader("Movie Info")
    st.caption("Movie Detail")

    if st.button("Back to Home", type="secondary"):
        st.session_state.active_page = "home"
        st.rerun()

    try:
        mapper = _get_id_mapper()
        catalog = _get_movies_catalog()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    client = _get_tmdb_client_optional(config)
    _render_movie_info_section(catalog, mapper, client)


def _render_new_user_phase3(config: AppConfig) -> None:
    st.subheader("New User")

    try:
        mapper = _get_id_mapper()
        recommender = _get_bert4rec_recommender()
        catalog = _get_movies_catalog()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except BERT4RecInferenceError as exc:
        st.error(str(exc))
        return

    eligible_ids = set(recommender.artifacts.item2idx.keys())
    st.caption(f"{catalog.count(eligible_ids)} movies available")

    client = _get_tmdb_client_optional(config)

    # Streamlit cache may keep an old MovieCatalog instance created before
    # trending_local was added; rebuild on the fly to avoid runtime crashes.
    if not hasattr(catalog, "trending_local"):
        catalog = MovieCatalog.from_movies_csv(_find_first_path(["movies.csv"]))

    with st.form("movie_search_form", clear_on_submit=False):
        query = st.text_input(
            "Search movies",
            value=st.session_state.new_user.get("search_query", ""),
            placeholder="Type movie title from movies.csv",
        )
        search_submitted = st.form_submit_button("Search")

    if search_submitted:
        st.session_state.new_user["search_query"] = query
        local_matches = catalog.search(query, allowed_ids=eligible_ids, limit=20)
        st.session_state.new_user["search_results"] = [item.movie_id for item in local_matches]

    search_result_ids: List[int] = st.session_state.new_user.get("search_results", [])
    search_results: List[MovieCatalogItem] = [
        item for item in (catalog.get(movie_id) for movie_id in search_result_ids) if item is not None
    ]

    search_map = _safe_hydrate(search_result_ids, mapper, client)

    if search_results:
        st.markdown("### Search Results")
        result_cols = st.columns(4)
        for index, item in enumerate(search_results):
            movie = search_map.get(item.movie_id, _summary_from_catalog(item.movie_id, catalog))
            with result_cols[index % 4]:
                _, info_clicked = _render_movie_card(
                    movie,
                    button_key=f"search_add_local_{item.movie_id}",
                    button_label="",
                    extra_caption=f"MovieLens ID: {item.movie_id}",
                    info_button_key=f"search_info_local_{item.movie_id}",
                )
                if info_clicked:
                    _show_movie_info(item.movie_id)
                    st.rerun()
    elif st.session_state.new_user.get("search_query", ""):
        st.info("No matching movies found in eligible movies.csv catalog.")

    st.markdown("### Trending")
    local_trending = catalog.trending_local(allowed_ids=eligible_ids, limit=12)
    trending_ids = [item.movie_id for item in local_trending]
    trending_map = _safe_hydrate(trending_ids, mapper, client)

    trending_cols = st.columns(4)
    for index, item in enumerate(local_trending):
        movie = trending_map.get(item.movie_id, _summary_from_catalog(item.movie_id, catalog))
        with trending_cols[index % 4]:
            _, info_clicked = _render_movie_card(
                movie,
                button_key=f"trend_add_local_{item.movie_id}",
                button_label="",
                extra_caption=f"MovieLens ID: {item.movie_id}",
                info_button_key=f"trend_info_local_{item.movie_id}",
            )
            if info_clicked:
                _show_movie_info(item.movie_id)
                st.rerun()

    selected_ids = st.session_state.new_user.get("selected_movie_ids", [])
    st.markdown("### Viewed Movies")
    if not selected_ids:
        st.info("No viewing history yet. Click View on any movie to get started.")
    else:
        selected_map = _safe_hydrate(selected_ids, mapper, client)

        st.write(f"Selected count: {len(selected_ids)}")
        selected_cols = st.columns(4)
        for index, movie_id in enumerate(selected_ids):
            movie = selected_map.get(
                movie_id,
                _summary_from_catalog(movie_id, catalog),
            )
            with selected_cols[index % 4]:
                _, info_clicked = _render_movie_card(
                    movie,
                    button_key=f"remove_{movie_id}",
                    button_label="",
                    extra_caption=f"MovieLens ID: {movie_id}",
                    info_button_key=f"selected_info_local_{movie_id}",
                )
                if info_clicked:
                    _show_movie_info(movie_id)
                    st.rerun()

    st.markdown("### For You (BERT4Rec)")
    if len(selected_ids) < config.min_clicks_for_cold_start:
        st.session_state.new_user["last_recommendations"] = []
        st.info(
            f"At least {config.min_clicks_for_cold_start} views are required to generate personalized recommendations."
        )
    else:
        try:
            recommendations = recommender.recommend(selected_ids, top_k=config.top_k)
            st.session_state.new_user["last_recommendations"] = recommendations
            st.session_state.new_user["last_recommendation_error"] = ""
        except (FileNotFoundError, BERT4RecInferenceError) as exc:
            st.session_state.new_user["last_recommendation_error"] = str(exc)

    if st.session_state.new_user.get("last_recommendation_error"):
        st.error(st.session_state.new_user["last_recommendation_error"])

    last_recommendations: List[int] = st.session_state.new_user.get("last_recommendations", [])
    if last_recommendations:
        rec_map = _safe_hydrate(last_recommendations, mapper, client)

        rec_cols = st.columns(5)
        for index, movie_id in enumerate(last_recommendations):
            movie = rec_map.get(
                movie_id,
                _summary_from_catalog(movie_id, catalog),
            )
            with rec_cols[index % 5]:
                _, info_clicked = _render_movie_card(
                    movie,
                    button_key=f"rec_item_{movie_id}",
                    button_label="",
                    extra_caption=f"MovieLens ID: {movie_id}",
                    button_disabled=True,
                    info_button_key=f"rec_info_local_{movie_id}",
                )
                if info_clicked:
                    _show_movie_info(movie_id)
                    st.rerun()


def _render_existing_user_phase2(config: AppConfig) -> None:
    st.subheader("Existing User")
    
    # Load recommender and other services
    try:
        recommender = _get_neumf_recommender()
        hybrid = _get_hybrid_recommender()
        mapper = _get_id_mapper()
        client = _get_tmdb_client_optional(config)
    except FileNotFoundError as exc:
        st.error(f"Failed to load models: {exc}")
        return

    catalog = _get_movies_catalog()

    quick_user_ids: List[int] = []
    try:
        quick_user_ids = _get_quick_user_ids()
    except Exception:
        quick_user_ids = []

    st.markdown("### For You")

    input_col, pick_col = st.columns([2, 2])
    with input_col:
        user_id_text = st.text_input(
            "Nhập User ID",
            value=st.session_state.existing_user.get("user_id", ""),
            placeholder="Ví dụ: 2",
        )
    with pick_col:
        selected_user = st.selectbox(
            "Hoặc chọn nhanh User ID",
            options=["-- Chọn user --"] + [str(uid) for uid in quick_user_ids],
            index=0,
        )

    user_id = selected_user if selected_user != "-- Chọn user --" else user_id_text
    st.session_state.existing_user["user_id"] = user_id

    if not user_id:
        st.info("No user_id selected. Showing the default For You list (popular).")
        st.session_state.existing_user["last_recommendations"] = recommender.artifacts.popular_movies[: config.top_k]
    else:
        # Validate user ID format
        try:
            user_id_int = int(user_id)
        except ValueError:
            st.error("User ID must be a valid integer.")
            return

        # Auto-generate hybrid recommendations as soon as user_id is valid.
        click_seq = st.session_state.existing_user.get("session_click_movie_ids", [])
        try:
            recommendations = hybrid.recommend(user_id_int, click_seq, top_k=config.top_k)
            if not recommendations:
                recommendations = recommender.artifacts.popular_movies[: config.top_k]
            st.session_state.existing_user["last_recommendations"] = recommendations
            st.session_state.existing_user["last_recommendation_error"] = ""
        except (HybridInferenceError, NeuMFInferenceError) as exc:
            st.session_state.existing_user["last_recommendation_error"] = str(exc)

        # Keep a non-empty For You list even when inference yields nothing.
        if not st.session_state.existing_user.get("last_recommendations"):
            st.session_state.existing_user["last_recommendations"] = recommender.artifacts.popular_movies[: config.top_k]

    # Display error if any
    if st.session_state.existing_user.get("last_recommendation_error"):
        st.error(st.session_state.existing_user["last_recommendation_error"])

    # Display recommendations
    last_recommendations: List[int] = st.session_state.existing_user.get("last_recommendations", [])
    
    if last_recommendations:
        click_count = len(st.session_state.existing_user.get("session_click_movie_ids", []))
        alpha = 0.7 if click_count >= 5 else (0.3 if click_count < 3 else 0.5)
        st.caption(f"Hybrid alpha: {alpha:.1f} (session clicks: {click_count})")

        # Check if user is cold-start (not in training)
        is_cold_start = (not user_id) or (user_id.isdigit() and int(user_id) not in recommender.artifacts.user2idx)
        if is_cold_start:
            if user_id:
                st.info(
                    f"User ID {user_id} is not in training data. Showing popular movies instead."
                )
            else:
                st.info("Showing popular movies as default For You list.")

        st.markdown("### For You (Hybrid Top 10)")
        
        # Hydrate with TMDB data
        rec_map = _safe_hydrate(last_recommendations, mapper, client)

        rec_cols = st.columns(5)
        for index, movie_id in enumerate(last_recommendations):
            movie = rec_map.get(
                movie_id,
                _summary_from_catalog(movie_id, catalog),
            )
            with rec_cols[index % 5]:
                _, info_clicked = _render_movie_card(
                    movie,
                    button_key=f"neumf_rec_item_{movie_id}",
                    button_label="",
                    extra_caption=f"MovieLens ID: {movie_id}",
                    button_disabled=True,
                    info_button_key=f"neumf_rec_info_{movie_id}",
                )
                if info_clicked:
                    _show_movie_info(movie_id)
                    st.rerun()
    elif user_id:
        st.info("Khong co ket qua ca nhan hoa, dang hien thi goi y mac dinh.")


def render_phase3_home(mode: UserMode, config: AppConfig) -> None:
    st.title("Movie Recommendation App")

    if not config.tmdb_api_key:
        st.warning(
            "TMDB_API_KEY is not configured. You can still select movies from movies.csv and run BERT4Rec, but posters/ratings from TMDB will be disabled."
        )

    if st.session_state.get("active_page") == "movie_info":
        _render_movie_info_page(config)
        return

    if mode == "new":
        _render_new_user_phase3(config)
    else:
        _render_existing_user_phase2(config)
