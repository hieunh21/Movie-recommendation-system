# Movie Recommendation App

Streamlit-based movie recommendation system using MovieLens + TMDB metadata.

Dataset source: MovieLens 1M from Kaggle: https://www.kaggle.com/datasets/cameliabenlaamari/movielens-1m-dataset

The app currently supports:
- BERT4Rec (sequence-based recommendations)
- NeuMF (collaborative filtering)
- Hybrid scoring for existing users
- Content-based similar-movie exploration (TF-IDF + cosine similarity)

## Core Recommendation Logic

### New User
- User explores movies from search/trending.
- Clicking View adds the movie to current session history.
- For You is generated from BERT4Rec using the click sequence.

### Existing User
- User enters/selects user_id.
- NeuMF produces collaborative scores.
- BERT4Rec produces sequence scores from session clicks.
- Final score is fused:

  final = alpha * bert_score + (1 - alpha) * neumf_score

- Alpha is dynamic by sequence length:
  - sequence >= 5: alpha = 0.7
  - sequence < 3: alpha = 0.3
  - otherwise: alpha = 0.5
- Both score sources are min-max normalized to [0, 1] before fusion.
- Already-viewed movies are filtered out.

### Movie Info + Similar Content
- View opens a dedicated movie-info page.
- Similar movies are retrieved from prebuilt content artifacts (offline TF-IDF pipeline).
- Runtime only reads artifacts; no fitting/retraining at request time.

## Offline Build (Content Similarity)

Build artifacts once (or when data changes):

```powershell
python scripts/build_content_similarity.py --base-path model --top-k 20
```

Generated artifacts:
- model/movies_clean.csv
- model/topk_similar.pkl
- model/tfidf_vectorizer.pkl
- model/tfidf_matrix.npz

## Setup

1. Create and activate virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure Streamlit secrets in .streamlit/secrets.toml:

```toml
TMDB_API_KEY = "YOUR_KEY"
TOP_K = 10
MIN_CLICKS_FOR_COLD_START = 3
TMDB_TIMEOUT_SECONDS = 10
```

4. Run app:

```powershell
streamlit run app.py
```

## Project Structure

- app.py: Streamlit entry point
- src/config.py: configuration loader (secrets)
- src/state.py: session-state initialization
- src/ui/layout.py: all UI flows and routing
- src/services/tmdb.py: TMDB client and movie metadata model
- src/services/id_mapper.py: MovieLens <-> TMDB id mapping
- src/services/movies_catalog.py: local movie catalog and search/trending
- src/services/content_similarity.py: content-similar artifact reader
- src/recommenders/bert4rec.py: BERT4Rec scoring and top-k inference
- src/recommenders/neumf.py: NeuMF scoring and top-k inference
- src/recommenders/hybrid.py: fusion recommender
- scripts/build_content_similarity.py: offline TF-IDF artifact builder
- model/: model files and mappings

## Notes

- The app can run without TMDB key, but posters and TMDB metadata may be limited.
- NeuMF/BERT4Rec run in inference mode only; no training occurs in app runtime.
