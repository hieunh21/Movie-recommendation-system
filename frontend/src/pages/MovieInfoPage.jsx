import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useApp } from "../context/AppContext";
import Carousel from "../components/Carousel";

const PLACEHOLDER = "https://placehold.co/300x450/1a1a2e/7c3aed?text=No+Poster";

export default function MovieInfoPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { mode, setNewUser, setExistingUser, movieInfo, setMovieInfo } = useApp();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const movieId = parseInt(id);
    setLoading(true);
    setError("");

    Promise.all([api.getMovie(movieId), api.similarMovies(movieId)])
      .then(([movie, similar]) => {
        setMovieInfo({ movie, similarMovies: similar });
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]); // eslint-disable-line

  const handleViewSimilar = (movie) => {
    if (mode === "new") {
      setNewUser((p) => {
        if (p.viewedIds.includes(movie.movie_id)) return p;
        return { ...p, viewedIds: [...p.viewedIds, movie.movie_id] };
      });
    } else {
      setExistingUser((p) => {
        if (p.clickSequence.includes(movie.movie_id)) return p;
        return { ...p, clickSequence: [...p.clickSequence, movie.movie_id] };
      });
    }
  };

  if (loading) return <div className="page"><div className="spinner">Loading…</div></div>;
  if (error) return (
    <div className="page">
      <div className="page-nav">
        <button className="btn-back" onClick={() => navigate(-1)}>← Back</button>
        <button className="btn-home" onClick={() => navigate("/")}>🏠 Home</button>
      </div>
      <p className="error-msg">{error}</p>
    </div>
  );

  const { movie, similarMovies } = movieInfo;

  return (
    <div className="page">
      <div className="page-nav">
        <button className="btn-back" onClick={() => navigate(-1)}>← Back</button>
        <button className="btn-home" onClick={() => navigate("/")}>🏠 Home</button>
      </div>

      {/* Selected Movie */}
      <div className="movie-detail">
        <img
          className="detail-poster"
          src={movie?.poster_url || PLACEHOLDER}
          alt={movie?.title}
          onError={(e) => { e.target.src = PLACEHOLDER; }}
        />
        <div className="detail-info">
          <h1 className="detail-title">{movie?.title}</h1>
          <p className="detail-meta">
            {movie?.year}
            {movie?.vote_average ? ` · ⭐ ${movie.vote_average.toFixed(1)}` : ""}
          </p>
          {movie?.genres?.length > 0 && (
            <div className="detail-genres">
              {movie.genres.map((g) => <span key={g} className="genre-tag">{g}</span>)}
            </div>
          )}
          {movie?.overview && <p className="detail-overview">{movie.overview}</p>}
          <p className="detail-id">MovieLens ID: {id}</p>
        </div>
      </div>

      {/* Similar Movies */}
      <section className="content-section">
        <h2 className="section-title">🎯 Top 10 Similar Movies (Content-based)</h2>
        {similarMovies.length === 0
          ? <p className="empty-msg">No similar movies found. Run the offline build script first.</p>
          : (
            <Carousel movies={similarMovies} onView={handleViewSimilar} />
          )
        }
      </section>
    </div>
  );
}
