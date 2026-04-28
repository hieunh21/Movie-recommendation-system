import { useNavigate } from "react-router-dom";

const PLACEHOLDER = "https://placehold.co/300x450/1a1a2e/7c3aed?text=No+Poster";

export default function MovieCard({ movie, onView, badge, disabled = false }) {
  const navigate = useNavigate();
  const genres = (movie.genres || []).slice(0, 3).join(" · ");
  const year = movie.year || "N/A";
  const score = movie.vote_average ? movie.vote_average.toFixed(1) : null;

  const handleView = () => {
    if (onView) onView(movie);
    navigate(`/movie/${movie.movie_id}`);
  };

  return (
    <div className="movie-card">
      <div className="poster-wrap">
        <img
          src={movie.poster_url || PLACEHOLDER}
          alt={movie.title}
          onError={(e) => { e.target.src = PLACEHOLDER; }}
        />
        {score && <span className="score-badge">⭐ {score}</span>}
      </div>
      <div className="card-body">
        <p className="card-title" title={movie.title}>{movie.title}</p>
        <p className="card-meta">{year}{genres ? ` · ${genres}` : ""}</p>
        {badge && <p className="card-badge">{badge}</p>}
        <button className="btn-view" onClick={handleView} disabled={disabled}>
          View
        </button>
      </div>
    </div>
  );
}
