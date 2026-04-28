import { useRef, useState, useEffect } from "react";
import MovieCard from "./MovieCard";


export default function Carousel({ movies, onView }) {
  const trackRef = useRef(null);
  const [canLeft, setCanLeft] = useState(false);
  const [canRight, setCanRight] = useState(false);

  const syncButtons = () => {
    const el = trackRef.current;
    if (!el) return;
    setCanLeft(el.scrollLeft > 1);
    setCanRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  };

  useEffect(() => {
    syncButtons();
  }, [movies]);

  const scroll = (dir) => {
    const el = trackRef.current;
    if (!el) return;
    // Cuộn đúng 1 trang; click cuối luôn đến tận cùng (card cuối không bị khuất)
    const target = dir > 0
      ? Math.min(el.scrollLeft + el.clientWidth, el.scrollWidth - el.clientWidth)
      : Math.max(el.scrollLeft - el.clientWidth, 0);
    el.scrollTo({ left: target, behavior: "smooth" });
    setTimeout(syncButtons, 400);
  };

  if (!movies.length) return <p className="empty-msg">No movies to show.</p>;

  return (
    <div className="carousel">
      <button
        className="carousel-btn"
        onClick={() => scroll(-1)}
        disabled={!canLeft}
        aria-label="Scroll left"
      >
        ‹
      </button>

      <div className="carousel-track" ref={trackRef} onScroll={syncButtons}>
        {movies.map((m) => (
          <div key={m.movie_id} className="carousel-item">
            <MovieCard movie={m} onView={onView} />
          </div>
        ))}
      </div>

      <button
        className="carousel-btn"
        onClick={() => scroll(1)}
        disabled={!canRight}
        aria-label="Scroll right"
      >
        ›
      </button>
    </div>
  );
}
