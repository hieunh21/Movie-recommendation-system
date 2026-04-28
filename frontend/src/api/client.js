const BASE = "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  config: () => get("/config"),
  searchMovies: (q, limit = 20) => get(`/movies/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  trendingMovies: (limit = 12) => get(`/movies/trending?limit=${limit}`),
  getMovie: (id) => get(`/movies/${id}`),
  similarMovies: (id, topK = 10) => get(`/movies/${id}/similar?top_k=${topK}`),
  recommendNewUser: (movieIds) => post("/recommend/new-user", { movie_ids: movieIds }),
  recommendExistingUser: (userId, clickSequence) =>
    post("/recommend/existing-user", { user_id: userId, click_sequence: clickSequence }),
  sampleUsers: (limit = 200) => get(`/users/sample?limit=${limit}`),
};
