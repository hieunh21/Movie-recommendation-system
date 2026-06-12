import { useState, useEffect, useCallback } from "react";
import { useApp } from "../context/AppContext";
import Carousel from "../components/Carousel";
import { api } from "../api/client";

const SESSION_MODELS = [
  { value: "bert4rec", label: "BERT4Rec" },
  { value: "bert4rec_improved", label: "BERT4Rec+" },
  { value: "lrurec", label: "LRURec" },
];

const HISTORY_MODELS = [
  { value: "neumf", label: "NeuMF" },
  { value: "lightgcn", label: "LightGCN" },
];

// ─── New User ────────────────────────────────────────────────────────────────
function NewUserView() {
  const { newUser, setNewUser } = useApp();
  const [query, setQuery] = useState("");
  const [trending, setTrending] = useState([]);
  const [searching, setSearching] = useState(false);
  const [recLoading, setRecLoading] = useState(false);
  const [minClicks, setMinClicks] = useState(3);

  useEffect(() => {
    api.config().then((c) => setMinClicks(c.min_clicks_for_cold_start)).catch(() => {});
    // Trending requires BERT4Rec to load first-time; retry once after a delay if empty
    const loadTrending = () =>
      api.trendingMovies(10).then((data) => {
        setTrending(data);
        if (data.length === 0) setTimeout(() => api.trendingMovies().then(setTrending).catch(() => {}), 8000);
      }).catch(console.error);
    loadTrending();
  }, []);

  // Fetch recommendations whenever viewedIds grows past threshold
  useEffect(() => {
    if (newUser.viewedIds.length < minClicks) {
      setNewUser((p) => ({ ...p, recommendations: [] }));
      return;
    }
    setRecLoading(true);
    api
      .recommendNewUser(newUser.viewedIds, newUser.model)
      .then((recs) => setNewUser((p) => ({ ...p, recommendations: recs })))
      .catch(console.error)
      .finally(() => setRecLoading(false));
  }, [newUser.viewedIds, minClicks]); // eslint-disable-line

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    try {
      const results = await api.searchMovies(query, 10);
      setNewUser((p) => ({ ...p, searchResults: results }));
    } catch (err) {
      console.error(err);
    } finally {
      setSearching(false);
    }
  };

  const handleView = useCallback((movie) => {
    setNewUser((p) => {
      const next = p.viewedIds.includes(movie.movie_id)
        ? p
        : { ...p, viewedIds: [...p.viewedIds, movie.movie_id] };
      return { ...next, searchResults: [] }; // xóa kết quả search khi navigate đi
    });
  }, []); // eslint-disable-line

  const remaining = Math.max(0, minClicks - newUser.viewedIds.length);

  return (
    <>
      <ModelPicker
        label="Session model"
        value={newUser.model}
        options={SESSION_MODELS}
        onChange={(model) =>
          setNewUser({
            viewedIds: [],
            searchResults: [],
            recommendations: [],
            model,
          })
        }
      />

      {/* Search */}
      <form className="search-bar" onSubmit={handleSearch}>
        <input
          className="search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search movies by title…"
        />
        <button className="btn-primary" type="submit" disabled={searching}>
          {searching ? "…" : "Search"}
        </button>
      </form>

      {newUser.searchResults.length > 0 && (
        <Section title="Search Results">
          <Carousel movies={newUser.searchResults} onView={handleView} />
        </Section>
      )}

      <Section title="🔥 Trending">
        <Carousel movies={trending} onView={handleView} />
      </Section>

      {newUser.viewedIds.length > 0 && (
        <div className="hint-bar">
          {remaining > 0
            ? `View ${remaining} more movie(s) to unlock your "For You" list`
            : "✨ Personalized recommendations are ready below!"}
        </div>
      )}

      {(newUser.recommendations.length > 0 || recLoading) && (
        <Section title={`✨ For You — ${labelFor(SESSION_MODELS, newUser.model)}`}>
          {recLoading
            ? <Spinner />
            : <Carousel movies={newUser.recommendations} onView={handleView} />}
        </Section>
      )}
    </>
  );
}

// ─── Existing User ────────────────────────────────────────────────────────────
function ExistingUserView() {
  const { existingUser, setExistingUser } = useApp();
  const [inputId, setInputId] = useState(existingUser.userId);
  const [userList, setUserList] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.sampleUsers(20).then(setUserList).catch(console.error);
  }, []);

  const fetchRecs = useCallback(async (userId, clickSeq, sessionModel, historyModel) => {
    setLoading(true);
    try {
      const recs = await api.recommendExistingUser(parseInt(userId), clickSeq, sessionModel, historyModel);
      setExistingUser((p) => ({ ...p, recommendations: recs }));
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []); // eslint-disable-line

  const applyUser = (uid) => {
    const uidStr = String(uid);
    // Reset clickSequence khi đổi sang user khác
    setInputId(uidStr);
    setExistingUser((p) => ({ ...p, userId: uidStr, clickSequence: [] }));
    fetchRecs(uid, [], existingUser.sessionModel, existingUser.historyModel);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const uid = inputId.trim();
    if (!uid || isNaN(parseInt(uid))) return;
    applyUser(uid);
  };

  const handleView = useCallback((movie) => {
    setExistingUser((prev) => {
      if (prev.clickSequence.includes(movie.movie_id)) return prev;
      const newSeq = [...prev.clickSequence, movie.movie_id];
      if (prev.userId) fetchRecs(prev.userId, newSeq, prev.sessionModel, prev.historyModel);
      return { ...prev, clickSequence: newSeq };
    });
  }, [fetchRecs]); // eslint-disable-line

  const seq = existingUser.clickSequence.length;
  const alpha = seq === 0 ? 0.0 : seq <= 2 ? 0.3 : seq <= 4 ? 0.5 : 0.7;

  return (
    <>
      <ModelPicker
        label="History model"
        value={existingUser.historyModel}
        options={HISTORY_MODELS}
        onChange={(historyModel) =>
          setExistingUser((p) => ({
            ...p,
            clickSequence: [],
            recommendations: [],
            historyModel,
          }))
        }
      />

      <ModelPicker
        label="Session model"
        value={existingUser.sessionModel}
        options={SESSION_MODELS}
        onChange={(sessionModel) =>
          setExistingUser((p) => ({
            ...p,
            clickSequence: [],
            recommendations: [],
            sessionModel,
          }))
        }
      />

      {/* User ID input */}
      <form className="search-bar" onSubmit={handleSubmit}>
        <input
          className="search-input"
          value={inputId}
          onChange={(e) => setInputId(e.target.value)}
          placeholder="Enter User ID (e.g. 1, 42, 500)…"
        />
        <button className="btn-primary" type="submit">Go</button>
      </form>

      {/* Quick-pick chips */}
      {userList.length > 0 && (
        <div className="chip-row">
          <span className="chip-label">Quick pick:</span>
          {userList.map((uid) => (
            <button
              key={uid}
              className={`chip ${existingUser.userId == uid ? "chip-active" : ""}`}
              onClick={() => applyUser(uid)}
            >
              #{uid}
            </button>
          ))}
        </div>
      )}

      {existingUser.userId && (
        <div className="hint-bar">
          User #{existingUser.userId} · Session clicks: {seq} · Hybrid α = {alpha.toFixed(1)}
          {" · "}
          {labelFor(HISTORY_MODELS, existingUser.historyModel)}
          {seq > 0 ? ` + ${labelFor(SESSION_MODELS, existingUser.sessionModel)}` : ""}
        </div>
      )}

      {loading && <Spinner />}

      {!loading && existingUser.recommendations.length > 0 && (
        <Section title={`✨ For You — ${seq > 0 ? "Hybrid" : labelFor(HISTORY_MODELS, existingUser.historyModel)}`}>
          <Carousel movies={existingUser.recommendations} onView={handleView} />
        </Section>
      )}
    </>
  );
}

// ─── Shared helpers ──────────────────────────────────────────────────────────
function Section({ title, children }) {
  return (
    <section className="content-section">
      <h2 className="section-title">{title}</h2>
      {children}
    </section>
  );
}

function ModelPicker({ label, value, options, onChange }) {
  return (
    <div className="model-picker">
      <span className="model-picker-label">{label}</span>
      <div className="model-options">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`model-option ${value === option.value ? "model-option-active" : ""}`}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function labelFor(options, value) {
  return options.find((option) => option.value === value)?.label || value;
}



function Spinner() {
  return <div className="spinner">Loading…</div>;
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const { mode } = useApp();
  return (
    <div className="page">
      <h1 className="page-title">
        {mode === "new" ? "Discover Movies" : "Your Recommendations"}
      </h1>
      {mode === "new" ? <NewUserView /> : <ExistingUserView />}
    </div>
  );
}
