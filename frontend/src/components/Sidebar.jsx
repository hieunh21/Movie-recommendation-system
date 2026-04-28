import { useNavigate } from "react-router-dom";
import { useApp } from "../context/AppContext";

export default function Sidebar() {
  const { mode, setMode, newUser, existingUser } = useApp();
  const navigate = useNavigate();

  const switchMode = (m) => {
    setMode(m);
    navigate("/");
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-icon">🎬</span>
        <span className="brand-name">MovieRec</span>
      </div>

      <nav className="sidebar-nav">
        <button
          className={`nav-item ${mode === "new" ? "active" : ""}`}
          onClick={() => switchMode("new")}
        >
          <span>🆕</span> New User
        </button>
        <button
          className={`nav-item ${mode === "existing" ? "active" : ""}`}
          onClick={() => switchMode("existing")}
        >
          <span>👤</span> Existing User
        </button>
      </nav>

      <div className="sidebar-footer">
        {mode === "new" ? (
          <p>{newUser.viewedIds.length} movie(s) viewed</p>
        ) : (
          <p>{existingUser.userId ? `User #${existingUser.userId}` : "No user selected"}</p>
        )}
      </div>
    </aside>
  );
}
