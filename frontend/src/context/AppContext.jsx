import { createContext, useContext, useState } from "react";

const Ctx = createContext(null);

export function AppProvider({ children }) {
  const [mode, setMode] = useState("new"); // "new" | "existing"

  const [newUser, setNewUser] = useState({
    viewedIds: [],
    searchResults: [],
    recommendations: [],
  });

  const [existingUser, setExistingUser] = useState({
    userId: "",
    clickSequence: [],
    recommendations: [],
  });

  const [movieInfo, setMovieInfo] = useState({
    movie: null,
    similarMovies: [],
  });

  return (
    <Ctx.Provider value={{ mode, setMode, newUser, setNewUser, existingUser, setExistingUser, movieInfo, setMovieInfo }}>
      {children}
    </Ctx.Provider>
  );
}

export const useApp = () => useContext(Ctx);
