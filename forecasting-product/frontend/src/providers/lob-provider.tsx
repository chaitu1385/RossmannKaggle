"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";

const LOB_STORAGE_KEY = "selected_lob";
const DEFAULT_LOB = "retail";

interface LobContextValue {
  lob: string;
  setLob: (lob: string) => void;
}

const LobContext = createContext<LobContextValue>({
  lob: DEFAULT_LOB,
  setLob: () => {},
});

export function LobProvider({ children }: { children: ReactNode }) {
  const [lob, setLobState] = useState(DEFAULT_LOB);

  // Restore from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(LOB_STORAGE_KEY);
    if (stored) {
      setLobState(stored);
    }
  }, []);

  const setLob = useCallback((value: string) => {
    setLobState(value);
    localStorage.setItem(LOB_STORAGE_KEY, value);
  }, []);

  return (
    <LobContext.Provider value={{ lob, setLob }}>
      {children}
    </LobContext.Provider>
  );
}

export function useLob() {
  return useContext(LobContext);
}
