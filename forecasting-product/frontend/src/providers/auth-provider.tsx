"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import type { User, Role } from "@/lib/types";
import { ROLE_PERMISSIONS } from "@/lib/constants";
import { api } from "@/lib/api-client";

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, role: Role) => Promise<void>;
  logout: () => void;
  hasPermission: (permission: string) => boolean;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  isAuthenticated: false,
  login: async () => {},
  logout: () => {},
  hasPermission: () => false,
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  // Restore session from localStorage on mount, checking token expiry
  useEffect(() => {
    const stored = localStorage.getItem("user");
    const token = localStorage.getItem("access_token");

    if (stored && token) {
      // Check JWT expiry
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        if (payload.exp && Date.now() >= payload.exp * 1000) {
          // Token expired — clear session
          localStorage.removeItem("user");
          localStorage.removeItem("access_token");
          return;
        }
      } catch {
        // Malformed token — clear session
        localStorage.removeItem("user");
        localStorage.removeItem("access_token");
        return;
      }

      try {
        setUser(JSON.parse(stored));
      } catch {
        localStorage.removeItem("user");
        localStorage.removeItem("access_token");
      }
    } else if (stored && !token) {
      // User data without token — stale session
      localStorage.removeItem("user");
    }
  }, []);

  const login = useCallback(async (username: string, role: Role) => {
    const res = await api.getToken(username, role);
    const userObj: User = {
      username,
      role,
      permissions: ROLE_PERMISSIONS[role],
    };
    localStorage.setItem("access_token", res.access_token);
    localStorage.setItem("user", JSON.stringify(userObj));
    setUser(userObj);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
    setUser(null);
  }, []);

  const hasPermission = useCallback(
    (permission: string) => {
      if (!user) return false;
      return user.permissions.includes(permission);
    },
    [user],
  );

  return (
    <AuthContext.Provider
      value={{ user, isAuthenticated: !!user, login, logout, hasPermission }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
