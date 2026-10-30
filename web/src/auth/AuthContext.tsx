import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, DEMO, tokenStore } from "../api/client";
import type { Role, User } from "../api/types";

const RANK: Record<Role, number> = { viewer: 0, analyst: 1, admin: 2 };

interface AuthState {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  can: (role: Role) => boolean;
}

const Ctx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!DEMO && !tokenStore.get()) { setLoading(false); return; }
    api.me().then(setUser).catch(() => tokenStore.set(null)).finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const t = await api.login(email, password);
    tokenStore.set(t.access_token);
    setUser(await api.me());
  }, []);

  const logout = useCallback(() => { tokenStore.set(null); setUser(null); }, []);
  const can = useCallback((role: Role) => !!user && RANK[user.role] >= RANK[role], [user]);
  const value = useMemo(() => ({ user, loading, login, logout, can }), [user, loading, login, logout, can]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAuth outside AuthProvider");
  return v;
}
