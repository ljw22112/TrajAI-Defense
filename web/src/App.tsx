import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { DEMO } from "./api/client";
import type { Role } from "./api/types";
import { useAuth } from "./auth/AuthContext";
import { Spinner } from "./components/ui";
import { AdminPage } from "./pages/AdminPage";
import { AssetsPage } from "./pages/AssetsPage";
import { DashboardPage } from "./pages/DashboardPage";
import { LoginPage } from "./pages/LoginPage";
import { ModelsPage } from "./pages/ModelsPage";
import { ScenarioDetailPage } from "./pages/ScenarioDetailPage";
import { ScenarioNewPage } from "./pages/ScenarioNewPage";
import { ScenariosPage } from "./pages/ScenariosPage";
import { WeatherPage } from "./pages/WeatherPage";

function Guard({ role, children }: { role?: Role; children: JSX.Element }) {
  const { user, loading, can } = useAuth();
  const loc = useLocation();
  if (loading) return <Spinner />;
  if (!user) return <Navigate to="/login" state={{ from: loc.pathname }} replace />;
  if (role && !can(role)) return <div className="error">이 화면은 {role} 이상 권한이 필요합니다.</div>;
  return children;
}

export function App() {
  const { user, logout, can } = useAuth();
  if (!user && !DEMO) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Guard><span /></Guard>} />
      </Routes>
    );
  }
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">TrajAI<i>-</i>Defense<small>OPERATIONS CONSOLE</small></div>
        <nav className="nav" aria-label="주 메뉴">
          <span className="sec">운용</span>
          <NavLink to="/" end>대시보드</NavLink>
          <NavLink to="/scenarios">시나리오</NavLink>
          {can("analyst") && <NavLink to="/scenarios/new">새 시나리오</NavLink>}
          <span className="sec">자료</span>
          <NavLink to="/weather">기상 프로파일</NavLink>
          <NavLink to="/assets">보호 자산</NavLink>
          <NavLink to="/models">모델 레지스트리</NavLink>
          {can("admin") && <><span className="sec">관리</span><NavLink to="/admin">사용자 · 감사 로그</NavLink></>}
        </nav>
        <div className="who">
          {DEMO && <span className="demo-flag">STATIC DEMO</span>}
          <span>{user?.full_name || user?.email}</span>
          <span className="role">{user?.role.toUpperCase()}</span>
          {!DEMO && <button className="btn ghost sm" onClick={logout} style={{ color: "inherit" }}>로그아웃</button>}
          <a href={DEMO ? "../" : "https://ljw22112.github.io/TrajAI-Defense/"} style={{ color: "inherit", fontSize: 12.5 }}>기술 문서 ↗</a>
        </div>
      </aside>
      <main>
        <Routes>
          <Route path="/login" element={<Navigate to="/" replace />} />
          <Route path="/" element={<Guard><DashboardPage /></Guard>} />
          <Route path="/scenarios" element={<Guard><ScenariosPage /></Guard>} />
          <Route path="/scenarios/new" element={<Guard role="analyst"><ScenarioNewPage /></Guard>} />
          <Route path="/scenarios/:id" element={<Guard><ScenarioDetailPage /></Guard>} />
          <Route path="/weather" element={<Guard><WeatherPage /></Guard>} />
          <Route path="/assets" element={<Guard><AssetsPage /></Guard>} />
          <Route path="/models" element={<Guard><ModelsPage /></Guard>} />
          <Route path="/admin" element={<Guard role="admin"><AdminPage /></Guard>} />
          <Route path="*" element={<div className="empty">페이지를 찾을 수 없습니다.</div>} />
        </Routes>
      </main>
    </div>
  );
}
