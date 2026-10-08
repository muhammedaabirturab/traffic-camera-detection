import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { api } from "../api.js";

const NAV = [
  ["/", "Dashboard"],
  ["/analyze", "Analyze"],
  ["/history", "Detection History"],
  ["/model", "Model"],
  ["/rules", "Traffic Rules"],
  ["/about", "About"],
];

export function Logo() {
  return (
    <svg width="34" height="34" viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="14" fill="#0e1a35" stroke="#1f2c4a" />
      <path d="M32 9 13 17v13c0 11 7.5 20 19 25 11.500-5 19-14 19-25V17z" fill="none" stroke="#22d3ee" strokeWidth="3.5" strokeLinejoin="round" />
      <circle cx="32" cy="30" r="6.500" fill="#22d3ee" />
      <circle cx="32" cy="30" r="2.500" fill="#0e1a35" />
    </svg>
  );
}

export default function Layout({ children }) {
  const [health, setHealth] = useState(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      api.health().then((h) => alive && (setHealth(h), setOffline(false))).catch(() => alive && setOffline(true));
    poll();
    const t = setInterval(poll, 15000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const state = offline ? "bad" : health?.models_loaded ? "ok" : "warn";
  const label = offline ? "Backend offline" : health?.models_loaded ? `Models ready · ${health.device}` : health ? "Models not loaded" : "Connecting…";

  return (
    <div className="shell">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <Logo />
          <span>
            <b>TRAFFICGUARD AI</b>
            <small>Intelligent Traffic Violation Detection System</small>
          </span>
        </NavLink>
        <nav className="nav">
          {NAV.map(([to, name]) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>
              {name}
            </NavLink>
          ))}
        </nav>
        <span className="spacer" />
        <span className="status-pill" title={health?.message || ""}>
          <span className={`dot ${state}`} /> {label}
        </span>
      </header>
      {offline && <div className="banner">The analysis backend is not reachable. Start it with <code>python -m app.main</code>, then reload.</div>}
      {health && !health.models_loaded && !offline && <div className="banner">{health.message || "Detection models are not loaded."} See app/models/README.md.</div>}
      <main className="main">{children}</main>
      <footer className="footer">
        TrafficGuard AI · Educational project · Results are AI-detected possible violations and require human verification.
      </footer>
    </div>
  );
}
