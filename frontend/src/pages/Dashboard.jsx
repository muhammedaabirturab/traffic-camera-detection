import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, fmtDate, pct } from "../api.js";
import { Stat } from "../components/Result.jsx";
import { TYPE_NAMES } from "./History.jsx";

const PIPE = ["Image / Video", "Preprocessing", "YOLO detection", "Object identification", "Tracking (video)", "Relationship analysis", "Rule engine", "Violation classification", "Confidence evaluation", "Evidence", "Dashboard"];

export default function Dashboard() {
  const [hist, setHist] = useState(null);
  const [health, setHealth] = useState(null);
  const [err, setErr] = useState("");
  const nav = useNavigate();
  useEffect(() => {
    api.history(6).then(setHist).catch((e) => setErr(e.message));
    api.health().then(setHealth).catch(() => {});
  }, []);
  const st = hist?.stats;

  return (
    <>
      <section className="hero">
        <div className="eyebrow">Smart city · Intelligent transportation</div>
        <h1>AI-powered <span>Indian traffic monitoring</span> with YOLO</h1>
        <p>
          TrafficGuard AI analyses traffic-camera images and video, links riders to two-wheelers, and reports
          <b> AI-detected possible violations</b> with evidence, confidence and a configurable legal reference. Every result requires human verification.
        </p>
        <div className="row">
          <button className="btn" onClick={() => nav("/analyze")}>Analyze traffic footage →</button>
          <Link to="/rules" className="btn ghost">View traffic rules</Link>
        </div>
      </section>

      {err && <div className="error-box">⚠ {err}</div>}

      <div className="grid g4 mt">
        <Stat label="Analyses run" value={st ? st.analyses : "–"} />
        <Stat label="Possible violations" value={st ? st.violations : "–"} tone="warn" sub="all require human review" />
        <Stat label="Vehicles seen" value={st ? st.vehicles : "–"} />
        <Stat label="Avg. confidence" value={st ? pct(st.average_confidence) : "–"} tone="ok" />
      </div>

      <div className="grid g2 mt" style={{ alignItems: "start" }}>
        <div className="panel">
          <h3 className="eyebrow">System status</h3>
          <table style={{ color: "var(--text)" }}>
            <tbody>
              {[
                ["Backend", health ? "online" : "offline", !!health],
                ["Detection models", health?.models_loaded ? "loaded" : "not loaded", !!health?.models_loaded],
                ["Compute device", health?.device ?? "–", !!health?.device],
                ["Helmet model", health?.helmet_model ? "installed" : "not installed (optional)", !!health?.helmet_model],
              ].map(([k, v, ok]) => (
                <tr key={k}>
                  <td style={{ borderColor: "var(--line)", color: "var(--muted)" }}>{k}</td>
                  <td style={{ borderColor: "var(--line)" }}><span className={`dot ${ok ? "ok" : "warn"}`} style={{ display: "inline-block", marginRight: 8 }} />{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h3 className="eyebrow">Analysis pipeline</h3>
          <div className="pipeline">
            {PIPE.map((p, i) => (
              <span key={p} className="row" style={{ gap: 6 }}>
                <span className={`step ${["YOLO detection", "Rule engine", "Relationship analysis"].includes(p) ? "hl" : ""}`}>{p}</span>
                {i < PIPE.length - 1 && <span className="arrow">›</span>}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="card mt tw">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
          <h3 style={{ margin: 0 }}>Recent analyses</h3>
          <Link to="/history" style={{ color: "#0369a1", fontSize: 13 }}>View all →</Link>
        </div>
        {hist && hist.items.length === 0 && <div className="muted-c" style={{ padding: "18px 0" }}>Nothing analysed yet — upload an image to get started.</div>}
        {hist && hist.items.length > 0 && (
          <table>
            <thead><tr><th>When</th><th>File</th><th>Vehicles</th><th>Violations</th><th>Types</th></tr></thead>
            <tbody>
              {hist.items.map((h) => (
                <tr key={h.id} className="click" onClick={() => nav(`/analysis/${h.id}`)}>
                  <td>{fmtDate(h.created_at)}</td><td>{h.filename}</td><td>{h.vehicles}</td>
                  <td>{h.violations ? <span className="chip low">{h.violations}</span> : <span className="chip high">0</span>}</td>
                  <td>{h.violation_types.map((t) => TYPE_NAMES[t] || t).join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
