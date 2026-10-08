import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, fmtDate, pct } from "../api.js";
import { Stat } from "../components/Result.jsx";

export const TYPE_NAMES = {
  triple_riding: "More than two on two-wheeler",
  no_helmet_rider: "Rider without helmet",
  no_helmet_pillion: "Pillion without helmet",
  red_light_jump: "Stop line on red",
};

export default function History() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const nav = useNavigate();
  const load = () => api.history().then(setData).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);

  const del = async (e, id) => {
    e.stopPropagation();
    if (!confirm("Delete this analysis and its saved images?")) return;
    try { await api.deleteAnalysis(id); load(); } catch (x) { setErr(x.message); }
  };

  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Records</div>
        <h1>Detection history</h1>
        <p>Every analysis is stored locally in SQLite. Open one to see its annotated frame, evidence and summary again.</p>
      </div>
      {err && <div className="error-box">⚠ {err}</div>}
      {data && (
        <div className="grid g4" style={{ marginBottom: 18 }}>
          <Stat label="Analyses" value={data.stats.analyses} />
          <Stat label="Possible violations" value={data.stats.violations} tone="warn" />
          <Stat label="Vehicles seen" value={data.stats.vehicles} />
          <Stat label="Avg. confidence" value={pct(data.stats.average_confidence)} tone="ok" />
        </div>
      )}
      <div className="card tw">
        {!data && !err && <div className="skeleton" />}
        {data && data.items.length === 0 && (
          <div className="toast-empty" style={{ color: "#64748b" }}>No analyses yet. Go to <b>Analyze</b> and upload a traffic image.</div>
        )}
        {data && data.items.length > 0 && (
          <table>
            <thead>
              <tr><th>Date / time</th><th>File</th><th>Type</th><th>Vehicles</th><th>Violations</th><th>Violation types</th><th>Confidence</th><th>Time</th><th /></tr>
            </thead>
            <tbody>
              {data.items.map((h) => (
                <tr key={h.id} className="click" onClick={() => nav(`/analysis/${h.id}`)}>
                  <td>{fmtDate(h.created_at)}</td>
                  <td style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis" }}>{h.filename}</td>
                  <td><span className="chip info">{h.kind}</span></td>
                  <td>{h.vehicles}</td>
                  <td>{h.violations ? <span className="chip low">{h.violations}</span> : <span className="chip high">0</span>}</td>
                  <td>{h.violation_types.map((t) => TYPE_NAMES[t] || t).join(", ") || "—"}</td>
                  <td>{pct(h.confidence)}</td>
                  <td>{h.processing_time}s</td>
                  <td><button className="btn sm danger" onClick={(e) => del(e, h.id)}>Delete</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
