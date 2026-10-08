import { useEffect, useState } from "react";
import { api } from "../api.js";

const TONE = { active: "high", experimental: "medium", info: "info", needs_model: "medium", not_implemented: "neutral", disabled: "neutral" };

export default function Rules() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.rules().then(setData).catch((e) => setErr(e.message)); }, []);
  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Rule database</div>
        <h1>Traffic rules</h1>
        <p>
          Loaded from <span className="mono">app/rules/traffic_rules.json</span>{data ? ` (v${data.version}, reviewed ${data.last_reviewed})` : ""}. Edit that file to change rules,
          thresholds or legal references; no code changes needed.
        </p>
      </div>
      <div className="info-box" style={{ marginBottom: 18 }}>
        ⚖ {data?.disclaimer || "Educational project."} Penalty amounts are intentionally not stored — laws and fines change.
      </div>
      {err && <div className="error-box">⚠ {err}</div>}
      <div className="card tw">
        {!data && !err && <div className="skeleton" />}
        {data && (
          <table>
            <thead><tr><th>Violation</th><th>Description</th><th>Detection method</th><th>Legal reference</th><th>Status</th></tr></thead>
            <tbody>
              {data.rules.map((r) => (
                <tr key={r.violation_id}>
                  <td style={{ minWidth: 170 }}><b>{r.violation_name}</b><div className="mono muted-c" style={{ fontSize: 11.5 }}>{r.violation_id}</div></td>
                  <td style={{ minWidth: 220 }}>{r.description}</td>
                  <td style={{ minWidth: 280, fontSize: 13 }}>{r.detection_logic}</td>
                  <td style={{ minWidth: 220, fontSize: 13 }}>
                    {r.legal_reference}
                    {r.penalty_note && r.legal_reference !== "N/A" && <div className="muted-c" style={{ fontSize: 12, marginTop: 4 }}>{r.penalty_note}</div>}
                    {r.legal_reference !== "N/A" && !r.legal_verified && <div style={{ marginTop: 4 }}><span className="chip medium">VERIFY BEFORE RELYING</span></div>}
                  </td>
                  <td><span className={`chip ${TONE[r.status.code]}`}>{r.status.label}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
