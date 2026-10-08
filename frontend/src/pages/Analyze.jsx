import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import ResultView from "../components/Result.jsx";

const UploadIcon = () => (
  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 16V4m0 0L8 8m4-4 4 4" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
  </svg>
);

function DropZone({ accept, hint, onFile, disabled }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  return (
    <div
      className={`drop ${over ? "over" : ""}`}
      onClick={() => !disabled && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f && !disabled) onFile(f); }}
    >
      <UploadIcon />
      <h3>Drop a file here or click to browse</h3>
      <p>{hint}</p>
      <input ref={input} type="file" accept={accept} hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
    </div>
  );
}

export default function Analyze() {
  const [mode, setMode] = useState("image");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [redLight, setRedLight] = useState({ stop_line: "", red_from: "", red_to: "", direction: "down" });
  const poll = useRef(null);

  useEffect(() => () => clearTimeout(poll.current), []);

  async function runImage(file) {
    setBusy(true); setError(""); setResult(null);
    try { setResult(await api.analyzeImage(file)); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function runVideo(file) {
    setBusy(true); setError(""); setResult(null); setProgress(0);
    try {
      const opts = redLight.stop_line !== "" ? redLight : {};
      const { job_id } = await api.analyzeVideo(file, opts);
      const tick = async () => {
        try {
          const j = await api.job(job_id);
          setProgress(j.progress);
          if (j.status === "done") { setResult(await api.analysis(j.result_id)); setBusy(false); }
          else if (j.status === "error") { setError(j.error); setBusy(false); }
          else poll.current = setTimeout(tick, 900);
        } catch (e) { setError(e.message); setBusy(false); }
      };
      tick();
    } catch (e) { setError(e.message); setBusy(false); }
  }

  const switchMode = (m) => { setMode(m); setResult(null); setError(""); };

  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Analysis workspace</div>
        <h1>Analyze traffic footage</h1>
        <p>Upload a traffic-camera image or video. YOLO detects vehicles and riders, a relationship layer links people to two-wheelers, and the rule engine reports <b>possible</b> violations with evidence.</p>
      </div>

      <div className="tabs">
        <button className={mode === "image" ? "active" : ""} onClick={() => switchMode("image")}>Image</button>
        <button className={mode === "video" ? "active" : ""} onClick={() => switchMode("video")}>Video</button>
      </div>

      {mode === "image" ? (
        <DropZone accept=".jpg,.jpeg,.png,.webp" hint="JPG, JPEG, PNG or WebP · up to 15 MB" onFile={runImage} disabled={busy} />
      ) : (
        <>
          <DropZone accept=".mp4,.avi,.mov" hint="MP4, AVI or MOV · up to 200 MB · processed in the background with ByteTrack tracking" onFile={runVideo} disabled={busy} />
          <details className="adv">
            <summary>Experimental: red-light check (operator-supplied signal context)</summary>
            <p style={{ color: "var(--muted)", fontSize: 13, lineHeight: 1.5 }}>
              The model cannot see the signal state, so this check only runs if <b>you</b> provide the stop-line position and the red-phase window.
              A tracked vehicle crossing the line during that window is flagged as a possible violation.
            </p>
            <div className="grid g4">
              <label className="field">Stop line (0.05–0.95 of height)
                <input type="number" step="0.01" min="0.05" max="0.95" value={redLight.stop_line} onChange={(e) => setRedLight({ ...redLight, stop_line: e.target.value })} placeholder="e.g. 0.7" />
              </label>
              <label className="field">Red from (s)
                <input type="number" step="0.1" min="0" value={redLight.red_from} onChange={(e) => setRedLight({ ...redLight, red_from: e.target.value })} />
              </label>
              <label className="field">Red until (s)
                <input type="number" step="0.1" min="0" value={redLight.red_to} onChange={(e) => setRedLight({ ...redLight, red_to: e.target.value })} />
              </label>
              <label className="field">Direction of travel
                <select value={redLight.direction} onChange={(e) => setRedLight({ ...redLight, direction: e.target.value })}>
                  <option value="down">Downwards (towards camera)</option>
                  <option value="up">Upwards (away from camera)</option>
                </select>
              </label>
            </div>
          </details>
        </>
      )}

      {busy && (
        <div className="panel mt">
          <div className="row" style={{ marginBottom: 10 }}>
            <span className="eyebrow">{mode === "video" ? `Analysing video… ${Math.round(progress * 100)}%` : "Running YOLO inference…"}</span>
          </div>
          <div className="progress"><div style={{ width: mode === "video" ? `${Math.max(4, progress * 100)}%` : "60%" }} /></div>
        </div>
      )}
      {error && <div className="error-box">⚠ {error}</div>}
      {result && <div className="mt"><ResultView result={result} /></div>}
    </>
  );
}
