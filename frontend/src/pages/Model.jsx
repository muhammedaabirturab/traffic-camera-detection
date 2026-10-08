import { useEffect, useState } from "react";
import { api, pct } from "../api.js";
import { Stat } from "../components/Result.jsx";

const num = (v) => (v == null ? "–" : v.toFixed(3));

function MetricsSection({ title, x, figDir, missing }) {
  const t = x?.training, d = x?.dataset;
  return (
    <>
      <h3 className="eyebrow mt" style={{ marginBottom: 12 }}>{title}{x ? ` — held-out ${x.evaluated_on} split metrics` : ""}</h3>
      {!x ? (
        <div className="card">
          <b>Model not trained / metrics unavailable.</b>
          <div className="muted-c" style={{ marginTop: 6 }}>{missing}</div>
        </div>
      ) : (
        <>
          <div className="grid g4">
            <Stat label="Precision" value={num(x.precision)} />
            <Stat label="Recall" value={num(x.recall)} />
            <Stat label="mAP@50" value={num(x.map50)} tone="ok" />
            <Stat label="mAP@50-95" value={num(x.map50_95)} tone="ok" />
          </div>
          <div className="grid g2 mt">
            <div className="card">
              <h3>Training information</h3>
              <table><tbody>
                <tr><td>F1 score</td><td>{num(x.f1)}</td></tr>
                <tr><td>Epochs</td><td>{t?.epochs_completed ?? "–"} completed (of {t?.epochs_requested ?? "–"} requested)</td></tr>
                <tr><td>Image size</td><td>{t?.image_size}px</td></tr>
                <tr><td>Batch size</td><td>{t?.batch_size}</td></tr>
                <tr><td>Hardware</td><td>{t?.hardware}</td></tr>
                <tr><td>Base checkpoint</td><td>{t?.base_checkpoint}</td></tr>
                <tr><td>Trained</td><td>{t?.trained_at}</td></tr>
              </tbody></table>
            </div>
            <div className="card">
              <h3>Classes &amp; dataset</h3>
              <table><tbody>
                <tr><td>Classes</td><td>{x.classes.join(", ")}</td></tr>
                {x.per_class && Object.entries(x.per_class).map(([c, v]) => (
                  <tr key={c}><td>AP@50 · {c}</td><td>{num(v.ap50)}</td></tr>
                ))}
                {d && <>
                  <tr><td>Images (train / val / test)</td><td>{d.images.train} / {d.images.val} / {d.images.test}</td></tr>
                  <tr><td>Boxes (train / val / test)</td><td>{d.boxes.train} / {d.boxes.val} / {d.boxes.test}</td></tr>
                </>}
              </tbody></table>
            </div>
          </div>
          <div className="grid g2 mt">
            {["results.png", "BoxPR_curve.png"].map((f) => (
              <div key={f} className="card"><h3>{f === "results.png" ? "Training curves" : "Precision–Recall curve"}</h3>
                <img src={`/figures/${figDir}/${f}`} alt={f} style={{ width: "100%" }} onError={(e) => (e.currentTarget.style.display = "none")} />
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}

export default function Model() {
  const [m, setM] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.modelInfo().then(setM).catch((e) => setErr(e.message)); }, []);
  if (err) return <div className="error-box">⚠ {err}</div>;
  if (!m) return <div className="skeleton" style={{ height: 300 }} />;

  const det = m.models;
  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Model card</div>
        <h1>YOLO detection models</h1>
        <p>All figures on this page are read from files written by <span className="mono">scripts/train.py</span>. Nothing is hard-coded.</p>
      </div>

      <div className="grid g2">
        <div className="card">
          <h3>Overview</h3>
          <table><tbody>
            <tr><td>Framework</td><td><b>{m.framework}</b>{m.metrics ? ` (${m.metrics.model})` : ""}</td></tr>
            <tr><td>Task</td><td>{m.task}</td></tr>
            <tr><td>Input</td><td>{m.input}</td></tr>
            <tr><td>Datasets</td><td>{m.dataset}</td></tr>
            <tr><td>Device</td><td>{m.config.device}</td></tr>
          </tbody></table>
        </div>
        <div className="card">
          <h3>Loaded detectors</h3>
          {!det && <div className="muted-c">Backend has no detector loaded.</div>}
          {det && [["rider", "Rider detector (fine-tuned)"], ["coco", "COCO vehicle/person detector"], ["helmet", "Helmet detector (fine-tuned)"]].map(([k, name]) => (
            <div key={k} style={{ marginBottom: 12 }}>
              <div className="row"><b>{name}</b> {det[k] ? <span className="chip high">LOADED</span> : <span className="chip neutral">{k === "helmet" ? "NOT INSTALLED" : "MISSING"}</span>}</div>
              <div className="muted-c mono" style={{ fontSize: 12.5, marginTop: 4 }}>
                {det[k] ? `${det[k].path} · classes: ${det[k].classes.slice(0, 8).join(", ")}${det[k].classes.length > 8 ? ` … (${det[k].classes.length})` : ""}` : det.problems?.[k] || ""}
              </div>
            </div>
          ))}
        </div>
      </div>

      <MetricsSection title="Rider detector" x={m.metrics} figDir="rider_detector"
        missing="Run python scripts/train.py to train the rider detector; its real metrics will appear here automatically." />
      <MetricsSection title="Helmet detector" x={m.helmet_metrics} figDir="helmet_detector"
        missing="No helmet model trained yet. See app/models/README.md; its real metrics will appear here automatically." />

      <div className="card mt">
        <h3>Runtime configuration</h3>
        <table><tbody>
          <tr><td>Confidence threshold</td><td>{m.config.confidence_threshold}</td></tr>
          <tr><td>IoU threshold (NMS)</td><td>{m.config.iou_threshold}</td></tr>
          <tr><td>Confidence bands</td><td>High ≥ {pct(m.config.confidence_bands.high)} · Medium ≥ {pct(m.config.confidence_bands.medium)} · otherwise Low</td></tr>
          <tr><td>Min. confidence for a violation</td><td>{pct(m.config.min_violation_confidence)} (below → “insufficient evidence”)</td></tr>
          <tr><td>Video</td><td>every {m.config.video_frame_skip} frame(s) · ≥ {m.config.video_min_violation_frames} frames to confirm</td></tr>
        </tbody></table>
      </div>
    </>
  );
}
