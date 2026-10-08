import { useRef, useState } from "react";
import { fmtDate, levelOf, pct } from "../api.js";

export function Stat({ label, value, sub, tone = "" }) {
  return (
    <div className={`stat ${tone}`}>
      <label>{label}</label>
      <strong>{value}</strong>
      {sub && <small>{sub}</small>}
    </div>
  );
}

export function ConfidenceBar({ value, level }) {
  const lv = level || levelOf(value);
  return (
    <div>
      <span className={`chip ${lv}`}>{lv.toUpperCase()} · {pct(value)}</span>
      <div className={`cbar ${lv}`}>
        <div style={{ width: pct(value) }} />
      </div>
    </div>
  );
}

const LABELS = {
  rider: "RIDER", pillion: "PILLION", motorcycle: "MOTORCYCLE", bicycle: "BICYCLE", car: "CAR", bus: "BUS",
  truck: "TRUCK", person: "PERSON", helmet: "HELMET", no_helmet: "NO HELMET",
};

export function DetectionOverlay({ src, items, showRiders = true, showPeds = false }) {
  const visible = items.filter((o) =>
    o.group === "rider" ? showRiders : o.group === "pedestrian" ? showPeds : o.minor ? showRiders : true
  );
  return (
    <div className="stage">
      <img src={src} alt="Analysed traffic frame" />
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none">
        {visible.map((o, i) => {
          const [x1, y1, x2, y2] = o.box_norm.map((v) => v * 1000);
          return (
            <g key={i}>
              <title>{`${LABELS[o.label] || o.label} · ${pct(o.confidence)}${o.detail ? " · " + o.detail : ""}`}</title>
              <rect className={`box ${o.state}`} x={x1} y={y1} width={x2 - x1} height={y2 - y1} />
            </g>
          );
        })}
      </svg>
      <Tags items={visible} />
    </div>
  );
}

// HTML labels (not SVG text) so they do not stretch with the non-uniform viewBox.
function Tags({ items }) {
  const colors = { normal: "#22d3ee", violation: "#ff5a36", insufficient: "#f5b301", low: "#9aa5b8" };
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", lineHeight: 1.2 }}>
      {items.map((o, i) => (
        <span
          key={i}
          style={{
            position: "absolute", left: `${o.box_norm[0] * 100}%`, top: `${o.box_norm[1] * 100}%`,
            transform: "translateY(-100%)", background: colors[o.state], color: "#071019", fontSize: 11, fontWeight: 700,
            padding: "2px 6px", borderRadius: "5px 5px 5px 0", whiteSpace: "nowrap",
          }}
        >
          {o.state === "violation" ? "⚠ " : ""}
          {LABELS[o.label] || o.label}
          {o.track_id != null ? (o.track_id < 100000 ? ` #${o.track_id}` : ` #R${o.track_id - 100000}`) : ""} {Math.round(o.confidence * 100)}%
        </span>
      ))}
    </div>
  );
}

export function Legend() {
  return (
    <div className="legend">
      <span><i style={{ background: "#22d3ee" }} />Normal detection</span>
      <span><i style={{ background: "#ff5a36" }} />Possible violation</span>
      <span><i style={{ background: "#f5b301" }} />Insufficient evidence</span>
      <span><i style={{ background: "#9aa5b8" }} />Low confidence</span>
    </div>
  );
}

export function ViolationCard({ v, onSeek, onZoom }) {
  const possible = v.status === "possible";
  return (
    <div className={`vcard ${possible ? "" : "insufficient"}`}>
      <div className="vhead">
        <span style={{ fontSize: 18 }}>⚠</span>
        {possible ? "Possible Traffic Violation" : "Insufficient Visual Evidence"}
      </div>
      <div className="vbody">
        <div>
          <h3 className="vtitle">{v.name}</h3>
          <dl className="kv">
            <dt>Confidence</dt>
            <dd><ConfidenceBar value={v.confidence} level={v.confidence_level} /></dd>
            <dt>Vehicle</dt>
            <dd>{v.vehicle}{v.track_id != null ? ` · ID ${v.track_id}` : ""}</dd>
            <dt>Evidence</dt>
            <dd style={{ fontWeight: 500 }}>{v.evidence}</dd>
            {v.timestamp && (<>
              <dt>Time / frame</dt>
              <dd>
                <button className="btn sm ghost" style={{ color: "#0f172a", borderColor: "#cbd5e1" }} onClick={() => onSeek?.(v.timestamp_seconds)}>
                  ▶ {v.timestamp}
                </button>{" "}
                <span className="mono muted-c">frame {v.frame_number}</span>
              </dd>
            </>)}
            <dt>Status</dt>
            <dd><span className="chip neutral">Requires Human Verification</span></dd>
            <dt>Legal reference</dt>
            <dd style={{ fontWeight: 500, fontSize: 13 }}>
              {v.legal_reference}
              {!v.legal_verified && <div className="muted-c" style={{ fontSize: 12, marginTop: 2 }}>Configured in traffic_rules.json · verify against current official text.</div>}
            </dd>
          </dl>
          <div className="disclaimer">{v.disclaimer}</div>
        </div>
        {v.evidence_image && (
          <div>
            <img className="evidence-img" src={v.evidence_image} alt="Evidence" onClick={() => onZoom?.(v.evidence_image)} />
            <div className="muted-c" style={{ fontSize: 12, marginTop: 6 }}>Evidence image (click to enlarge)</div>
          </div>
        )}
      </div>
    </div>
  );
}

export function NoViolation({ helmetModel = true }) {
  return (
    <div className="card clear-card">
      <div className="ic">✓</div>
      <div>
        <h3 style={{ margin: 0 }}>No possible violations detected</h3>
        <div className="muted-c" style={{ fontSize: 13.5, marginTop: 4 }}>
          Based on the available visual evidence and the rules currently enabled. This is not a guarantee that no violation occurred.
          {!helmetModel && " Helmet compliance was not assessed (no helmet model installed)."}
        </div>
      </div>
    </div>
  );
}

export function Intelligence({ intel }) {
  return (
    <div className="card">
      <h3>Traffic Intelligence Score</h3>
      <span className={`risk ${intel.risk_level}`}>RISK · {intel.risk_level}</span>
      <div className="score-grid">
        <div><label>Traffic objects</label><b>{intel.traffic_objects}</b></div>
        <div><label>Normal objects</label><b>{intel.normal_objects}</b></div>
        <div><label>Possible violations</label><b>{intel.possible_violations}</b></div>
        <div><label>Avg. confidence</label><b>{pct(intel.average_confidence)}</b></div>
      </div>
      <div className="muted-c" style={{ fontSize: 12.5 }}>
        Risk index {intel.risk_index}/100 = Σ violation confidence ÷ vehicles. {intel.note}
      </div>
    </div>
  );
}

export function Timeline({ data }) {
  const max = Math.max(1, ...data.map((d) => d.violations));
  const W = 600, H = 120, bw = W / Math.max(1, data.length);
  return (
    <div className="card">
      <h3>Violations over time</h3>
      <svg className="chart" viewBox={`0 0 ${W} ${H + 18}`} preserveAspectRatio="none">
        {data.map((d, i) => {
          const h = (d.violations / max) * H;
          return (
            <g key={i}>
              <title>{`t=${d.second}s · ${d.violations} violation(s)`}</title>
              <rect className={`bar ${d.violations ? "" : "zero"}`} x={i * bw + 1} width={Math.max(1, bw - 2)} y={d.violations ? H - h : H - 2} height={d.violations ? h : 2} rx="2" />
            </g>
          );
        })}
        <text x="0" y={H + 14}>0s</text>
        <text x={W} y={H + 14} textAnchor="end">{data.length - 1}s</text>
      </svg>
    </div>
  );
}

export function Lightbox({ src, onClose }) {
  if (!src) return null;
  return <div className="lightbox" onClick={onClose}><img src={src} alt="Evidence enlarged" /></div>;
}

function Meta({ r }) {
  return (
    <div className="row mono muted-c" style={{ color: "var(--muted)", fontSize: 12.5, marginBottom: 16 }}>
      <span>{r.filename}</span><span>·</span><span>{fmtDate(r.created_at)}</span><span>·</span>
      <span>processed in {r.processing_time}s</span><span>·</span><span>ID {r.id}</span>
    </div>
  );
}

function Notes({ notes }) {
  if (!notes?.length) return null;
  return <div className="info-box mt">{notes.map((n, i) => <div key={i}>ℹ {n}</div>)}</div>;
}

export function ImageResult({ r, helmetModel }) {
  const [riders, setRiders] = useState(r.summary.motorcycles <= 3); // dense scenes start uncluttered
  const [peds, setPeds] = useState(false);
  const [zoom, setZoom] = useState(null);
  const s = r.summary;
  return (
    <>
      <Meta r={r} />
      <div className="grid g2">
        <div className="feed">
          <div className="feed-head"><span>Original image</span><span className="rec">CAM</span></div>
          <div className="stage"><img src={r.media.original} alt="Original" /></div>
        </div>
        <div className="feed">
          <div className="feed-head">
            <span>AI analysis</span>
            <span className="row" style={{ gap: 14 }}>
              <label className="toggle"><input type="checkbox" checked={riders} onChange={(e) => setRiders(e.target.checked)} /> riders &amp; helmets</label>
              <label className="toggle"><input type="checkbox" checked={peds} onChange={(e) => setPeds(e.target.checked)} /> other persons</label>
            </span>
          </div>
          <DetectionOverlay src={r.media.original} items={r.overlay} showRiders={riders} showPeds={peds} />
          <Legend />
        </div>
      </div>

      <h3 className="eyebrow mt" style={{ marginBottom: 12 }}>Detection summary</h3>
      <div className="grid g4">
        <Stat label="Vehicles detected" value={s.vehicles_detected} />
        <Stat label="Motorcycles" value={s.motorcycles} sub={`${s.riders} rider(s) linked`} />
        <Stat label="Cars / buses / trucks" value={s.cars + s.buses + s.trucks} sub={`${s.cars} car · ${s.buses} bus · ${s.trucks} truck`} />
        <Stat label="Possible violations" value={s.possible_violations} tone={s.possible_violations ? "warn" : "ok"} sub={s.insufficient_evidence ? `${s.insufficient_evidence} with insufficient evidence` : "none flagged"} />
      </div>

      <div className="grid mt" style={{ gridTemplateColumns: "minmax(0,1.8fr) minmax(0,1fr)", alignItems: "start" }}>
        <div className="grid">
          {r.violations.length === 0 && <NoViolation helmetModel={helmetModel} />}
          {r.violations.map((v, i) => <ViolationCard key={i} v={v} onZoom={setZoom} />)}
        </div>
        <Intelligence intel={r.intelligence} />
      </div>
      <Notes notes={r.notes} />
      <Lightbox src={zoom} onClose={() => setZoom(null)} />
    </>
  );
}

export function VideoResult({ r, helmetModel }) {
  const vref = useRef(null);
  const [zoom, setZoom] = useState(null);
  const s = r.summary, v = r.video;
  const seek = (t) => { if (vref.current) { vref.current.currentTime = Math.max(0, t - 0.5); vref.current.play(); } };
  return (
    <>
      <Meta r={r} />
      <div className="grid" style={{ gridTemplateColumns: "minmax(0,1.6fr) minmax(0,1fr)", alignItems: "start" }}>
        <div className="feed">
          <div className="feed-head"><span>Processed video · YOLO + ByteTrack</span><span className="rec">ANALYSED</span></div>
          <div className="stage"><video ref={vref} src={r.media.processed_video} controls playsInline /></div>
          <Legend />
        </div>
        <Intelligence intel={r.intelligence} />
      </div>
      <h3 className="eyebrow mt" style={{ marginBottom: 12 }}>Video summary</h3>
      <div className="grid g4">
        <Stat label="Unique vehicles" value={s.unique_vehicles} sub={`${s.vehicles_detected} detections in total`} />
        <Stat label="Possible violations" value={s.possible_violations} tone={s.possible_violations ? "warn" : "ok"} sub={s.insufficient_evidence ? `${s.insufficient_evidence} short-lived candidates discarded` : "none flagged"} />
        <Stat label="Frames analysed" value={v.frames_analyzed} sub={`of ${v.frames_total} · every ${v.frame_skip} frame(s)`} />
        <Stat label="Duration" value={`${v.duration_seconds}s`} sub={`${v.fps} fps · ${v.width}×${v.height}`} />
      </div>
      <div className="mt"><Timeline data={r.timeline} /></div>
      <div className="grid mt">
        {r.violations.length === 0 && <NoViolation helmetModel={helmetModel} />}
        {r.violations.map((x, i) => <ViolationCard key={i} v={x} onSeek={seek} onZoom={setZoom} />)}
      </div>
      <Notes notes={r.notes} />
      <Lightbox src={zoom} onClose={() => setZoom(null)} />
    </>
  );
}

export default function ResultView({ result }) {
  const helmetModel = !(result.notes || []).some((n) => n.includes("No helmet model"));
  return result.kind === "video"
    ? <VideoResult r={result} helmetModel={helmetModel} />
    : <ImageResult r={result} helmetModel={helmetModel} />;
}
