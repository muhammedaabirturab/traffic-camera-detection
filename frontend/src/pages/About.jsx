import { useEffect, useState } from 'react'
import { AlertOctagon, Info, Layers, SlidersHorizontal, Sparkles } from 'lucide-react'
import { api } from '../lib/api.js'
import { Card } from '../components/ui.jsx'
import { label } from '../lib/format.js'

function HelmetDiagram() {
  return (
    <svg viewBox="0 0 360 250" style={{ width: '100%', maxWidth: 380, background: '#050a13', borderRadius: 10 }}>
      <defs>
        <pattern id="g" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="#12203a" /></pattern>
      </defs>
      <rect width="360" height="250" fill="url(#g)" />
      {/* motorcycle */}
      <rect x="70" y="120" width="190" height="105" fill="rgba(0,200,255,0.08)" stroke="#00C8FF" strokeWidth="2" />
      <text x="74" y="240" fill="#00C8FF" fontSize="11" fontFamily="monospace">MOTORCYCLE</text>
      {/* rider */}
      <rect x="125" y="40" width="80" height="150" fill="none" stroke="#E1E1E1" strokeWidth="2" />
      <text x="210" y="52" fill="#E1E1E1" fontSize="11" fontFamily="monospace">PERSON</text>
      {/* head region */}
      <rect x="117" y="35" width="96" height="45" fill="rgba(255,140,0,0.12)" stroke="#FF8C00" strokeDasharray="5 3" strokeWidth="2" />
      <text x="217" y="72" fill="#FF8C00" fontSize="11" fontFamily="monospace">HEAD REGION (top 30%)</text>
      {/* helmet search */}
      <circle cx="165" cy="56" r="16" fill="none" stroke="#50DC6E" strokeWidth="2" strokeDasharray="3 3" />
      <text x="20" y="22" fill="#8ea2bf" fontSize="11" fontFamily="monospace">1 bike · 2 person on bike · 3 head · 4 helmet?</text>
      {/* pedestrian */}
      <rect x="292" y="110" width="40" height="118" fill="none" stroke="#5d7190" strokeWidth="2" strokeDasharray="4 3" />
      <text x="278" y="104" fill="#5d7190" fontSize="10" fontFamily="monospace">PEDESTRIAN</text>
      <text x="270" y="244" fill="#5d7190" fontSize="9.5" fontFamily="monospace">not associated</text>
    </svg>
  )
}

export default function About() {
  const [settings, setSettings] = useState(null)
  useEffect(() => { api.settings().then(setSettings).catch(() => {}) }, [])

  return (
    <div className="stack">
      <div className="grid g-main">
        <Card title="TrafficGuard AI" icon={Sparkles} sub="YOLO-Based Indian Traffic Violation Detection & Analysis System">
          <div className="prose">
            TrafficGuard AI analyses traffic-camera images and videos with YOLO object detection and identifies
            <b> possible</b> traffic-rule violations that can reasonably be inferred from the visual evidence, with
            reference to Indian traffic law. It deliberately separates three steps:
            <ul>
              <li><b>Object detection</b> — YOLO finds motorcycles, cars, buses, trucks, bicycles, persons, traffic lights and (with the helmet model) helmets.</li>
              <li><b>Traffic-rule inference</b> — geometric relationship analysis decides who is riding which two-wheeler and whether a helmet is on the rider’s head.</li>
              <li><b>Violation classification</b> — a configurable rule engine (<code>traffic_rules.json</code>) maps candidates to rules, applies confidence thresholds and attaches legal references.</li>
            </ul>
            <h3>What it detects</h3>
            <ul>
              <li>Rider / pillion without helmet (images and video)</li>
              <li>More than two persons on a two-wheeler (images and video)</li>
              <li>Crossing the stop line on red — <i>video only</i>, needs an operator-defined stop line and a visible signal</li>
              <li>Vehicle type counts; optional AI number-plate reading if a plate model is installed</li>
            </ul>
            <h3>What it does not claim</h3>
            Seat-belt use, phone use and lane discipline are listed in the rule database as not implemented, because they
            cannot be reliably determined from the available data. Every output is an <b>AI-detected possible violation</b>
            that <b>requires human verification</b>.
          </div>
        </Card>
        <Card title="Helmet-violation logic" icon={Layers} sub="Spatial reasoning, not independent detections">
          <HelmetDiagram />
          <ol className="prose" style={{ paddingLeft: 18, marginTop: 12 }}>
            <li>Detect the motorcycle and every person.</li>
            <li>Score each person against each bike: horizontal alignment, seated vertical layout, overlap and scale. Pedestrians beside the bike fail these checks.</li>
            <li>Take the top ~30% of each rider box as the head region.</li>
            <li>Look for a helmet box centred in that region (or a bare-head detection).</li>
            <li>Too small / cut-off riders or a missing helmet model → <i>insufficient evidence</i>, never a violation.</li>
          </ol>
        </Card>
      </div>

      <div className="grid g-2">
        <Card title="Active thresholds" icon={SlidersHorizontal} sub={settings?.note}>
          {settings ? (
            <div className="table-wrap">
              <table className="table">
                <tbody>
                  {Object.entries(settings.thresholds).map(([k, v]) => (
                    <tr key={k}><td className="small">{label(k)}</td><td className="mono small" style={{ textAlign: 'right' }}>{String(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="small muted">Unavailable (API offline)</div>}
        </Card>
        <div className="stack">
          <Card title="Confidence system" icon={Info}>
            <div className="prose">
              <ul>
                <li><span className="badge high">HIGH</span> ≥ {settings?.thresholds.conf_high ?? 0.85} — reported with the full violation name</li>
                <li><span className="badge medium">MEDIUM</span> ≥ {settings?.thresholds.conf_medium ?? 0.65} — reported as “Possible … Violation”</li>
                <li><span className="badge low">LOW</span> ≥ {settings?.thresholds.report_min_confidence ?? 0.45} — reported as “Possible … Violation”, low confidence</li>
                <li>Below that — listed as <b>insufficient visual evidence</b>, not reported</li>
              </ul>
              Video violations must also persist for the same tracked vehicle across several frames (temporal validation).
            </div>
          </Card>
          <Card title="Limitations & responsible use" icon={AlertOctagon}>
            <div className="prose">
              <ul>
                <li>Educational project — not an enforcement system and not legal advice.</li>
                <li>Accuracy depends on the installed models; see the Model page for measured metrics.</li>
                <li>Occlusion, night footage, low resolution and unusual camera angles reduce reliability.</li>
                <li>Exemptions (e.g. the Sikh-turban proviso to s.129) cannot be judged by the system.</li>
                <li>Laws and penalties change — the rule database is editable and dated.</li>
              </ul>
              Built with Ultralytics YOLO, ByteTrack, OpenCV, FastAPI, SQLite and React + Vite.
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
