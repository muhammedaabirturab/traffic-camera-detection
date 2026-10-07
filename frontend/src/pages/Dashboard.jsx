import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  AlertTriangle, ArrowRight, BarChart3, Car, Clapperboard, Cpu, Gauge, ImageIcon, ListChecks, Radar, ScanSearch, Workflow,
} from 'lucide-react'
import { api } from '../lib/api.js'
import { Card, Empty, Kpi, Notice } from '../components/ui.jsx'
import { dateTime, pct, ruleName, secs } from '../lib/format.js'
import { useStatus } from '../lib/status.jsx'

const PIPELINE = [
  ['Input', 'Image / video upload'],
  ['Preprocessing', 'Resize, low-light CLAHE'],
  ['YOLO detection', 'Vehicles, persons, helmets'],
  ['Tracking', 'ByteTrack IDs (video)'],
  ['Relationship analysis', 'Rider ↔ bike ↔ helmet'],
  ['Rule engine', 'traffic_rules.json'],
  ['Confidence', 'Bands + evidence threshold'],
  ['Evidence', 'Annotated frames & crops'],
]

function DayBars({ data }) {
  const entries = Object.entries(data || {})
  if (!entries.length) return <Empty icon={BarChart3} title="No data yet">Run an analysis to populate the chart.</Empty>
  const max = Math.max(1, ...entries.map(([, v]) => v))
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, height: 150, paddingTop: 10 }}>
      {entries.map(([day, v]) => (
        <div key={day} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          <span className="mono small" style={{ color: '#3f4f66' }}>{v}</span>
          <div style={{ width: '100%', maxWidth: 34, height: `${(v / max) * 100}px`, minHeight: 3, borderRadius: '5px 5px 2px 2px',
                        background: v ? 'linear-gradient(180deg,#fb923c,#f97316)' : '#e2e8f0' }} />
          <span className="small muted mono" style={{ fontSize: 10.5 }}>{day.slice(5)}</span>
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [recent, setRecent] = useState([])
  const [error, setError] = useState(null)
  const { health } = useStatus()
  const nav = useNavigate()

  useEffect(() => {
    Promise.all([api.stats(), api.history()])
      .then(([s, h]) => {
        setStats(s)
        setRecent(h.items.slice(0, 6))
      })
      .catch((e) => setError(e.message))
  }, [])

  const byType = Object.entries(stats?.violations_by_type || {}).sort((a, b) => b[1] - a[1])
  const maxType = Math.max(1, ...byType.map(([, v]) => v))
  const d = health?.detectors

  return (
    <div className="stack">
      {error && <Notice>Cannot reach the API server: {error}. Start it with <code>python run.py</code>.</Notice>}
      {d && !d.helmet && (
        <Notice variant="info" icon={Cpu}>
          No helmet model is installed, so helmet checks currently return “insufficient evidence”. See{' '}
          <Link to="/model" style={{ textDecoration: 'underline' }}>Model → Installing detectors</Link>.
        </Notice>
      )}

      <div className="grid g-4">
        <Kpi icon={ListChecks} label="Analyses run" value={stats?.analyses ?? '—'} foot={stats ? `${stats.images} images · ${stats.videos} videos` : ' '} />
        <Kpi icon={Car} label="Vehicles detected" value={stats?.vehicles ?? '—'} foot="unique tracks for videos" />
        <Kpi icon={AlertTriangle} label="Possible violations" value={stats?.violations ?? '—'} alert={stats?.violations > 0}
             foot={stats ? `in ${stats.analyses_with_violations} analyses · need verification` : ' '} />
        <Kpi icon={Gauge} label="Avg processing time" value={stats?.avg_processing_time != null ? secs(stats.avg_processing_time) : '—'} foot="per analysis" />
      </div>

      <div className="grid g-main">
        <Card title="Start an analysis" icon={ScanSearch} sub="Upload traffic-camera footage">
          <div className="grid g-2">
            <button className="btn btn-primary" style={{ padding: '18px 16px' }} onClick={() => nav('/analyze')}>
              <ImageIcon size={18} /> Analyze image
            </button>
            <button className="btn btn-ghost" style={{ padding: '18px 16px' }} onClick={() => nav('/analyze?mode=video')}>
              <Clapperboard size={18} /> Analyze video
            </button>
          </div>
          <div className="small muted" style={{ marginTop: 12, lineHeight: 1.5 }}>
            The system reports <b>AI-detected possible violations</b> based on available visual evidence. Every result
            requires human verification.
          </div>
        </Card>
        <Card title="Detector status" icon={Radar} sub="Loaded from the models/ folder">
          {[
            ['Vehicle / person YOLO (COCO)', d?.vehicle],
            ['Helmet detector', d?.helmet],
            ['Rider (person-on-bike) detector', d?.rider],
            ['Number-plate detector (optional)', d?.plate],
          ].map(([n, ok]) => (
            <div key={n} className="row between" style={{ padding: '7px 0', borderBottom: '1px dashed #e3e9f2' }}>
              <span className="small">{n}</span>
              <span className={`badge ${ok == null ? 'muted' : ok ? 'ok' : 'muted'}`}>{ok == null ? 'LOADING' : ok ? 'READY' : 'NOT INSTALLED'}</span>
            </div>
          ))}
          <Link to="/model" className="row small" style={{ marginTop: 10, color: '#0891b2', fontWeight: 600, gap: 4 }}>
            Model details <ArrowRight size={13} />
          </Link>
        </Card>
      </div>

      <div className="grid g-2">
        <Card title="Possible violations by type" icon={AlertTriangle}>
          {byType.length ? (
            <div className="bar-list">
              {byType.map(([k, v]) => (
                <div className="bar-item" key={k}>
                  <div className="row between"><span>{ruleName(k)}</span><span className="mono">{v}</span></div>
                  <div className="bar-track"><div className="bar-fill warn" style={{ width: `${(v / maxType) * 100}%` }} /></div>
                </div>
              ))}
            </div>
          ) : (
            <Empty icon={AlertTriangle} title="No possible violations recorded">Results will appear here after analyses.</Empty>
          )}
        </Card>
        <Card title="Possible violations per day" icon={BarChart3} sub="Last 14 active days">
          <DayBars data={stats?.violations_by_day} />
        </Card>
      </div>

      <Card title="Recent analyses" icon={ListChecks} actions={<Link to="/history" className="btn btn-ghost btn-sm">View all</Link>} bodyClass="card-body tight">
        {recent.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th></th><th>File</th><th>Date / time</th><th>Vehicles</th><th>Possible violations</th><th>Top confidence</th></tr>
              </thead>
              <tbody>
                {recent.map((r) => (
                  <tr key={r.id} className="clickable" onClick={() => nav(`/history/${r.id}`)}>
                    <td style={{ width: 84 }}>{r.thumbnail ? <img className="thumb" src={r.thumbnail} alt="" /> : <div className="thumb" />}</td>
                    <td><b>{r.filename}</b><div className="small muted">{r.kind}{r.status !== 'completed' ? ` · ${r.status}` : ''}</div></td>
                    <td className="small">{dateTime(r.created_at)}</td>
                    <td className="mono">{r.vehicles}</td>
                    <td>{r.violations ? <span className="badge medium">{r.violations} possible</span> : <span className="badge ok">none</span>}</td>
                    <td className="mono small">{pct(r.confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title="No analyses yet">Upload an image or video on the Analyze page.</Empty>
        )}
      </Card>

      <Card title="Analysis pipeline" icon={Workflow} sub="Object detection → traffic-rule inference → violation classification">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(118px, 1fr))', gap: 10 }}>
          {PIPELINE.map(([n, dsc], i) => (
            <div key={n} style={{ background: '#f4f7fb', border: '1px solid #e3e9f2', borderRadius: 10, padding: '12px 12px', position: 'relative' }}>
              <div className="mono" style={{ fontSize: 10.5, color: '#0891b2', fontWeight: 700 }}>STAGE {String(i + 1).padStart(2, '0')}</div>
              <div style={{ fontWeight: 700, fontSize: 13, marginTop: 4 }}>{n}</div>
              <div className="small muted" style={{ marginTop: 2 }}>{dsc}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
