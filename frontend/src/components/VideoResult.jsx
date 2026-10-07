import { useRef } from 'react'
import { Activity, Film, Settings2 } from 'lucide-react'
import { Card, Notice, useLightbox } from './ui.jsx'
import { Disclaimer, Observations, SummaryStrip, ViolationList } from './ResultParts.jsx'
import { label, secs, timecode } from '../lib/format.js'

const SIGNAL_COLORS = { red: '#ef4444', yellow: '#f59e0b', green: '#22c55e', unknown: '#cbd5e1' }

function Timeline({ points, violations, duration, onSeek, showSignal }) {
  if (!points?.length) return null
  const W = 1000, H = 120, pad = 6
  const tMax = duration || points[points.length - 1].t || 1
  const yMax = Math.max(1, ...points.map((p) => Math.max(p.vehicles, p.riders)))
  const x = (t) => pad + (t / tMax) * (W - 2 * pad)
  const y = (v) => H - 22 - (v / yMax) * (H - 40)
  const path = (key) => points.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ')
  return (
    <div className="timeline">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
           onClick={(e) => {
             const r = e.currentTarget.getBoundingClientRect()
             onSeek?.(((e.clientX - r.left) / r.width) * tMax)
           }} style={{ cursor: 'crosshair' }}>
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={pad} x2={W - pad} y1={y(yMax * f)} y2={y(yMax * f)} stroke="#e8eef6" />
        ))}
        {points.filter((p) => p.candidates > 0).map((p, i) => (
          <rect key={i} x={x(p.t) - 1} y={H - 20} width={2.5} height={8} fill="#fdba74" />
        ))}
        {showSignal && points.map((p, i) => (
          <rect key={`s${i}`} x={x(p.t)} y={H - 8} width={Math.max(1, (W - 2 * pad) / points.length + 0.5)} height={6} fill={SIGNAL_COLORS[p.signal] || '#cbd5e1'} />
        ))}
        <path d={path('vehicles')} fill="none" stroke="#0891b2" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <path d={path('riders')} fill="none" stroke="#a78bfa" strokeWidth="1.6" strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />
        {violations.map((v) => (
          <g key={v.id}>
            <line x1={x(v.timestamp)} x2={x(v.timestamp)} y1={8} y2={H - 22} stroke="#ef4444" strokeWidth="2" vectorEffect="non-scaling-stroke" />
            <circle cx={x(v.timestamp)} cy={10} r={5} fill="#ef4444" />
          </g>
        ))}
      </svg>
      <div className="row wrap small muted" style={{ gap: 16, marginTop: 6 }}>
        <span><i style={{ display: 'inline-block', width: 14, height: 2, background: '#0891b2', verticalAlign: 'middle' }} /> vehicles in frame</span>
        <span><i style={{ display: 'inline-block', width: 14, height: 2, background: '#a78bfa', verticalAlign: 'middle' }} /> riders</span>
        <span><i style={{ display: 'inline-block', width: 4, height: 9, background: '#fdba74', verticalAlign: 'middle' }} /> per-frame candidate</span>
        <span><i style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 9, background: '#ef4444', verticalAlign: 'middle' }} /> confirmed possible violation</span>
        {showSignal && <span>bottom band: detected signal state</span>}
        <span style={{ marginLeft: 'auto' }}>click to seek</span>
      </div>
    </div>
  )
}

export default function VideoResult({ result }) {
  const videoRef = useRef(null)
  const [zoom, lightbox] = useLightbox()
  const { video, media, summary, violations } = result
  const seek = (t) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, t - 0.5)
      videoRef.current.play().catch(() => {})
      videoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }
  const cells = [
    { l: 'Unique vehicles', v: summary.unique_vehicles },
    ...Object.entries(summary.unique_by_class || {})
      .filter(([k]) => k !== 'person')
      .map(([k, v]) => ({ l: `${label(k)}s`, v })),
    { l: 'Max in one frame', v: summary.max_vehicles_in_frame },
    { l: 'Possible violations', v: violations.length, alert: violations.length > 0 },
    { l: 'Frames analysed', v: video.analysed_frames },
    { l: 'Processing time', v: secs(result.processing_time) },
  ]

  return (
    <div className="stack fade-in">
      {video.truncated && (
        <Notice variant="info">Only the first {video.max_seconds}s of the video were analysed (configurable via TG_VIDEO_MAX_SECONDS).</Notice>
      )}
      <div className="grid g-main">
        <div className="viewport">
          <div className="viewport-head">
            <span className={`rec ${violations.length ? 'alert' : ''}`}>PROCESSED FEED · {video.tracker.toUpperCase()} TRACKING</span>
            <span>{video.width}×{video.height} · {video.analysed_fps} FPS ANALYSED</span>
          </div>
          <div className="viewport-body">
            <div className="frame">
              <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
              <video ref={videoRef} src={media.processed_video} controls playsInline poster={media.first_frame} />
            </div>
            {!media.video_browser_playable && (
              <div className="small" style={{ color: '#fbbf24', marginTop: 8, lineHeight: 1.4 }}>
                The processed video was encoded without H.264 and may not play in the browser. <a href={media.processed_video} download style={{ textDecoration: 'underline' }}>Download it</a> instead.
              </div>
            )}
          </div>
        </div>
        <Card title="Video details" icon={Settings2} bodyClass="card-body tight">
          <dl className="kv" style={{ gridTemplateColumns: '140px 1fr' }}>
            <dt>File</dt><dd style={{ wordBreak: 'break-all' }}>{result.filename}</dd>
            <dt>Duration</dt><dd>{video.duration != null ? timecode(video.duration) : '—'} @ {video.fps} fps</dd>
            <dt>Sampling</dt><dd>every {video.frame_stride} frame(s) → {video.analysed_fps} fps</dd>
            <dt>Tracker</dt><dd>{video.tracker} (persistent IDs across frames)</dd>
            <dt>Temporal rule</dt><dd>violation must persist across frames for the same track</dd>
            <dt>Red-light module</dt>
            <dd>{video.stop_line != null ? `stop line at ${Math.round(video.stop_line * 100)}% height, direction: ${video.line_direction}` : 'off (no stop line configured)'}</dd>
            <dt>Low-light frames</dt><dd>{video.low_light_frames}</dd>
            <dt>Download</dt><dd><a href={media.processed_video} download style={{ color: '#0891b2', fontWeight: 600 }}>processed.mp4</a></dd>
          </dl>
        </Card>
      </div>

      <SummaryStrip cells={cells} />

      <Card title="Activity timeline" icon={Activity} sub="Objects per analysed frame and when possible violations were confirmed">
        <Timeline points={result.timeline} violations={violations} duration={video.duration} onSeek={seek} showSignal={video.stop_line != null} />
      </Card>

      <div className="grid g-main">
        <div className="stack">
          <div className="card-title" style={{ color: '#e6eef9' }}><Film size={16} style={{ color: '#22d3ee' }} /> Evidence frames ({violations.length})</div>
          <ViolationList violations={violations} onZoom={zoom} onSeek={seek} />
          <Disclaimer text={result.disclaimer} />
        </div>
        <Observations items={result.observations} />
      </div>
      {lightbox}
    </div>
  )
}
