import { useCallback, useRef, useState } from 'react'
import { AlertTriangle, Inbox, Loader2, UploadCloud } from 'lucide-react'
import { pct } from '../lib/format.js'

export function Card({ title, sub, icon: Icon, actions, children, className = '', bodyClass = 'card-body' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && (
              <div className="card-title">
                {Icon && <Icon size={16} />}
                {title}
              </div>
            )}
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {actions}
        </div>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  )
}

export function Kpi({ label, value, foot, icon: Icon, alert }) {
  return (
    <div className={`card kpi ${alert ? 'alert' : ''}`}>
      <div className="kpi-label">
        {Icon && <Icon size={14} />}
        {label}
      </div>
      <div className="kpi-value">{value}</div>
      {foot && <div className="kpi-foot">{foot}</div>}
    </div>
  )
}

export function ConfidenceMeter({ value, band, neutral }) {
  const cls = neutral ? 'neutral' : band || 'low'
  return (
    <div className="meter" title={band ? `${band} confidence` : undefined}>
      <div className="meter-track">
        <div className={`meter-fill ${cls}`} style={{ width: `${Math.round((value || 0) * 100)}%` }} />
      </div>
      <span className="meter-value">{pct(value)}</span>
    </div>
  )
}

export function BandBadge({ band }) {
  if (!band) return null
  return <span className={`badge ${band}`}>{band.toUpperCase()} CONFIDENCE</span>
}

export function Empty({ icon: Icon = Inbox, title, children }) {
  return (
    <div className="empty">
      <Icon size={34} />
      {title && <h4>{title}</h4>}
      <div className="small">{children}</div>
    </div>
  )
}

export function Notice({ children, variant = '', icon: Icon = AlertTriangle }) {
  return (
    <div className={`notice ${variant}`}>
      <Icon size={16} />
      <div>{children}</div>
    </div>
  )
}

export function Spinner({ size = 16 }) {
  return <Loader2 size={size} className="spin" />
}

export function useLightbox() {
  const [src, setSrc] = useState(null)
  const node = src ? (
    <div className="lightbox" onClick={() => setSrc(null)}>
      <img src={src} alt="Evidence" />
    </div>
  ) : null
  return [setSrc, node]
}

export function Dropzone({ accept, onFile, title, hint, formats, disabled }) {
  const input = useRef(null)
  const [drag, setDrag] = useState(false)
  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDrag(false)
      if (disabled) return
      const f = e.dataTransfer.files?.[0]
      if (f) onFile(f)
    },
    [onFile, disabled],
  )
  return (
    <div
      className={`dropzone ${drag ? 'drag' : ''}`}
      onClick={() => !disabled && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDrag(true)
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      role="button"
      tabIndex={0}
    >
      <div className="drop-icon">
        <UploadCloud size={26} />
      </div>
      <h3>{title}</h3>
      <p>{hint}</p>
      <div className="formats">
        {formats.map((f) => (
          <span key={f} className="chip cyan">
            {f}
          </span>
        ))}
      </div>
      <input
        ref={input}
        type="file"
        accept={accept}
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = ''
        }}
      />
    </div>
  )
}
