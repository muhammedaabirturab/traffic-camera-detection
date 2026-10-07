import { AlertTriangle, Clock, ExternalLink, Maximize2, ShieldQuestion } from 'lucide-react'
import { ConfidenceMeter, BandBadge } from './ui.jsx'
import { label, pct, timecode } from '../lib/format.js'

function LegalRef({ ref_ }) {
  if (!ref_?.section) return <span className="muted">Not verified for this project (intentionally left blank)</span>
  return (
    <span className="legal">
      <b>{ref_.section}</b> — {ref_.title}, {ref_.act}
      {ref_.penalty_section && (
        <>
          <br />
          <span className="muted">Penalty provision: {ref_.penalty_section}</span>
        </>
      )}
      {ref_.source_url && (
        <a href={ref_.source_url} target="_blank" rel="noreferrer" style={{ marginLeft: 6, color: '#0891b2' }}>
          <ExternalLink size={12} style={{ verticalAlign: '-2px' }} />
        </a>
      )}
    </span>
  )
}

export default function ViolationCard({ v, onZoom, onSeek }) {
  const img = v.evidence_images?.crop
  return (
    <article className={`violation band-${v.confidence_band} fade-in`}>
      <div className="violation-media">
        {img ? <img src={img} alt="Evidence crop" onClick={() => onZoom?.(v.evidence_images.frame)} /> : null}
        <div className="media-actions">
          {v.evidence_images?.frame && (
            <button className="btn btn-dark btn-sm" onClick={() => onZoom?.(v.evidence_images.frame)}>
              <Maximize2 size={13} /> Evidence frame
            </button>
          )}
          {v.timestamp != null && onSeek && (
            <button className="btn btn-dark btn-sm" onClick={() => onSeek(v.timestamp)}>
              <Clock size={13} /> {timecode(v.timestamp)}
            </button>
          )}
        </div>
      </div>
      <div className="violation-body">
        <div className="row between wrap">
          <div className="violation-flag">
            <AlertTriangle size={15} /> Possible traffic violation
          </div>
          <BandBadge band={v.confidence_band} />
        </div>
        <h3 className="violation-title">{v.title}</h3>
        <dl className="kv">
          <dt>Violation</dt>
          <dd>{v.violation_name}</dd>
          <dt>Confidence</dt>
          <dd>
            <ConfidenceMeter value={v.confidence} band={v.confidence_band} />
          </dd>
          <dt>Vehicle</dt>
          <dd>
            {label(v.vehicle.type)} <span className="muted">· detected at {pct(v.vehicle.confidence)}</span>
            {v.vehicle.track_id != null && <span className="badge cyan" style={{ marginLeft: 8 }}>TRACK #{v.vehicle.track_id}</span>}
          </dd>
          <dt>Evidence</dt>
          <dd>{v.evidence}</dd>
          {v.timestamp != null && (
            <>
              <dt>Observed</dt>
              <dd>
                {timecode(v.timestamp)} (frame {v.frame_index}) · seen in {v.frames_observed}/{v.frames_evaluable} analysed frames
              </dd>
            </>
          )}
          {v.plate_reading && (
            <>
              <dt>Plate (AI read)</dt>
              <dd>
                <span className="mono" style={{ fontWeight: 600 }}>{v.plate_reading.text || 'unreadable'}</span>{' '}
                <span className="muted small">— {v.plate_reading.note}</span>
              </dd>
            </>
          )}
          <dt>Status</dt>
          <dd>
            <span className="verify-pill">
              <ShieldQuestion size={14} /> {v.status}
            </span>
          </dd>
          <dt>Legal reference</dt>
          <dd>
            <LegalRef ref_={v.legal_reference} />
          </dd>
        </dl>
        <div className="explain">{v.explanation}</div>
        {v.penalty_note && <div className="small muted" style={{ marginTop: 8 }}>Penalty note: {v.penalty_note}</div>}
      </div>
    </article>
  )
}
