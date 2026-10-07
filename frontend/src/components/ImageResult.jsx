import { useState } from 'react'
import { Bike, Boxes, Download } from 'lucide-react'
import DetectionOverlay, { LAYERS } from './DetectionOverlay.jsx'
import { Card, ConfidenceMeter, Notice, useLightbox } from './ui.jsx'
import { Disclaimer, Observations, PipelineTrace, SummaryStrip, ViolationList } from './ResultParts.jsx'
import { CLASS_COLORS, VEHICLES, label, pct, secs } from '../lib/format.js'

function Viewport({ head, alert, children, foot }) {
  return (
    <div className="viewport">
      <div className="viewport-head">
        <span className={`rec ${alert ? 'alert' : ''}`}>{head}</span>
        <span>{foot}</span>
      </div>
      <div className="viewport-body">{children}</div>
    </div>
  )
}

const HELMET_STATUS = {
  helmet: ['ok', 'Helmet detected'],
  no_helmet: ['high', 'No helmet detected'],
  insufficient_evidence: ['low', 'Insufficient evidence'],
  not_evaluated: ['muted', 'Not evaluated'],
}

export default function ImageResult({ result }) {
  const [layers, setLayers] = useState(Object.fromEntries(LAYERS.map((l) => [l.key, true])))
  const [hovered, setHovered] = useState(null)
  const [zoom, lightbox] = useLightbox()
  const { image, media, summary, detections, violations } = result
  const byClass = summary.by_class || {}

  const cells = [
    { l: 'Vehicles detected', v: summary.vehicles },
    ...VEHICLES.filter((k) => byClass[k]).map((k) => ({ l: `${label(k)}s`, v: byClass[k] })),
    { l: 'Persons', v: summary.persons ?? 0 },
    { l: 'Riders', v: summary.riders ?? 0 },
    { l: 'Possible violations', v: violations.length, alert: violations.length > 0 },
    { l: 'Processing time', v: secs(result.processing_time) },
  ]

  return (
    <div className="stack fade-in">
      {!result.models?.helmet_model && summary.riders > 0 && (
        <Notice variant="info">
          Helmet model not loaded — helmet checks are reported as <b>insufficient evidence</b> rather than guessed. Add
          the Kaggle YOLOv3 helmet weights or a trained <code>helmet.pt</code> to <code>models/</code> (see the Model page).
        </Notice>
      )}
      <div className="grid g-2">
        <Viewport head="ORIGINAL INPUT" foot={`${image.width}×${image.height}${image.low_light_enhanced ? ' · LOW-LIGHT ENHANCED' : ''}`}>
          <div className="frame">
            <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
            <img src={media.original} alt={result.filename} />
          </div>
        </Viewport>
        <div className="viewport">
          <div className="viewport-head">
            <span className={`rec ${violations.length ? 'alert' : ''}`}>AI ANALYSIS · {detections.length} OBJECTS</span>
            <a href={media.annotated} download className="row" style={{ gap: 6, color: '#8ea2bf' }}>
              <Download size={13} /> ANNOTATED
            </a>
          </div>
          <div className="viewport-body">
            <div className="frame">
              <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
              <img src={media.original} alt="Analysis" />
              <DetectionOverlay width={image.width} height={image.height} detections={detections} violations={violations}
                                layers={layers} hovered={hovered} onHover={setHovered} />
            </div>
          </div>
          <div className="viewport-foot">
            {LAYERS.map((l) => (
              <button key={l.key} className={`layer-toggle ${layers[l.key] ? 'on' : ''}`}
                      onClick={() => setLayers({ ...layers, [l.key]: !layers[l.key] })}>
                <i style={{ background: l.color }} /> {l.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <SummaryStrip cells={cells} />

      <div className="grid g-main">
        <div className="stack">
          <ViolationList violations={violations} onZoom={zoom} />
          <Disclaimer text={result.disclaimer} />
        </div>
        <div className="stack">
          <Observations items={result.observations} />
          {result.two_wheelers?.length > 0 && (
            <Card title="Two-wheeler relationship analysis" icon={Bike} sub="Person ↔ motorcycle association and helmet outcome" bodyClass="card-body tight">
              {result.two_wheelers.map((u, i) => (
                <div key={i} style={{ padding: '8px 0', borderBottom: '1px dashed #e3e9f2' }}>
                  <div className="row between small">
                    <b>Motorcycle #{u.vehicle_id}</b>
                    <span className="muted">{u.riders.length} rider(s) · det. {pct(u.vehicle_confidence)}</span>
                  </div>
                  {u.riders.length === 0 && <div className="small muted">No person seated on this vehicle.</div>}
                  {u.riders.map((r) => {
                    const [cls, text] = HELMET_STATUS[r.helmet_status] || ['muted', r.helmet_status]
                    return (
                      <div key={r.person_id} className="row between small" style={{ marginTop: 6 }}
                           onMouseEnter={() => setHovered(r.person_id)} onMouseLeave={() => setHovered(null)}>
                        <span>Person #{r.person_id} · association {r.association.toFixed(2)}</span>
                        <span className={`badge ${cls}`} title={r.reason}>{text}</span>
                      </div>
                    )
                  })}
                </div>
              ))}
            </Card>
          )}
          <PipelineTrace stages={result.pipeline} />
          <Card title="Detected objects" icon={Boxes} sub="Hover a row to highlight it" bodyClass="card-body tight">
            <div className="table-wrap" style={{ maxHeight: 320, overflowY: 'auto' }}>
              <table className="table">
                <thead>
                  <tr><th>#</th><th>Class</th><th>Confidence</th><th>Model</th></tr>
                </thead>
                <tbody>
                  {detections.map((d) => (
                    <tr key={d.id} onMouseEnter={() => setHovered(d.id)} onMouseLeave={() => setHovered(null)}
                        style={{ background: hovered === d.id ? '#f0fbfe' : undefined }}>
                      <td className="mono">{d.id}</td>
                      <td>
                        <span className="row" style={{ gap: 7 }}>
                          <i style={{ width: 9, height: 9, borderRadius: 2, background: CLASS_COLORS[d.label] || '#ccc', border: '1px solid #0002' }} />
                          {label(d.label)}
                        </span>
                      </td>
                      <td style={{ width: 170 }}><ConfidenceMeter value={d.confidence} neutral /></td>
                      <td className="small muted">{d.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
      {lightbox}
    </div>
  )
}
