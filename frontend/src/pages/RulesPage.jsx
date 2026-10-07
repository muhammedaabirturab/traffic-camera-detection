import { Fragment, useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, ExternalLink, Scale } from 'lucide-react'
import { api } from '../lib/api.js'
import { Card, Notice, Spinner } from '../components/ui.jsx'
import { label } from '../lib/format.js'

const STATUS = {
  active: ['ok', 'ACTIVE'],
  video_only: ['info', 'VIDEO ONLY'],
  not_implemented: ['muted', 'NOT IMPLEMENTED'],
}

export default function RulesPage() {
  const [data, setData] = useState(null)
  const [open, setOpen] = useState({})
  const [error, setError] = useState(null)
  useEffect(() => { api.rules().then(setData).catch((e) => setError(e.message)) }, [])

  if (error) return <Notice>{error}</Notice>
  if (!data) return <div className="empty"><Spinner size={22} /></div>

  return (
    <div className="stack">
      <Notice variant="info" icon={Scale}>
        {data.disclaimer} <span style={{ opacity: 0.8 }}>Rule database v{data.version} · last reviewed {data.last_reviewed} · {data.jurisdiction}</span>
      </Notice>
      <Card title="Rule database" icon={Scale} sub="Click a row for detection logic, penalty note and source. Edit app/rules/traffic_rules.json to update — changes are picked up automatically." bodyClass="card-body tight">
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th style={{ width: 28 }}></th><th>Violation</th><th>Description</th><th>Detection method</th><th>Legal reference</th><th>Status</th></tr>
            </thead>
            <tbody>
              {data.rules.map((r) => {
                const [cls, text] = STATUS[r.status] || ['muted', r.status]
                const ref = r.legal_reference
                const isOpen = open[r.violation_id]
                return (
                  <Fragment key={r.violation_id}>
                    <tr className="clickable" onClick={() => setOpen({ ...open, [r.violation_id]: !isOpen })}>
                      <td>{isOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}</td>
                      <td style={{ minWidth: 170 }}>
                        <b>{r.violation_name}</b>
                        <div className="mono small muted">{r.violation_id}</div>
                      </td>
                      <td className="small" style={{ maxWidth: 260 }}>{r.description}</td>
                      <td className="small" style={{ maxWidth: 220 }}>
                        {r.input_types.length ? r.input_types.map((t) => label(t)).join(' + ') : '—'}
                        {r.required_models.length > 0 && <div className="muted">needs: {r.required_models.join(', ')} model</div>}
                        {r.min_confidence != null && <div className="muted">report ≥ {Math.round(r.min_confidence * 100)}% confidence</div>}
                      </td>
                      <td className="legal" style={{ minWidth: 200 }}>
                        {ref.section ? (
                          <>
                            <b>{ref.section}</b>, {ref.act}
                            <div className="muted small">{ref.title}</div>
                            {ref.penalty_section && <div className="muted small">Penalty: {ref.penalty_section}</div>}
                          </>
                        ) : (
                          <span className="muted small">Not verified — left blank</span>
                        )}
                      </td>
                      <td><span className={`badge ${cls} rule-status`}>{text}</span></td>
                    </tr>
                    {isOpen && (
                      <tr className="expand-row">
                        <td></td>
                        <td colSpan={5}>
                          <div className="grid g-2" style={{ gap: 24 }}>
                            <div>
                              <b>Detection logic.</b> {r.detection_logic}
                              <div style={{ marginTop: 8 }}><b>Applies to:</b> {r.applicable_vehicle_type.map(label).join(', ') || '—'}</div>
                            </div>
                            <div>
                              {ref.notes && <div><b>Legal note.</b> {ref.notes}</div>}
                              <div style={{ marginTop: 6 }}><b>Penalty note.</b> {r.penalty_note}</div>
                              <div style={{ marginTop: 6 }}><b>Verification.</b> {ref.verification}</div>
                              {ref.source_url && (
                                <a href={ref.source_url} target="_blank" rel="noreferrer" className="row" style={{ gap: 5, color: '#0891b2', marginTop: 6, fontWeight: 600 }}>
                                  Source <ExternalLink size={12} />
                                </a>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
      <Card title="Status legend" bodyClass="card-body tight">
        {Object.entries(data.status_legend).map(([k, v]) => (
          <div key={k} className="row small" style={{ padding: '5px 0', gap: 10 }}>
            <span className={`badge ${(STATUS[k] || ['muted'])[0]} rule-status`} style={{ minWidth: 130, justifyContent: 'center' }}>{(STATUS[k] || [0, k])[1]}</span>
            {v}
          </div>
        ))}
      </Card>
    </div>
  )
}
