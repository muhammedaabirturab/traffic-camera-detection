import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ExternalLink, History as HistoryIcon, Trash2 } from 'lucide-react'
import { api } from '../lib/api.js'
import { Card, Empty, Notice, Spinner } from '../components/ui.jsx'
import { dateTime, pct, ruleName, secs } from '../lib/format.js'

const STATUS = { completed: 'ok', processing: 'info', failed: 'high' }

export default function History() {
  const [items, setItems] = useState(null)
  const [kind, setKind] = useState('')
  const [error, setError] = useState(null)
  const nav = useNavigate()

  const load = () => api.history(kind || undefined).then((r) => setItems(r.items)).catch((e) => setError(e.message))
  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind])

  const remove = async (e, id) => {
    e.stopPropagation()
    if (!confirm('Delete this analysis and its evidence files?')) return
    await api.deleteHistory(id)
    load()
  }

  return (
    <div className="stack">
      {error && <Notice>{error}</Notice>}
      <Card title="Analysis records" icon={HistoryIcon} sub="Stored in a local SQLite database (data/history.sqlite3)"
            actions={
              <div className="row" style={{ gap: 6 }}>
                {[['', 'All'], ['image', 'Images'], ['video', 'Videos']].map(([k, n]) => (
                  <button key={k} className={`btn btn-sm ${kind === k ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setKind(k)}>{n}</button>
                ))}
              </div>
            } bodyClass="card-body tight">
        {items == null ? (
          <div className="empty"><Spinner /></div>
        ) : items.length === 0 ? (
          <Empty icon={HistoryIcon} title="No analyses recorded">Analyses you run are listed here automatically.</Empty>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th></th><th>Date / time</th><th>File</th><th>Type</th><th>Vehicles</th><th>Possible violations</th>
                  <th>Violation types</th><th>Top confidence</th><th>Processing</th><th></th>
                </tr>
              </thead>
              <tbody>
                {items.map((r) => {
                  const types = [...new Set(r.violation_types)]
                  return (
                    <tr key={r.id} className="clickable" onClick={() => r.status === 'completed' && nav(`/history/${r.id}`)}>
                      <td style={{ width: 84 }}>{r.thumbnail ? <img className="thumb" src={r.thumbnail} alt="" /> : <div className="thumb" />}</td>
                      <td className="small" style={{ whiteSpace: 'nowrap' }}>{dateTime(r.created_at)}</td>
                      <td style={{ maxWidth: 220, wordBreak: 'break-all' }}><b>{r.filename}</b><div className="mono small muted">{r.id}</div></td>
                      <td>
                        <span className="badge cyan">{r.kind.toUpperCase()}</span>
                        {r.status !== 'completed' && <span className={`badge ${STATUS[r.status]}`} style={{ marginLeft: 6 }}>{r.status}</span>}
                      </td>
                      <td className="mono">{r.vehicles}</td>
                      <td className="mono">{r.violations}</td>
                      <td>
                        <div className="row wrap" style={{ gap: 4 }}>
                          {types.length ? types.map((t) => <span key={t} className="badge medium">{ruleName(t)}</span>) : <span className="muted small">—</span>}
                        </div>
                      </td>
                      <td className="mono small">{pct(r.confidence)}</td>
                      <td className="mono small">{secs(r.processing_time)}</td>
                      <td style={{ whiteSpace: 'nowrap' }}>
                        {r.status === 'completed' && (
                          <button className="icon-btn" title="Open" onClick={(e) => { e.stopPropagation(); nav(`/history/${r.id}`) }}>
                            <ExternalLink size={14} />
                          </button>
                        )}{' '}
                        <button className="icon-btn" title="Delete" onClick={(e) => remove(e, r.id)}><Trash2 size={14} /></button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
