import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { api } from '../lib/api.js'
import { Notice, Spinner } from '../components/ui.jsx'
import ImageResult from '../components/ImageResult.jsx'
import VideoResult from '../components/VideoResult.jsx'
import { dateTime } from '../lib/format.js'

export default function HistoryDetail() {
  const { id } = useParams()
  const [rec, setRec] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setRec(null)
    api.historyItem(id).then(setRec).catch((e) => setError(e.message))
  }, [id])

  return (
    <div className="stack">
      <div className="row between wrap">
        <Link to="/history" className="btn btn-dark"><ArrowLeft size={15} /> Back to history</Link>
        {rec && (
          <div className="small" style={{ color: '#8ea2bf' }}>
            <span className="mono" style={{ color: '#22d3ee' }}>{rec.id}</span> · {rec.filename} · {dateTime(rec.created_at)}
          </div>
        )}
      </div>
      {error && <Notice>{error}</Notice>}
      {!rec && !error && <div className="empty"><Spinner size={22} /></div>}
      {rec && rec.status !== 'completed' && (
        <Notice>
          This analysis is <b>{rec.status}</b>
          {rec.result?.error ? `: ${rec.result.error}` : '.'}
        </Notice>
      )}
      {rec?.status === 'completed' && rec.result && (rec.kind === 'video' ? <VideoResult result={rec.result} /> : <ImageResult result={rec.result} />)}
    </div>
  )
}
