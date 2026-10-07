import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Clapperboard, ImageIcon, RotateCcw, TrafficCone, X } from 'lucide-react'
import { api } from '../lib/api.js'
import { Dropzone, Notice, Spinner } from '../components/ui.jsx'
import ImageResult from '../components/ImageResult.jsx'
import VideoResult from '../components/VideoResult.jsx'

function Processing({ title, progress, message, preview }) {
  return (
    <div className="grid g-2 fade-in">
      <div className="viewport scanning">
        <div className="viewport-head"><span className="rec">PROCESSING</span><span>YOLO INFERENCE</span></div>
        <div className="viewport-body">
          {preview ? <img src={preview} alt="" style={{ width: '100%', borderRadius: 6, opacity: 0.85 }} />
                   : <div style={{ height: 260 }} />}
        </div>
      </div>
      <div className="panel-dark" style={{ alignSelf: 'start' }}>
        <div className="row" style={{ gap: 10 }}><Spinner /> <h4 style={{ margin: 0 }}>{title}</h4></div>
        <p style={{ marginTop: 10 }}>{message || 'Running detection, relationship analysis and the traffic rule engine…'}</p>
        {progress != null && (
          <>
            <div className="progress" style={{ marginTop: 14 }}><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>
            <div className="mono small" style={{ marginTop: 6, color: '#8ea2bf' }}>{Math.round(progress * 100)}%</div>
          </>
        )}
      </div>
    </div>
  )
}

// Grabs the first frame of the selected video in the browser so the operator can place the stop line.
function useFirstFrame(file) {
  const [frame, setFrame] = useState(null)
  useEffect(() => {
    if (!file) return setFrame(null)
    const url = URL.createObjectURL(file)
    const v = document.createElement('video')
    v.muted = true
    v.preload = 'auto'
    v.src = url
    const onSeeked = () => {
      const c = document.createElement('canvas')
      c.width = v.videoWidth
      c.height = v.videoHeight
      c.getContext('2d').drawImage(v, 0, 0)
      try {
        setFrame(c.toDataURL('image/jpeg', 0.85))
      } catch {
        setFrame(null)
      }
    }
    v.addEventListener('loadeddata', () => { v.currentTime = Math.min(0.2, (v.duration || 1) / 2) }, { once: true })
    v.addEventListener('seeked', onSeeked, { once: true })
    v.addEventListener('error', () => setFrame(null), { once: true })
    return () => URL.revokeObjectURL(url)
  }, [file])
  return frame
}

function VideoSetup({ file, onCancel, onStart }) {
  const frame = useFirstFrame(file)
  const [enabled, setEnabled] = useState(false)
  const [line, setLine] = useState(0.65)
  const [direction, setDirection] = useState('any')
  return (
    <div className="grid g-main fade-in">
      <div className="viewport">
        <div className="viewport-head">
          <span className="rec">{file.name}</span>
          <span>{(file.size / 1e6).toFixed(1)} MB</span>
        </div>
        <div className="viewport-body">
          {frame ? (
            <div className="line-editor">
              <img src={frame} alt="First frame" style={{ width: '100%' }} />
              {enabled && (
                <div className="stopline" style={{ top: `${line * 100}%` }}>
                  <span>STOP LINE {Math.round(line * 100)}%</span>
                </div>
              )}
            </div>
          ) : (
            <div className="small" style={{ color: '#8ea2bf', padding: 30, textAlign: 'center', lineHeight: 1.6 }}>
              Preview not available for this format in the browser (e.g. AVI) — analysis still works.
            </div>
          )}
        </div>
      </div>
      <div className="panel-dark stack" style={{ gap: 16, alignSelf: 'start' }}>
        <div>
          <h4>Video analysis</h4>
          <p>Frames are sampled, tracked with ByteTrack, and a possible violation is only reported if it persists across
            frames for the same vehicle / rider.</p>
        </div>
        <label className="switch">
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
          <TrafficCone size={16} style={{ color: '#f97316' }} /> Enable red-light module
        </label>
        {enabled && (
          <>
            <p>Place the stop line for this camera. The rule is only evaluated while a traffic light is visible and its
              colour can be read; otherwise it is reported as insufficient evidence.</p>
            <div className="field">
              Stop line position ({Math.round(line * 100)}% of frame height)
              <input type="range" min="0.05" max="0.95" step="0.01" value={line} onChange={(e) => setLine(Number(e.target.value))} />
            </div>
            <div className="field">
              Direction of travel across the line
              <select value={direction} onChange={(e) => setDirection(e.target.value)}>
                <option value="any">Any direction</option>
                <option value="down">Moving down the frame (towards camera)</option>
                <option value="up">Moving up the frame (away from camera)</option>
              </select>
            </div>
          </>
        )}
        <div className="row" style={{ gap: 10 }}>
          <button className="btn btn-primary" onClick={() => onStart(enabled ? { stopLine: line, direction } : {})}>
            <Clapperboard size={16} /> Analyze video
          </button>
          <button className="btn btn-dark" onClick={onCancel}><X size={15} /> Cancel</button>
        </div>
      </div>
    </div>
  )
}

export default function Analyze() {
  const [params, setParams] = useSearchParams()
  const mode = params.get('mode') === 'video' ? 'video' : 'image'
  const [state, setState] = useState({ phase: 'idle' })
  const previewUrl = useRef(null)

  const reset = () => setState({ phase: 'idle' })
  const switchMode = (m) => {
    setParams(m === 'video' ? { mode: 'video' } : {})
    reset()
  }

  const runImage = async (file) => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current)
    previewUrl.current = URL.createObjectURL(file)
    setState({ phase: 'processing', preview: previewUrl.current, title: `Analyzing ${file.name}` })
    try {
      const result = await api.analyzeImage(file)
      setState({ phase: 'done', result })
    } catch (e) {
      setState({ phase: 'error', error: e.message })
    }
  }

  const runVideo = async (file, opts) => {
    setState({ phase: 'processing', title: `Analyzing ${file.name}`, progress: 0, message: 'Uploading video…' })
    try {
      const { job_id, analysis_id } = await api.analyzeVideo(file, opts)
      await api.waitForJob(job_id, (job) =>
        setState((s) => ({ ...s, progress: job.progress, message: job.message })))
      const rec = await api.historyItem(analysis_id)
      setState({ phase: 'done', result: rec.result })
    } catch (e) {
      setState({ phase: 'error', error: e.message })
    }
  }

  return (
    <div className="stack">
      <div className="row between wrap">
        <div className="tabs">
          <button className={`tab ${mode === 'image' ? 'active' : ''}`} onClick={() => switchMode('image')}>
            <ImageIcon size={15} /> Image analysis
          </button>
          <button className={`tab ${mode === 'video' ? 'active' : ''}`} onClick={() => switchMode('video')}>
            <Clapperboard size={15} /> Video analysis
          </button>
        </div>
        {state.phase !== 'idle' && state.phase !== 'processing' && (
          <button className="btn btn-dark" onClick={reset}><RotateCcw size={15} /> New analysis</button>
        )}
      </div>

      {state.phase === 'idle' && mode === 'image' && (
        <Dropzone accept=".jpg,.jpeg,.png,image/jpeg,image/png" onFile={runImage}
                  title="Drop a traffic-camera image here" hint="or click to browse — the image is analysed locally by the YOLO pipeline"
                  formats={['JPG', 'JPEG', 'PNG']} />
      )}
      {state.phase === 'idle' && mode === 'video' && (
        <Dropzone accept=".mp4,.avi,.mov,video/mp4,video/quicktime,video/x-msvideo" onFile={(f) => setState({ phase: 'setup', file: f })}
                  title="Drop a traffic-camera video here" hint="frame-by-frame YOLO detection with ByteTrack object tracking"
                  formats={['MP4', 'AVI', 'MOV']} />
      )}
      {state.phase === 'setup' && <VideoSetup file={state.file} onCancel={reset} onStart={(o) => runVideo(state.file, o)} />}
      {state.phase === 'processing' && <Processing {...state} />}
      {state.phase === 'error' && (
        <Notice>
          <b>Analysis failed:</b> {state.error}
        </Notice>
      )}
      {state.phase === 'done' && (
        <>
          <div className="row small" style={{ color: '#8ea2bf', gap: 8 }}>
            Saved to history as <span className="mono" style={{ color: '#22d3ee' }}>{state.result.analysis_id}</span> ·
            <Link to={`/history/${state.result.analysis_id}`} style={{ textDecoration: 'underline' }}>open record</Link>
          </div>
          {state.result.kind === 'video' ? <VideoResult result={state.result} /> : <ImageResult result={state.result} />}
        </>
      )}
    </div>
  )
}
