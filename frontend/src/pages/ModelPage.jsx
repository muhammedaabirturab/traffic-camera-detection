import { useEffect, useState } from 'react'
import { BrainCircuit, CircleSlash, Cpu, Database, FolderInput, LineChart, RefreshCw } from 'lucide-react'
import { api } from '../lib/api.js'
import { Card, Empty, Notice, Spinner, useLightbox } from '../components/ui.jsx'
import { dateTime, label, pct } from '../lib/format.js'
import { useStatus } from '../lib/status.jsx'

const METRICS = [
  ['precision', 'Precision'],
  ['recall', 'Recall'],
  ['f1', 'F1 score'],
  ['map50', 'mAP@50'],
  ['map50_95', 'mAP@50-95'],
]

function TrainingRun({ run, onZoom }) {
  const m = run.metrics
  const t = run.training
  const ds = run.dataset_report
  return (
    <Card title={`${label(run.role)} model — ${run.weights_name}`} icon={LineChart}
          sub={`Evaluated on the held-out "${run.evaluated_split}" split · ${dateTime(run.created_at)}`}>
      <div className="metric-grid">
        {METRICS.map(([k, n]) => (
          <div key={k} className="metric"><div className="l">{n}</div><div className="v">{pct(m[k], 1)}</div></div>
        ))}
      </div>
      {m.per_class?.length > 0 && (
        <div className="table-wrap mt">
          <table className="table">
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>mAP@50</th><th>mAP@50-95</th></tr></thead>
            <tbody>
              {m.per_class.map((c) => (
                <tr key={c.class}>
                  <td><span className="class-chip">{c.class}</span></td>
                  <td className="mono">{pct(c.precision, 1)}</td><td className="mono">{pct(c.recall, 1)}</td>
                  <td className="mono">{pct(c.f1, 1)}</td><td className="mono">{pct(c.map50, 1)}</td><td className="mono">{pct(c.map50_95, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="grid g-2 mt">
        <dl className="kv">
          <dt>Classes</dt><dd>{run.classes.join(', ')}</dd>
          {t ? (
            <>
              <dt>Base model</dt><dd>{t.base_model}</dd>
              <dt>Epochs</dt><dd>{t.epochs_completed} completed / {t.epochs_requested} requested</dd>
              <dt>Image size</dt><dd>{t.imgsz}px</dd>
              <dt>Batch size</dt><dd>{t.batch}</dd>
              <dt>Training time</dt><dd>{t.duration_seconds < 120 ? `${Math.round(t.duration_seconds)} s` : `${Math.round(t.duration_seconds / 60)} min`}</dd>
            </>
          ) : (
            <><dt>Training</dt><dd className="muted">Metrics from scripts/validate.py (training details not recorded)</dd></>
          )}
          <dt>Hardware</dt><dd>{run.hardware?.accelerator} · torch {run.hardware?.torch}</dd>
        </dl>
        <dl className="kv">
          {ds ? (
            <>
              <dt>Dataset source</dt><dd style={{ wordBreak: 'break-all' }}>{ds.source?.split('/').slice(-1)[0]}</dd>
              <dt>Images</dt><dd>{Object.entries(ds.images_per_split || {}).map(([k, v]) => `${k} ${v}`).join(' · ')}</dd>
              <dt>Split</dt><dd>{ds.split_unit}, seed {ds.split_seed}</dd>
              <dt>Label checks</dt><dd>{Object.entries(ds.checks || {}).map(([k, v]) => `${label(k)}: ${v}`).join(' · ')}</dd>
            </>
          ) : (
            <><dt>Dataset</dt><dd>{run.dataset_yaml}</dd></>
          )}
        </dl>
      </div>
      {run.plots?.length > 0 && (
        <div className="plots mt">
          {run.plots.map((p) => (
            <figure key={p} style={{ margin: 0 }}>
              <img src={`/model-assets/${p}`} alt={p} onClick={() => onZoom(`/model-assets/${p}`)} />
              <figcaption className="small muted" style={{ marginTop: 4 }}>{p.split('/').pop()}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </Card>
  )
}

export default function ModelPage() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [zoom, lightbox] = useLightbox()
  const { refresh } = useStatus()

  const load = () => api.modelInfo().then(setInfo).catch((e) => setError(e.message))
  useEffect(() => { load() }, [])

  const reload = async () => {
    setBusy(true)
    try {
      await api.reloadModels()
      await load()
      refresh()
    } finally {
      setBusy(false)
    }
  }

  if (error) return <Notice>{error}</Notice>
  if (!info) return <div className="empty"><Spinner size={22} /></div>

  const allClasses = [...new Set(info.detectors.filter((d) => d.available).flatMap((d) => d.canonical_classes || []))]

  return (
    <div className="stack">
      <div className="grid g-main">
        <Card title="Model overview" icon={BrainCircuit} actions={
          <button className="btn btn-ghost btn-sm" onClick={reload} disabled={busy}>
            {busy ? <Spinner size={13} /> : <RefreshCw size={13} />} Reload models
          </button>}>
          <dl className="kv">
            <dt>Model</dt><dd>YOLO ({info.framework})</dd>
            <dt>Task</dt><dd>{info.task} + multi-object tracking ({info.tracker})</dd>
            <dt>Dataset</dt><dd>COCO (pretrained vehicles/persons) + Kaggle motorbike / helmet reference data for rider & helmet models</dd>
            <dt>Classes used</dt>
            <dd className="row wrap" style={{ gap: 5 }}>{allClasses.map((c) => <span key={c} className="class-chip">{c}</span>)}</dd>
            <dt>Input</dt><dd>{info.inputs.join(' · ')}</dd>
            <dt>Inference size</dt><dd>{info.inference_imgsz}px · device setting “{info.device_setting}”</dd>
            <dt>Runtime</dt><dd>Python {info.runtime.python} · torch {info.runtime.torch} · {info.runtime.accelerator}</dd>
          </dl>
        </Card>
        <Card title="Measured performance" icon={LineChart}>
          {info.trained ? (
            <div className="small" style={{ lineHeight: 1.6 }}>
              {info.training_runs.map((r) => (
                <div key={r.role} className="row between" style={{ padding: '6px 0', borderBottom: '1px dashed #e3e9f2' }}>
                  <span><b>{label(r.role)}</b> · {r.weights_name}</span>
                  <span className="mono">mAP@50 {pct(r.metrics.map50, 1)}</span>
                </div>
              ))}
              <div className="muted" style={{ marginTop: 8 }}>Read from models/metrics/*.json — produced by scripts/train.py / validate.py.</div>
            </div>
          ) : (
            <Empty icon={CircleSlash} title="Model not trained / metrics unavailable">
              No metrics file exists in <code>models/metrics/</code>. Train a model with <code>scripts/train.py</code> (or the
              notebook in <code>notebooks/training/</code>) and the measured precision, recall, F1 and mAP will appear here.
              Numbers are never estimated or hard-coded.
            </Empty>
          )}
        </Card>
      </div>

      <Card title="Detectors" icon={Cpu} sub="Each role is loaded if its weights are present in models/" bodyClass="card-body tight">
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Role</th><th>Status</th><th>Backend / weights</th><th>Classes (canonical)</th><th>Provenance</th></tr></thead>
            <tbody>
              {info.detectors.map((d) => (
                <tr key={d.role}>
                  <td><b>{label(d.role)}</b><div className="small muted" style={{ maxWidth: 260 }}>{d.description}</div></td>
                  <td><span className={`badge ${d.available ? 'ok' : 'muted'}`}>{d.available ? 'LOADED' : 'NOT INSTALLED'}</span>
                    {d.error && <div className="small" style={{ color: '#b42318', maxWidth: 200 }}>{d.error}</div>}</td>
                  <td className="small">{d.available ? <><span className="mono">{d.weights}</span><div className="muted">{d.backend}</div></> : '—'}</td>
                  <td><div className="row wrap" style={{ gap: 4 }}>{(d.canonical_classes || []).map((c) => <span key={c} className="class-chip">{c}</span>)}</div></td>
                  <td className="small muted" style={{ maxWidth: 320 }}>{d.provenance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {info.training_runs.map((r) => <TrainingRun key={r._file} run={r} onZoom={zoom} />)}

      <Card title="Installing detectors" icon={FolderInput} sub="Put files in the models/ folder, then click “Reload models”">
        <div className="grid g-3">
          <div className="prose">
            <h3 style={{ marginTop: 0 }}>Helmet — pretrained (no training)</h3>
            Download the Kaggle dataset <code>savanagrawal/helmet-detection-yolov3</code> and copy{' '}
            <code>yolov3-helmet.cfg</code>, <code>yolov3-helmet.weights</code> and <code>helmet.names</code> into <code>models/</code>.
            They run through OpenCV DNN, exactly as in the reference notebook.
          </div>
          <div className="prose">
            <h3 style={{ marginTop: 0 }}>Rider — train on the Kaggle data</h3>
            <code>python scripts/prepare_dataset.py --source &lt;kaggle folder&gt; --out datasets/rider --names rider</code><br />
            <code>python scripts/train.py --data datasets/rider/data.yaml --role rider</code><br />
            Weights and metrics are installed automatically.
          </div>
          <div className="prose">
            <h3 style={{ marginTop: 0 }}>Helmet — train your own</h3>
            With any helmet dataset that includes images (e.g. classes <i>With Helmet / Without Helmet</i>):{' '}
            <code>prepare_dataset.py --format voc --class-map "With Helmet=helmet,Without Helmet=no_helmet"</code> then{' '}
            <code>train.py --role helmet</code>. A bare-head class makes no-helmet calls stronger.
          </div>
        </div>
        <div className="small muted mt row" style={{ gap: 6 }}><Database size={13} /> Datasets and weights are git-ignored — see datasets/README.md and models/README.md.</div>
      </Card>
      {lightbox}
    </div>
  )
}
