import { CheckCircle2, CircleDashed, HelpCircle, ShieldCheck, Workflow } from 'lucide-react'
import { Card, Empty } from './ui.jsx'
import ViolationCard from './ViolationCard.jsx'
import { ruleName } from '../lib/format.js'

export function SummaryStrip({ cells }) {
  return (
    <div className="card">
      <div className="summary-strip">
        {cells.map((c) => (
          <div key={c.l} className={`summary-cell ${c.alert ? 'alert' : ''}`}>
            <div className="l">{c.l}</div>
            <div className="v">{c.v}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function ViolationList({ violations, onZoom, onSeek }) {
  if (!violations?.length) {
    return (
      <Card>
        <Empty icon={ShieldCheck} title="No possible violations reported">
          No rule produced a candidate above its reporting threshold based on the available visual evidence. Items the
          system could not decide on are listed under “Insufficient visual evidence”.
        </Empty>
      </Card>
    )
  }
  return (
    <div className="stack">
      {violations.map((v) => (
        <ViolationCard key={v.id} v={v} onZoom={onZoom} onSeek={onSeek} />
      ))}
    </div>
  )
}

export function Observations({ items }) {
  return (
    <Card title="Insufficient visual evidence" icon={HelpCircle} sub="Checked but not reported — avoids false positives">
      {items?.length ? (
        items.map((o, i) => (
          <div className="obs" key={i}>
            <CircleDashed size={15} />
            <div>
              <div className="small" style={{ fontWeight: 600, color: '#0b1424' }}>{ruleName(o.rule_id)}</div>
              {o.message}
            </div>
          </div>
        ))
      ) : (
        <div className="obs">
          <CheckCircle2 size={15} style={{ color: '#16a34a' }} />
          Every evaluated rider / vehicle produced a definite outcome.
        </div>
      )}
    </Card>
  )
}

export function PipelineTrace({ stages }) {
  if (!stages?.length) return null
  return (
    <Card title="Analysis pipeline" icon={Workflow} sub="Executed stages with timings">
      <div className="pipeline">
        {stages.map((s, i) => (
          <div className="pipe-step" key={s.stage}>
            <div className="pipe-dot mono" style={{ fontSize: 10, fontWeight: 700 }}>{i + 1}</div>
            <div>
              <div className="pipe-name">{s.stage}</div>
              <div className="pipe-detail">{s.detail}</div>
            </div>
            <div className="pipe-ms">{s.ms} ms</div>
          </div>
        ))}
      </div>
    </Card>
  )
}

export function Disclaimer({ text }) {
  return (
    <div className="disclaimer">
      <ShieldCheck size={15} style={{ flex: 'none', color: '#22d3ee' }} />
      {text}
    </div>
  )
}
