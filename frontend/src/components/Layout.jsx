import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { BookOpenText, Cpu, History, Info, LayoutDashboard, ScanSearch, Scale, ShieldCheck } from 'lucide-react'
import { useStatus } from '../lib/status.jsx'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/analyze', label: 'Analyze', icon: ScanSearch },
  { to: '/history', label: 'Detection History', icon: History },
  { to: '/model', label: 'Model', icon: Cpu },
  { to: '/rules', label: 'Traffic Rules', icon: Scale },
  { to: '/about', label: 'About', icon: Info },
]

function Clock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return (
    <div className="clock">
      {now.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' })} ·{' '}
      {now.toLocaleTimeString(undefined, { hour12: false })}
    </div>
  )
}

function DetectorRow({ name, state }) {
  const cls = state == null ? 'off' : state ? 'ok' : 'warn'
  return (
    <div className="sys-row">
      <span>
        <span className={`dot ${cls}`} />
        {name}
      </span>
      <span className="mono" style={{ fontSize: 11 }}>
        {state == null ? '—' : state ? 'READY' : 'N/A'}
      </span>
    </div>
  )
}

export default function Layout({ title, sub, children }) {
  const { health, online } = useStatus()
  const d = health?.detectors
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={22} />
          </div>
          <div>
            <div className="brand-name">
              TRAFFIC<span>GUARD</span> AI
            </div>
            <div className="brand-sub">Intelligent Traffic Violation Detection System</div>
          </div>
        </div>
        <nav className="nav">
          <div className="nav-label">Operations</div>
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}>
              <Icon size={17} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="nav-label" style={{ paddingLeft: 6 }}>System status</div>
          <div className="sys-row">
            <span>
              <span className={`dot ${online ? 'ok' : 'warn'}`} />
              API server
            </span>
            <span className="mono" style={{ fontSize: 11 }}>{online ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
          {health && !health.models_loaded && (
            <div className="sys-row">
              <span>
                <span className="dot warn" />
                Loading models…
              </span>
            </div>
          )}
          <DetectorRow name="Vehicle YOLO" state={d?.vehicle} />
          <DetectorRow name="Helmet model" state={d?.helmet} />
          <DetectorRow name="Rider model" state={d?.rider} />
          <Clock />
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            <div className="eyebrow">TrafficGuard AI · AI-Powered Indian Traffic Monitoring</div>
            <h1 className="page-title">{title}</h1>
            <div className="page-sub">{sub}</div>
          </div>
          <div className="top-chips">
            <span className="chip cyan">
              <BookOpenText size={12} /> EDUCATIONAL USE
            </span>
            <span className="chip">POSSIBLE VIOLATIONS · HUMAN VERIFICATION REQUIRED</span>
          </div>
        </header>
        <div className="content">{children}</div>
      </main>
    </div>
  )
}
