import { Route, Routes, useLocation } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import { StatusProvider } from './lib/status.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Analyze from './pages/Analyze.jsx'
import History from './pages/History.jsx'
import HistoryDetail from './pages/HistoryDetail.jsx'
import ModelPage from './pages/ModelPage.jsx'
import RulesPage from './pages/RulesPage.jsx'
import About from './pages/About.jsx'

const PAGES = {
  '/': { title: 'Command Dashboard', sub: 'Overview of analyses, detections and possible violations' },
  '/analyze': { title: 'Analyze Footage', sub: 'Upload a traffic-camera image or video for YOLO-based analysis' },
  '/history': { title: 'Detection History', sub: 'Every analysis is stored locally and can be reopened' },
  '/model': { title: 'Model Information', sub: 'Detectors, classes and measured performance — read from disk, never hard-coded' },
  '/rules': { title: 'Traffic Rules', sub: 'Configurable rule database (app/rules/traffic_rules.json)' },
  '/about': { title: 'About the System', sub: 'How TrafficGuard AI reasons about traffic scenes' },
}

export default function App() {
  const { pathname } = useLocation()
  const key = pathname.startsWith('/history/') ? '/history' : pathname
  const page = PAGES[key] || PAGES['/']
  return (
    <StatusProvider>
      <Layout title={pathname.startsWith('/history/') ? 'Analysis Record' : page.title} sub={page.sub}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/history" element={<History />} />
          <Route path="/history/:id" element={<HistoryDetail />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/rules" element={<RulesPage />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </Layout>
    </StatusProvider>
  )
}
