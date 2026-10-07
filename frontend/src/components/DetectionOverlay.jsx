import { CLASS_COLORS, label as fmtLabel } from '../lib/format.js'

export const LAYERS = [
  { key: 'vehicles', name: 'Vehicles', color: '#00C8FF', labels: ['motorcycle', 'bicycle', 'car', 'bus', 'truck'] },
  { key: 'people', name: 'People / riders', color: '#E1E1E1', labels: ['person', 'rider'] },
  { key: 'helmets', name: 'Helmets', color: '#50DC6E', labels: ['helmet', 'no_helmet'] },
  { key: 'other', name: 'Signals / plates', color: '#FFDC00', labels: ['traffic_light', 'number_plate'] },
  { key: 'violations', name: 'Possible violations', color: '#FF5A4E', labels: [] },
]

const layerOf = (lab) => LAYERS.find((l) => l.labels.includes(lab))?.key || 'other'

// SVG overlay drawn in image-pixel coordinates so it scales with the <img>.
export default function DetectionOverlay({ width, height, detections = [], violations = [], layers, hovered, onHover }) {
  const fs = Math.max(11, width / 70)
  const sw = Math.max(1.5, width / 520)
  const tag = (x, y, text, color, dark = true) => {
    const w = text.length * fs * 0.62 + fs * 0.8
    const ty = Math.max(fs * 1.35, y)
    return (
      <g>
        <rect x={x} y={ty - fs * 1.35} width={w} height={fs * 1.35} fill={color} rx={fs * 0.2} />
        <text x={x + fs * 0.4} y={ty - fs * 0.38} fontSize={fs} fill={dark ? '#06101c' : '#fff'}>
          {text}
        </text>
      </g>
    )
  }
  return (
    <svg className="overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {detections.map((d) => {
        if (!layers[layerOf(d.label)]) return null
        const [x1, y1, x2, y2] = d.bbox
        const c = CLASS_COLORS[d.label] || '#ccc'
        const active = hovered === d.id
        return (
          <g key={`d${d.id}`} onMouseEnter={() => onHover?.(d.id)} onMouseLeave={() => onHover?.(null)} style={{ cursor: 'pointer' }}>
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={active ? `${c}22` : 'transparent'} stroke={c} strokeWidth={active ? sw * 2 : sw} />
            {tag(x1, y1, `${fmtLabel(d.label).toUpperCase()} ${Math.round(d.confidence * 100)}%`, c)}
          </g>
        )
      })}
      {layers.violations &&
        violations.map((v) => {
          const [x1, y1, x2, y2] = v.region
          const pad = (x2 - x1) * 0.03
          return (
            <g key={`v${v.id}`}>
              <rect
                x={x1 - pad} y={y1 - pad} width={x2 - x1 + 2 * pad} height={y2 - y1 + 2 * pad}
                fill="rgba(255,90,78,0.08)" stroke="#FF5A4E" strokeWidth={sw * 2.2} strokeDasharray={`${sw * 6} ${sw * 3}`}
              />
              {tag(x1 - pad, Math.min(height, y2 + pad + fs * 1.4), `⚠ ${v.title.toUpperCase()} ${Math.round(v.confidence * 100)}%`, '#FF5A4E', false)}
            </g>
          )
        })}
    </svg>
  )
}
