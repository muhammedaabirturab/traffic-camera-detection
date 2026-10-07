export const pct = (v, digits = 0) => (v == null ? '—' : `${(v * 100).toFixed(digits)}%`)

export const secs = (v) => (v == null ? '—' : v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(2)} s`)

export const timecode = (t) => {
  if (t == null) return '—'
  const m = Math.floor(t / 60)
  const s = (t % 60).toFixed(1).padStart(4, '0')
  return `${String(m).padStart(2, '0')}:${s}`
}

export const dateTime = (iso) => {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString(undefined, { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export const label = (s) => (s || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

export const RULE_NAMES = {
  NO_HELMET: 'Rider Without Helmet',
  TRIPLE_RIDING: 'Multiple Riding',
  RED_LIGHT_JUMP: 'Red-Light Crossing',
  NO_SEATBELT: 'No Seat Belt',
  LANE_INDISCIPLINE: 'Lane Indiscipline',
  MOBILE_PHONE: 'Phone Use',
}

export const ruleName = (id) => RULE_NAMES[id] || label(id)

// Colour per canonical object label (mirrors app/utils/visualization.py)
export const CLASS_COLORS = {
  motorcycle: '#00C8FF',
  bicycle: '#78E6FF',
  car: '#3C8CFF',
  bus: '#FF78D2',
  truck: '#E66EB4',
  person: '#E1E1E1',
  rider: '#96FFC8',
  helmet: '#50DC6E',
  no_helmet: '#FF8C00',
  traffic_light: '#FFDC00',
  number_plate: '#FFFFFF',
}

export const VEHICLES = ['motorcycle', 'bicycle', 'car', 'bus', 'truck']

export const bandOf = (v, high = 0.85, medium = 0.65) => (v == null ? null : v >= high ? 'high' : v >= medium ? 'medium' : 'low')
