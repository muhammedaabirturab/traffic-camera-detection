// Thin wrapper around the TrafficGuard FastAPI backend.

async function request(path, options = {}) {
  const res = await fetch(path, options)
  let body = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON response */
  }
  if (!res.ok) {
    const detail = body?.detail
    const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : res.statusText
    throw new Error(msg || `Request failed (${res.status})`)
  }
  return body
}

export const api = {
  health: () => request('/api/health'),
  settings: () => request('/api/settings'),
  stats: () => request('/api/stats'),
  rules: () => request('/api/rules'),
  modelInfo: () => request('/api/model/info'),
  reloadModels: () => request('/api/model/reload', { method: 'POST' }),
  history: (kind) => request(`/api/history${kind ? `?kind=${kind}` : ''}`),
  historyItem: (id) => request(`/api/history/${id}`),
  deleteHistory: (id) => request(`/api/history/${id}`, { method: 'DELETE' }),
  job: (id) => request(`/api/jobs/${id}`),

  analyzeImage(file) {
    const fd = new FormData()
    fd.append('file', file)
    return request('/api/analyze/image', { method: 'POST', body: fd })
  },

  analyzeVideo(file, { stopLine = null, direction = 'any' } = {}) {
    const fd = new FormData()
    fd.append('file', file)
    if (stopLine != null) fd.append('stop_line', String(stopLine))
    fd.append('line_direction', direction)
    return request('/api/analyze/video', { method: 'POST', body: fd })
  },

  // Poll a background job until it finishes; calls onProgress(job) on every update.
  async waitForJob(jobId, onProgress, intervalMs = 1000) {
    for (;;) {
      const job = await api.job(jobId)
      onProgress?.(job)
      if (job.status === 'completed') return job
      if (job.status === 'failed') throw new Error(job.error || 'Video analysis failed')
      await new Promise((r) => setTimeout(r, intervalMs))
    }
  },
}
