// Thin fetch wrapper. All backend errors arrive as {status:"error", message}; we surface only that message.
const OFFLINE =
  "The analysis backend is not reachable. Start it with `python -m app.main` (see README) and reload this page.";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error(OFFLINE);
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON response */
  }
  if (!res.ok) {
    // A dev proxy returns 5xx with an empty body when the backend is down.
    const fallback = res.status >= 500 && !body ? OFFLINE : `Request failed (${res.status}).`;
    throw new Error(body?.message || fallback);
  }
  return body;
}

export const api = {
  health: () => request("/api/health"),
  rules: () => request("/api/rules"),
  modelInfo: () => request("/api/model/info"),
  history: (limit = 100) => request(`/api/history?limit=${limit}`),
  analysis: (id) => request(`/api/history/${id}`),
  deleteAnalysis: (id) => request(`/api/history/${id}`, { method: "DELETE" }),
  job: (id) => request(`/api/jobs/${id}`),
  analyzeImage(file) {
    const fd = new FormData();
    fd.append("file", file);
    return request("/api/analyze/image", { method: "POST", body: fd });
  },
  analyzeVideo(file, opts = {}) {
    const fd = new FormData();
    fd.append("file", file);
    Object.entries(opts).forEach(([k, v]) => v !== "" && v != null && fd.append(k, v));
    return request("/api/analyze/video", { method: "POST", body: fd });
  },
};

export const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
export const fmtDate = (iso) => {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};
export const levelOf = (c, bands = { high: 0.85, medium: 0.65 }) => (c >= bands.high ? "high" : c >= bands.medium ? "medium" : "low");
