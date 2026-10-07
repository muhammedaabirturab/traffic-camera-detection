import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the dashboard runs on :5173 and proxies API / media requests to the
// FastAPI server on :8000. In production `npm run build` writes to dist/, which the
// FastAPI server serves directly (single command demo: `python run.py`).
const backend = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': backend,
      '/media': backend,
      '/model-assets': backend,
    },
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 900 },
})
