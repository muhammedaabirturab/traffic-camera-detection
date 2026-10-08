import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the dashboard talks to the FastAPI backend through this proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/media": "http://127.0.0.1:8000",
      "/figures": "http://127.0.0.1:8000",
    },
  },
});
