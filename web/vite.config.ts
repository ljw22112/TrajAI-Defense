/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: proxies /api (HTTP + WebSocket) to the FastAPI server on :8000
// demo mode (GitHub Pages): static snapshot, served under /TrajAI-Defense/app/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === "demo" ? "/TrajAI-Defense/app/" : "/",
  build: { outDir: mode === "demo" ? "../app" : "dist", emptyOutDir: true, sourcemap: false },
  server: { port: 5173, proxy: { "/api": { target: "http://127.0.0.1:8000", ws: true, changeOrigin: true } } },
  test: { environment: "jsdom", globals: true, setupFiles: ["src/test/setup.ts"] },
}));
