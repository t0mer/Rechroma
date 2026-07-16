import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Built assets are emitted into the FastAPI static dir (app/webui/dist), which
// is served with StaticFiles(html=True). `base: "./"` keeps asset URLs relative
// so they resolve when served from the app root.
export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  build: {
    outDir: "../app/webui/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/healthz": "http://localhost:8000",
    },
  },
});
