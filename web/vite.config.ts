import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: `npm run dev` serves on :5173 and proxies /api to the FastAPI backend
// (run `ndiff-web` or uvicorn on :8000).  Build: emits the SPA into the Python
// package so the installed wheel can serve it.
export default defineConfig({
  plugins: [react()],
  // Project Pages serve under /<repo>/.  Set GH_PAGES=1 in the Pages build so
  // assets resolve under the subpath; local dev and self-hosted builds stay at /.
  // See docs/github-pages-webgpu-plan.md for the full static-data migration.
  base: process.env.GH_PAGES ? "/neutron-diffuse/" : "/",
  build: {
    outDir: "../src/ndiff/server/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
