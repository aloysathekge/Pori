import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// Harvested from Hermes web/vite.config.ts (MIT) — the Hermes dashboard-token
// plugin, /api proxy, and xterm/TUI wiring were stripped. Aloy surfaces talk to
// the backend over plain REST + SSE via @aloy/shared.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@aloy/shared": path.resolve(__dirname, "../shared/src"),
    },
  },
});
