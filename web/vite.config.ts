import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173 },
  build: {
    // lightweight-charts is most of the bundle; keep it in its own chunk so
    // an app-code change does not invalidate the cached chart library.
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ["lightweight-charts"],
          supabase: ["@supabase/supabase-js"],
        },
      },
    },
  },
});
