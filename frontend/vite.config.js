import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  // In development the API stays same-origin too, so the backend needs no CORS.
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: {
    environment: "jsdom",
    setupFiles: "./src/setupTests.js",
    include: ["src/**/*.test.jsx"],
  },
});
