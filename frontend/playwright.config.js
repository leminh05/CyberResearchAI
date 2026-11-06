import { defineConfig } from "@playwright/test";

const python = process.env.E2E_PYTHON ?? (process.platform === "win32" ? "..\\.venv\\Scripts\\python" :"../.venv/bin/python");

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:8765",
    // An installed browser, so no browser download is needed. "msedge" ships with Windows.
    channel: process.env.E2E_BROWSER_CHANNEL ?? "msedge",
  },
  webServer: {
    command: `${python} ../tests/e2e_server.py`,
    url: "http://127.0.0.1:8765/api/health",
    timeout: 180_000,
    reuseExistingServer: false,
  },
});
