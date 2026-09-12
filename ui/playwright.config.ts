import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 20_000,
  use: {
    baseURL: 'http://127.0.0.1:8080',
    trace: 'retain-on-failure'
  },
  webServer: {
    command: '../.venv/bin/python ../tests/e2e_fake_server.py',
    url: 'http://127.0.0.1:8080/health',
    reuseExistingServer: !process.env.CI,
    timeout: 20_000
  }
});
