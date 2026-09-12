import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

const backend = process.env.MLX_ONE_SERVER_ORIGIN ?? 'http://127.0.0.1:8080';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  resolve: { conditions: ['browser'] },
  server: {
    proxy: { '/health': backend, '/v1': backend }
  },
  test: { environment: 'jsdom', include: ['src/**/*.test.ts'] }
});
