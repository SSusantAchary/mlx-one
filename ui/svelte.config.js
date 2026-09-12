import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: '../src/mlx_one/ui/dist',
      assets: '../src/mlx_one/ui/dist',
      fallback: 'index.html',
      strict: true
    }),
    paths: { relative: true }
  }
};
