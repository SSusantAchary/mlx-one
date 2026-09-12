import { writable } from 'svelte/store';
import type { Settings } from '$lib/types/api';
export const settings = writable<Settings>({
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  api_key: '',
  reasoning: 'auto',
  reasoning_budget: -1
});
