import { writable } from 'svelte/store';
import type { ModelInfo } from '$lib/types/api';
export const models = writable<ModelInfo[]>([]);
export const selectedModel = writable('');
