import { writable } from 'svelte/store';
import type { RuntimeInfo } from '$lib/types/api';
export const runtime = writable<RuntimeInfo | null>(null);
