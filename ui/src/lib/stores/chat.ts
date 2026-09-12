import { writable } from 'svelte/store';
import type { Message } from '$lib/types/api';

export const messages = writable<Message[]>([]);
export const streaming = writable(false);
export const error = writable<string | null>(null);
