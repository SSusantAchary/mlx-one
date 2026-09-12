export type Role = 'system' | 'user' | 'assistant';
export interface Message { id: string; role: Role; content: string; }
export interface ModelInfo { id: string; object: 'model'; owned_by: string; mlx: { architecture: string; parameters: number | null; quantization: Record<string, unknown> | null; context_length: number; revision: string | null; }; }
export interface RuntimeInfo { backend: string; device: string; chip?: string | null; model?: ModelInfo['mlx'] | null; prompt_tokens?: number; generated_tokens?: number; tokens_per_second?: number; time_to_first_token_ms?: number; generation_latency_ms?: number; context_used?: number; memory_used_bytes?: number; memory_peak_bytes?: number; }
export interface Settings { temperature: number; top_p: number; max_tokens: number; }
export interface StreamResult { finishReason: string | null; metrics: Record<string, number>; }
