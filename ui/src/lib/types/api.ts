export type Role = 'system' | 'user' | 'assistant';
export interface Message { id: string; role: Role; content: string; reasoning_content?: string; }
export interface ModelInfo { id: string; object: 'model'; owned_by: string; mlx: { architecture: string; parameters: number | null; quantization: Record<string, unknown> | null; context_length: number; revision: string | null; tasks?: string[]; modalities?: string[]; }; }
export interface RuntimeInfo {
  backend: string;
  device: string;
  chip?: string | null;
  model?: ModelInfo['mlx'] | null;
  prompt_tokens?: number;
  generated_tokens?: number;
  tokens_per_second?: number;
  time_to_first_token_ms?: number;
  generation_latency_ms?: number;
  context_used?: number;
  memory_used_bytes?: number;
  memory_peak_bytes?: number;
  cache_bytes?: number;
  prompt_cache_hit?: boolean;
  reused_prompt_tokens?: number;
  drafted_tokens?: number;
  accepted_draft_tokens?: number;
  draft_acceptance_ratio?: number;
  scheduler?: { active_slots: number; parallel_slots: number; queue_depth: number; queue_capacity: number };
  server?: { cache_type_k?: string; cache_type_v?: string; max_batch_tokens?: number; prefill_chunk_size?: number };
}
export interface Settings { temperature: number; top_p: number; max_tokens: number; api_key: string; reasoning: 'auto' | 'on' | 'off'; reasoning_budget: number; }
export interface StreamResult { finishReason: string | null; metrics: Record<string, number>; }
