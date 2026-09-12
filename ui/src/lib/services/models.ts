import type { ModelInfo } from '$lib/types/api';

export async function fetchModels(apiKey = ''): Promise<ModelInfo[]> {
  const response = await fetch('/v1/models', {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {}
  });
  if (!response.ok) throw new Error(`Models request failed (${response.status})`);
  return (await response.json()).data as ModelInfo[];
}
