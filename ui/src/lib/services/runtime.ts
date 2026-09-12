import type { RuntimeInfo } from '$lib/types/api';

export async function fetchRuntime(apiKey = ''): Promise<RuntimeInfo> {
  const response = await fetch('/v1/runtime', {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {}
  });
  if (!response.ok) throw new Error(`Runtime request failed (${response.status})`);
  return response.json() as Promise<RuntimeInfo>;
}
