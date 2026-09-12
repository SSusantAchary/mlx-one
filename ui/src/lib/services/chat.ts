import type { Message, Settings, StreamResult } from '$lib/types/api';

export async function streamChat(
  model: string,
  messages: Message[],
  settings: Settings,
  signal: AbortSignal,
  onText: (text: string, reasoning: string) => void
): Promise<StreamResult> {
  const { api_key, ...generation } = settings;
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(api_key ? { authorization: `Bearer ${api_key}` } : {})
    },
    signal,
    body: JSON.stringify({
      model,
      messages: messages.map(({ role, content, reasoning_content }) => ({
        role,
        content,
        ...(reasoning_content ? { reasoning_content } : {})
      })),
      ...generation,
      stream: true
    })
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `Generation failed (${response.status})`);
  }
  if (!response.body) throw new Error('Streaming response has no body');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = ''; let finishReason: string | null = null; let metrics: Record<string, number> = {};
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const records = buffer.split('\n\n'); buffer = records.pop() ?? '';
    for (const record of records) {
      const data = record.split('\n').find((line) => line.startsWith('data: '))?.slice(6);
      if (!data || data === '[DONE]') continue;
      const chunk = JSON.parse(data);
      if (chunk.error?.message) throw new Error(chunk.error.message);
      const choice = chunk.choices?.[0];
      if (choice?.delta?.content || choice?.delta?.reasoning_content) {
        onText(choice.delta.content ?? '', choice.delta.reasoning_content ?? '');
      }
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      if (chunk.mlx) metrics = chunk.mlx;
    }
    if (done) break;
  }
  return { finishReason, metrics };
}
