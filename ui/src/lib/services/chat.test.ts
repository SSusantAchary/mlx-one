import { describe, expect, it, vi } from 'vitest';
import { streamChat } from './chat';

describe('streamChat', () => {
  it('parses incremental OpenAI SSE chunks and terminal metrics', async () => {
    const body = [
      'data: {"choices":[{"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n',
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"mlx":{"generated_tokens":1}}\n\n',
      'data: [DONE]\n\n'
    ].join('');
    vi.stubGlobal('fetch', vi.fn(async () => new Response(body, { status: 200 })));
    let text = '';
    const result = await streamChat(
      'model', [{ id: '1', role: 'user', content: 'hello' }],
      { temperature: 0.7, top_p: 0.9, max_tokens: 8, api_key: '', reasoning: 'auto', reasoning_budget: -1 }, new AbortController().signal,
      (piece) => text += piece
    );
    expect(text).toBe('Hi');
    expect(result.finishReason).toBe('stop');
    expect(result.metrics.generated_tokens).toBe(1);
  });

  it('surfaces HTTP and in-stream API errors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ error: { message: 'queue full' } }), { status: 429 }
    )));
    const invoke = () => streamChat(
      'model', [{ id: '1', role: 'user', content: 'hello' }],
      { temperature: 0.7, top_p: 0.9, max_tokens: 8, api_key: '', reasoning: 'auto', reasoning_budget: -1 }, new AbortController().signal,
      () => undefined
    );
    await expect(invoke()).rejects.toThrow('queue full');

    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      'data: {"error":{"message":"decode failed"}}\n\ndata: [DONE]\n\n', { status: 200 }
    )));
    await expect(invoke()).rejects.toThrow('decode failed');
  });
});
