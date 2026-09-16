export interface TranscriptionResponse {
  text: string;
  language?: string;
  mlx?: Record<string, unknown>;
}

export async function transcribeAudio(
  model: string,
  audio: File,
  apiKey: string,
  signal: AbortSignal
): Promise<TranscriptionResponse> {
  const form = new FormData();
  form.set('model', model);
  form.set('file', audio, audio.name);
  form.set('task', 'transcribe');
  const response = await fetch('/v1/audio/transcriptions', {
    method: 'POST',
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {},
    signal,
    body: form
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `Transcription failed (${response.status})`);
  }
  return response.json();
}
