import { cleanup, fireEvent, render, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ChatForm from './ChatForm.svelte';
import ChatMessage from './ChatMessage.svelte';
import RuntimeStatus from './RuntimeStatus.svelte';

describe('chat components', () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
  const formProps = (busy: boolean, submit = vi.fn(), stop = vi.fn()) => ({
    busy,
    transcribing: false,
    transcriptionEnabled: false,
    model: 'test/model',
    onsubmit: submit,
    onstop: stop,
    onaudio: vi.fn(),
    oncanceltranscription: vi.fn(),
    opensettings: vi.fn(),
    opensystem: vi.fn(),
    onerror: vi.fn()
  });

  it('submits Enter, preserves Shift+Enter, and exposes Stop while streaming', async () => {
    const submit = vi.fn();
    const stop = vi.fn();
    const form = render(ChatForm, formProps(false, submit, stop));
    const input = form.getByLabelText('Message');
    await fireEvent.input(input, { target: { value: 'hello' } });
    await fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(submit).not.toHaveBeenCalled();
    await fireEvent.keyDown(input, { key: 'Enter' });
    expect(submit).toHaveBeenCalledWith('hello');

    form.unmount();
    const active = render(ChatForm, formProps(true, submit, stop));
    await fireEvent.click(active.getByLabelText('Stop generation'));
    expect(stop).toHaveBeenCalledOnce();
  });

  it('sanitizes streamed Markdown HTML', () => {
    const view = render(ChatMessage, {
      message: {
        id: 'assistant', role: 'assistant',
        content: '<img src="x" onerror="alert(1)"> **safe**'
      }
    });
    const image = view.container.querySelector('img');
    expect(image).not.toBeNull();
    expect(image?.hasAttribute('onerror')).toBe(false);
    expect(view.container.querySelector('strong')?.textContent).toBe('safe');
  });

  it('records microphone audio and hands it to transcription without submitting', async () => {
    const audio = vi.fn();
    const submit = vi.fn();
    const track = { stop: vi.fn() };
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [track] })) }
    });
    class FakeRecorder {
      static isTypeSupported() { return true; }
      state = 'inactive';
      mimeType = 'audio/webm;codecs=opus';
      ondataavailable: ((event: { data: Blob }) => void) | null = null;
      onstop: (() => void) | null = null;
      constructor(_stream: MediaStream, _options?: MediaRecorderOptions) {}
      start() { this.state = 'recording'; }
      stop() {
        this.state = 'inactive';
        this.ondataavailable?.({ data: new Blob(['voice'], { type: this.mimeType }) });
        this.onstop?.();
      }
    }
    vi.stubGlobal('MediaRecorder', FakeRecorder);
    const form = render(ChatForm, {
      ...formProps(false, submit), transcriptionEnabled: true, onaudio: audio
    });
    await fireEvent.click(form.getByLabelText('Record audio'));
    await fireEvent.click(await form.findByText('Use recording'));
    await waitFor(() => expect(audio).toHaveBeenCalledOnce());
    expect(audio.mock.calls[0][0]).toBeInstanceOf(File);
    expect(submit).not.toHaveBeenCalled();
    expect(track.stop).toHaveBeenCalledOnce();
  });

  it('shows scheduler, cache, and MTP runtime statistics', () => {
    const view = render(RuntimeStatus, { props: {
      context: 4096,
      runtime: {
        backend: 'mlx', device: 'gpu', prompt_cache_hit: true,
        reused_prompt_tokens: 256, drafted_tokens: 4, accepted_draft_tokens: 3,
        scheduler: { active_slots: 1, parallel_slots: 2, queue_depth: 3, queue_capacity: 8 },
        server: { cache_type_k: 'q4_0', cache_type_v: 'f16' }
      }
    }});
    expect(view.getByText('slots 1/2')).toBeTruthy();
    expect(view.getByText('prefix 256 hit')).toBeTruthy();
    expect(view.getByText('MTP 3/4')).toBeTruthy();
  });
});
