import { fireEvent, render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import ChatForm from './ChatForm.svelte';
import ChatMessage from './ChatMessage.svelte';

describe('chat components', () => {
  it('submits Enter, preserves Shift+Enter, and exposes Stop while streaming', async () => {
    const submit = vi.fn();
    const stop = vi.fn();
    const form = render(ChatForm, { busy: false, onsubmit: submit, onstop: stop });
    const input = form.getByLabelText('Message');
    await fireEvent.input(input, { target: { value: 'hello' } });
    await fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(submit).not.toHaveBeenCalled();
    await fireEvent.keyDown(input, { key: 'Enter' });
    expect(submit).toHaveBeenCalledWith('hello');

    form.unmount();
    const active = render(ChatForm, { busy: true, onsubmit: submit, onstop: stop });
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
});
