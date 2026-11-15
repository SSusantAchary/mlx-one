<script lang="ts">
  import { browser } from '$app/environment';
  import { Check, Copy, Pencil, GitBranch, RotateCcw, Trash2, Lightbulb, Clock3, Gauge, TextCursorInput } from '@lucide/svelte';
  import DOMPurify from 'dompurify';
  import { marked } from 'marked';
  import ModelBadge from './ModelBadge.svelte';
  import type { Message } from '$lib/types/api';
  let { message, inputMetrics, busy = false, onedit, onbranch, ondelete, onregenerate, onretry } = $props<{
    message: Message; inputMetrics?: Message['metrics']; busy?: boolean;
    onedit?: (id: string, text: string) => void; onbranch?: (id: string) => void;
    ondelete?: (id: string) => void; onregenerate?: (id: string) => void; onretry?: (id: string) => void;
  }>();
  let copied = $state(false), editing = $state(false), edited = $state(''), copyError = $state('');
  const content = $derived(message.content);
  let html = $derived(browser ? DOMPurify.sanitize(marked.parse(content) as string) : '');
  const metrics = $derived(message.role === 'assistant' ? message.metrics : inputMetrics);
  const tokens = $derived(message.role === 'assistant' ? metrics?.generated_tokens : metrics?.actual_prefill_tokens ?? metrics?.prompt_tokens);
  const duration = $derived(message.role === 'assistant' ? metrics?.decode_ms : metrics?.prefill_ms);
  const rate = $derived(message.role === 'assistant' ? metrics?.decode_tokens_per_second : metrics?.prompt_tokens_per_second);
  async function copy() { try { await navigator.clipboard.writeText(message.content); copied = true; copyError = ''; setTimeout(() => copied = false, 1200); } catch { copyError = 'Copy unavailable. Select the text to copy it.'; } }
  function edit() { edited = message.content; editing = true; }
</script>
<article class:assistant={message.role === 'assistant'} aria-label={`${message.role} message`}>
  <div class="bubble">
    {#if editing}
      <textarea class="edit" bind:value={edited} aria-label="Edit message" rows="5"></textarea>
      <div class="edit-actions"><button onclick={() => editing = false}>Cancel</button><button disabled={busy || !edited.trim()} onclick={() => { onedit?.(message.id, edited.trim()); editing = false; }}>Save{message.role === 'user' ? ' & send' : ''}</button></div>
    {:else}
      {#if message.reasoning_content}<details><summary><Lightbulb size={16}/> Reasoning</summary><pre>{message.reasoning_content}</pre></details>{/if}
      {#if message.role === 'assistant'}<div class="markdown">{@html html}</div>{:else}<p>{message.content}</p>{/if}
    {/if}
  </div>
  {#if message.role === 'assistant' && message.streaming}
    <div class="generation-progress" role="status"><span class="pulse"></span>
      {message.progress?.phase === 'prefill' ? 'Processing prompt…' : message.progress?.phase === 'reasoning' ? 'Thinking…' : 'Generating…'}
      {#if message.progress?.generated_tokens}<span>{message.progress.generated_tokens.toLocaleString()} tokens generated{message.progress.reasoning_tokens ? ` · ${message.progress.reasoning_tokens} reasoning` : ''}</span>{/if}
    </div>
  {/if}
  {#if message.finish_reason === 'length'}
    <div class="completion-notice" role="status">
      <p>{message.content.trim() ? 'Output limit reached. This answer may be incomplete.' : 'Output limit reached before an answer was produced. Reasoning uses the same token budget.'}</p>
      {#if message.metrics?.reasoning_tokens}<small>{message.metrics.reasoning_tokens} reasoning tokens used.</small>{/if}
      {#if onretry}<button disabled={busy} onclick={() => onretry?.(message.id)}>Retry with more tokens</button>{/if}
    </div>
  {:else if message.finish_reason === 'cancelled'}<p class="generation-progress" role="status">Generation stopped. Partial output is kept.</p>
  {:else if message.finish_reason === 'error'}<p class="completion-notice" role="status">Response interrupted. Retry to continue.</p>{/if}
  {#if metrics || message.model}
    <div class="turn-metrics" aria-label="Message metrics">
      {#if message.model}<ModelBadge model={message.model} info={message.model_info} />{/if}
      {#if tokens != null}<span><TextCursorInput size={13}/>{tokens.toLocaleString()} tokens</span>{/if}
      {#if duration != null}<span title="Model-forward time"><Clock3 size={13}/>{(duration/1000).toFixed(2)}s forward</span>{/if}
      {#if rate != null}<span><Gauge size={13}/>{rate.toFixed(2)} tok/s</span>{/if}
      {#if message.role === 'assistant' && metrics?.time_to_first_token_ms != null}<span title="First model token; visible text may arrive later">TTFT {metrics.time_to_first_token_ms.toFixed(0)} ms</span>{/if}
    </div>
  {/if}
  <div class="message-actions">
    {#if message.content}<button onclick={copy} aria-label="Copy message" title="Copy">{#if copied}<Check size={17}/>{:else}<Copy size={17}/>{/if}</button>{/if}
    {#if onedit}<button onclick={edit} disabled={busy} aria-label="Edit message" title="Edit"><Pencil size={17}/></button>{/if}
    {#if onregenerate && message.role === 'assistant'}<button onclick={() => onregenerate?.(message.id)} disabled={busy} aria-label="Regenerate response" title="Regenerate"><RotateCcw size={17}/></button>{/if}
    {#if onbranch}<button onclick={() => onbranch?.(message.id)} disabled={busy} aria-label="Branch conversation" title="Branch into a new conversation"><GitBranch size={17}/></button>{/if}
    {#if ondelete}<button onclick={() => ondelete?.(message.id)} disabled={busy} aria-label="Delete from here" title="Delete this message and following messages"><Trash2 size={17}/></button>{/if}
  </div>
  {#if copyError}<small role="status">{copyError}</small>{/if}
</article>
<style>
  article{max-width:56rem;margin:0 auto 2.7rem;min-width:0}.bubble{position:relative;line-height:1.7}.assistant .bubble{padding:.2rem 0}article:not(.assistant){display:flex;flex-direction:column;align-items:flex-end}article:not(.assistant) .bubble{max-width:85%;border-radius:1.4rem;background:var(--hover);padding:.65rem 1rem}p{white-space:pre-wrap;margin:0;line-height:1.65}.turn-metrics{display:flex;align-items:center;flex-wrap:wrap;gap:.65rem;margin-top:1rem;color:var(--muted);font-size:.7rem}.turn-metrics>span{display:flex;align-items:center;gap:.3rem}.message-actions{display:flex;gap:.6rem;margin-top:.8rem}.message-actions button{display:grid;place-items:center;border:0;background:none;color:var(--muted);padding:.3rem;border-radius:.4rem}.message-actions button:hover{color:var(--ink);background:var(--hover)}button:disabled{opacity:.35;cursor:not-allowed}details{margin:0 0 1.2rem;color:var(--muted)}summary{display:flex;align-items:center;gap:.5rem;cursor:pointer;list-style:none}details pre{white-space:pre-wrap;font:inherit;font-size:.85rem;line-height:1.7;padding:.6rem 0 0 1.5rem;max-height:22rem;overflow:auto}.edit{min-width:min(35rem,65vw);width:100%;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:.8rem;padding:.7rem}.edit-actions{display:flex;justify-content:flex-end;gap:.5rem;margin-top:.5rem}.edit-actions button{background:var(--hover);color:var(--ink);border:1px solid var(--line);border-radius:.5rem;padding:.4rem .8rem}small{color:var(--muted)}
  .generation-progress{display:flex;align-items:center;flex-wrap:wrap;gap:.5rem;color:var(--muted);font-size:.78rem;margin:.8rem 0}.generation-progress>span:last-child{font-size:.7rem}.pulse{height:7px;width:7px;flex-shrink:0;border-radius:50%;background:var(--accent);animation:breathe 1.2s infinite}.completion-notice{border:1px solid var(--line);background:var(--hover);border-radius:.7rem;padding:.8rem;margin-top:.8rem;font-size:.8rem;color:var(--muted)}.completion-notice small{display:block;margin-top:.3rem}.completion-notice button{display:block;border:1px solid var(--line);border-radius:.5rem;background:var(--panel);color:var(--ink);padding:.45rem .7rem;margin-top:.6rem}@keyframes breathe{50%{opacity:.25}}
</style>
