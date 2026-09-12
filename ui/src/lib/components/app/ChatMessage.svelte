<script lang="ts">
  import { browser } from '$app/environment';
  import { Check, Copy } from '@lucide/svelte';
  import DOMPurify from 'dompurify';
  import { marked } from 'marked';
  import type { Message } from '$lib/types/api';
  let { message } = $props<{ message: Message }>();
  let copied = $state(false);
  let html = $derived(browser ? DOMPurify.sanitize(marked.parse(message.content) as string) : '');
  async function copy() { await navigator.clipboard.writeText(message.content); copied = true; setTimeout(() => copied = false, 1200); }
</script>

<article class:assistant={message.role === 'assistant'} aria-label={`${message.role} message`}>
  <div class="role">{message.role === 'assistant' ? 'mlx-one' : message.role}</div>
  <div class="bubble">
    {#if message.role === 'assistant'}<div class="markdown">{@html html}</div>{:else}<p>{message.content}</p>{/if}
    {#if message.content}<button class="copy" onclick={copy} aria-label="Copy message">{#if copied}<Check size={15} />{:else}<Copy size={15} />{/if}</button>{/if}
  </div>
</article>

<style>
  article { max-width:52rem; margin:0 auto 1.5rem; }
  .role { color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin:0 0 .45rem .15rem; }
  .bubble { position:relative; border:1px solid var(--line); background:var(--panel); border-radius:1rem; padding:.85rem 2.7rem .85rem 1rem; box-shadow:0 8px 28px rgb(0 0 0 / 4%); }
  article:not(.assistant) .bubble { background:var(--accent2); border-color:transparent; }
  p { white-space:pre-wrap; margin:0; line-height:1.6; }
  .copy { position:absolute; right:.65rem; top:.65rem; border:0; background:transparent; color:var(--muted); padding:.25rem; }
</style>
